#!/usr/bin/env python3
"""Train the full 34-state classifier — the original proposed Improvement #2.

This is the harder version of train_stage_classifier.py: 34 labels instead of 5.
Stage-c alone has 22 within-stage states. The flat 34-way classification is
the most direct replacement for the LLM consultant's state prediction.

Outputs: results/state_classifier_v1/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

DEFAULT_MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEFAULT_OUT_DIR = Path("results/state_classifier_v1")

# Per-arch LoRA target_modules. The resolver matches on `config.model_type`.
# Validated against actual named_modules() inspection on each backbone.
LORA_TARGETS_BY_MODEL_TYPE: dict[str, list[str]] = {
    "bert": ["query", "value"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # qwen3_5 is a hybrid Mamba+attention arch: 6 full-attn blocks (q/k/v/o_proj)
    # interleaved with 18 linear-attn blocks (in_proj_qkv, out_proj). Covering
    # both keeps LoRA uniform across layer types. Note: AutoModelForSequenceClassification
    # resolves this multimodal checkpoint to its text sub-config, exposing
    # model_type='qwen3_5_text' at runtime (rather than the outer 'qwen3_5'),
    # so both keys are registered.
    "qwen3_5": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
    "qwen3_5_text": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
}

# Full 34 states per SocRule
ALL_STATES = (
    ["a0", "a1"]
    + [f"b{i}" for i in range(2, 8)]
    + [f"c{i}" for i in range(8, 30)]
    + [f"d{i}" for i in range(30, 34)]
    + ["e34"]
)
STATE_TO_LABEL = {s: i for i, s in enumerate(ALL_STATES)}


def build_examples(split: str) -> list[dict]:
    from src.project.kele import load_dataset

    data = load_dataset(split=split)
    examples: list[dict] = []
    for dlg in data:
        history_lines: list[str] = []
        for turn in dlg.get("dialogue", []):
            student = turn.get("student", "").strip()
            state = turn.get("state", "")
            if state not in STATE_TO_LABEL:
                continue
            current_input = f"学生: {student}"
            history_text = "\n".join(history_lines + [current_input])
            examples.append(
                {
                    "text": history_text[-4000:],
                    "label": STATE_TO_LABEL[state],
                    "state": state,
                }
            )
            history_lines.append(f"学生: {student}")
            teacher = turn.get("teacher", "").strip()
            if teacher:
                history_lines.append(f"老师: {teacher}")
    return examples


HEAD_NAMES = ("classifier", "score")  # SeqClassification head attribute names across families


def _freeze_backbone(model: torch.nn.Module) -> None:
    """Set requires_grad=False on every parameter except the classification head.
    The head is identified by attribute name (classifier or score) and re-enabled."""
    head_param_ids: set[int] = set()
    for head_name in HEAD_NAMES:
        head = getattr(model, head_name, None)
        if isinstance(head, torch.nn.Module):
            head_param_ids.update(id(p) for p in head.parameters())
    if not head_param_ids:
        raise RuntimeError(
            "Could not locate a classification head (looked for: "
            f"{HEAD_NAMES}); refusing to freeze the entire model."
        )
    n_frozen = 0
    n_train = 0
    for p in model.parameters():
        if id(p) in head_param_ids:
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    print(
        f"  frozen backbone: {n_frozen / 1e6:.1f}M params frozen, "
        f"{n_train / 1e6:.3f}M trainable (head only)"
    )


def _force_correct_head(model: torch.nn.Module, num_labels: int) -> None:
    """Ensure the classification head is the right shape AND has finite weights.

    Two failure modes seen in transformers 5.8.1:
    1. Qwen3_5ForSequenceClassification ignores num_labels= and ships a binary head.
    2. Qwen3ForSequenceClassification's missing-from-checkpoint `score.weight`
       is initialized to ALL NaN in the default from_pretrained code path (the
       accelerate-based `low_cpu_mem_usage=True` path Kaiming-inits it properly).
       Without this guard, the head is NaN at training start → NaN loss → 0%.
    Replace the head when shape mismatches OR when its weights contain NaN/Inf.
    """
    import math

    for head_name in ("classifier", "score"):
        head = getattr(model, head_name, None)
        if not isinstance(head, torch.nn.Linear):
            continue
        shape_wrong = head.out_features != num_labels
        weights_bad = torch.isnan(head.weight).any().item() or torch.isinf(head.weight).any().item()
        if not (shape_wrong or weights_bad):
            return
        new_head = torch.nn.Linear(
            head.in_features,
            num_labels,
            bias=head.bias is not None,
            dtype=head.weight.dtype,
            device=head.weight.device,
        )
        torch.nn.init.kaiming_uniform_(new_head.weight, a=math.sqrt(5))
        if new_head.bias is not None:
            torch.nn.init.zeros_(new_head.bias)
        setattr(model, head_name, new_head)
        model.config.num_labels = num_labels
        # Restore the canonical id2label/label2id. Qwen3_5ForSequenceClassification's
        # original 2-class head leaves model.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
        # which then persists through from_pretrained(id2label=...) and reaches the
        # on-disk config.json after save_pretrained. The classifier still learns the
        # right indices, but the saved string-label mapping is wrong → kele.py's
        # bert-consultant emits "LABEL_17" instead of "c17" → all string-vs-string
        # comparisons fail → state_acc collapses to ~0%. Fix here so future saves
        # carry the right mapping; broken T3/T4 configs were patched in-place separately.
        model.config.id2label = {i: s for i, s in enumerate(ALL_STATES)}
        model.config.label2id = {s: i for i, s in enumerate(ALL_STATES)}
        reason = []
        if shape_wrong:
            reason.append(f"shape {head.out_features}->{num_labels}")
        if weights_bad:
            reason.append("NaN/Inf weights detected")
        print(f"  head '{head_name}' reinitialized ({', '.join(reason)})")
        return


def _apply_lora(
    model: torch.nn.Module, model_type: str, r: int, alpha: int, target_modules: str | None
) -> torch.nn.Module:
    """Wrap model with a PEFT LoRA adapter. Returns the PEFT-wrapped model."""
    from peft import LoraConfig, TaskType, get_peft_model

    if target_modules is None or target_modules == "auto":
        targets = LORA_TARGETS_BY_MODEL_TYPE.get(model_type)
        if targets is None:
            raise ValueError(
                f"No auto LoRA targets registered for model_type={model_type!r}. "
                f"Pass --lora-target-modules <comma-separated names>."
            )
    else:
        targets = [m.strip() for m in target_modules.split(",") if m.strip()]

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        modules_to_save=["classifier", "score"],  # train head alongside adapter
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def _init_wandb(args: argparse.Namespace, model_type: str, resume_id: str | None = None):
    """Start a Weights & Biases run when tracking is enabled (--wandb flag or a
    WANDB_PROJECT env var). Returns the run handle, or None when disabled — every
    call site guards on None, so the no-tracking path is byte-identical to before.
    The import is local so the script runs without wandb installed. When resume_id
    is given (resuming from a checkpoint), the same run is continued so the report
    stays a single uninterrupted curve rather than a fresh run per attempt."""
    if not (args.wandb or os.environ.get("WANDB_PROJECT")):
        return None
    import wandb

    method = "lora" if args.lora else ("frozen" if args.freeze_backbone else "full-ft")
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "csen346-state-classifier"),
        name=os.environ.get("WANDB_RUN_NAME") or args.out_dir.name,
        id=resume_id,
        resume="allow" if resume_id else None,
        config={
            "model_id": args.model_id,
            "model_type": model_type,
            "method": method,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_target_modules": args.lora_target_modules,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
            "num_labels": len(ALL_STATES),
            "gradient_checkpointing": args.gradient_checkpointing,
            "bf16_autocast": args.bf16_autocast,
        },
    )


def _sync(device: str) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HF model ID for the backbone (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for checkpoints + final/ (default: %(default)s)",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classification head (linear probe). Mutually exclusive with --lora.",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute activations during backward instead of saving them. "
        "Cuts activation memory by ~4x at ~30%% wall-clock cost. "
        "Essential for LoRA on larger backbones (T4 on Qwen3.5).",
    )
    p.add_argument(
        "--bf16-autocast",
        action="store_true",
        help="Wrap forward passes in torch.autocast(dtype=bfloat16). Model "
        "stays fp32 (params + optimizer state), only forward computes "
        "in bf16. Cuts forward time/memory ~2x. Safe with manual loop: "
        "the HF Trainer issues that bit us earlier don't apply here.",
    )
    p.add_argument(
        "--lora",
        action="store_true",
        help="Wrap the backbone with a PEFT LoRA adapter; adapters are "
        "merged into the base weights before save so the artifact is "
        "a standard HF SeqClassification checkpoint.",
    )
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="auto",
        help="'auto' (per-arch defaults) or comma-separated module suffixes.",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-5 for full-FT/LoRA, "
        "1e-3 for --freeze-backbone linear-probe).",
    )
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="'auto' picks cuda when available, else cpu. 'mps' (Apple GPU) is opt-in: "
        "set PYTORCH_ENABLE_MPS_FALLBACK=1 so unsupported ops fall back to the CPU.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Benchmark/smoke cap: stop after this many optimizer steps and report "
        "steps/min + peak VRAM, skipping eval/save/publish. 0 (default) = full run.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log loss/grad_norm/lr (per step), eval acc (per epoch), and final "
        "test metrics to Weights & Biases. Also auto-enabled if WANDB_PROJECT is "
        "set. Project/run names come from WANDB_PROJECT/WANDB_RUN_NAME (defaults: "
        "'csen346-state-classifier' / the --out-dir basename). Requires `wandb` "
        "(install via the 'wandb' extra) and `wandb login` for online logging.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <out-dir>/checkpoint.pt if it exists (model + optimizer + "
        "scheduler + epoch + global_step + wandb run id), continuing the same W&B "
        "run. A checkpoint is written at every epoch boundary regardless of this flag.",
    )
    args = p.parse_args()

    if args.freeze_backbone and args.lora:
        p.error("--freeze-backbone and --lora are mutually exclusive.")
    if args.lr is None:
        args.lr = 1e-3 if args.freeze_backbone else 2e-5

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(ALL_STATES),
        id2label={i: s for s, i in STATE_TO_LABEL.items()},
        label2id=STATE_TO_LABEL,
        # The default load path can leave Qwen3ForSequenceClassification's
        # missing score.weight as all-NaN. low_cpu_mem_usage=True triggers the
        # accelerate-based load path which Kaiming-inits missing weights
        # correctly. _force_correct_head below is a belt-and-suspenders guard.
        low_cpu_mem_usage=True,
    )
    _force_correct_head(model, len(ALL_STATES))
    # Defensive: force the entire model to fp32. Memory cost is modest on a 5090
    # (~1.2 GB extra for Qwen3-Embedding, ~1.5 GB for Qwen3.5) and we get a
    # numerically stable forward pass even if AMP autocast hits an edge case in
    # a multimodal/hybrid backbone we haven't fully characterized.
    if next(model.parameters()).dtype != torch.float32:
        print(f"  casting model {next(model.parameters()).dtype} -> torch.float32 (defensive)")
        model = model.float()

    model_type = getattr(model.config, "model_type", "unknown")
    if args.freeze_backbone:
        _freeze_backbone(model)
    if args.lora:
        print(
            f"  applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}, "
            f"targets={args.lora_target_modules}) for model_type={model_type!r} ..."
        )
        model = _apply_lora(
            model, model_type, args.lora_r, args.lora_alpha, args.lora_target_modules
        )

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            # PEFT wraps the base model; call enable on the underlying transformer.
            base = getattr(model, "model", model)  # PEFT wrapper exposes .model
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_enable()
            # use_cache must be False under gradient checkpointing (incompatible)
            if hasattr(model, "config"):
                model.config.use_cache = False
            print("  gradient checkpointing enabled")
        else:
            print("  WARNING: model has no gradient_checkpointing_enable() method")

    # Decoder LMs typically lack a tokenizer pad token; set it to EOS.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Always propagate pad_token_id to model.config. Qwen3ForSequenceClassification's
    # pooling logic pulls the last-non-pad-token hidden state; if pad_token_id is None
    # it falls back to a degenerate path that yields NaN logits → NaN loss → NaN head
    # weights within epoch 1 (the T1 v1/v2 failure mode). The tokenizer already having
    # a pad_token_id is NOT sufficient; the model config needs it too.
    # transformers 5.9.0 drift: the multimodal Qwen3_5Config no longer exposes
    # pad_token_id as a top-level attribute (5.8.1 carried it as None); it now lives
    # on the nested text config. Read via getattr to avoid AttributeError, and set it
    # on both the outer config and the text sub-config the SeqCls head reads from.
    # For BERT/Qwen3 there is no text_config, so this collapses to the original path.
    text_config = getattr(model.config, "text_config", model.config)
    if getattr(text_config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        text_config.pad_token_id = tokenizer.pad_token_id
        print(f"  set pad_token_id = {tokenizer.pad_token_id} (config + text_config)")

    print("Building examples ...")
    train_examples = build_examples("train")
    print(f"  Train turns: {len(train_examples)}")
    test_examples = build_examples("test")
    print(f"  Test turns:  {len(test_examples)}")

    # Hold out 5% of train as in-domain eval
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_examples))
    n_eval = int(len(train_examples) * 0.05)
    eval_examples_in = [train_examples[i] for i in idx[:n_eval]]
    train_examples = [train_examples[i] for i in idx[n_eval:]]

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    train_ds = Dataset.from_list(train_examples).map(
        tokenize, batched=True, remove_columns=["text", "state"]
    )
    eval_ds = Dataset.from_list(eval_examples_in).map(
        tokenize, batched=True, remove_columns=["text", "state"]
    )

    # Manual training loop — bypassing HF Trainer entirely (rationale in commit history).
    # Model stays fp32 (params + optimizer state); --bf16-autocast wraps the forward
    # pass in torch.autocast for bf16 forward compute and ~2x speed/memory savings.
    import contextlib

    def amp_ctx():
        if args.bf16_autocast and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    if args.bf16_autocast:
        print(
            f"  precision: fp32 weights + bf16 autocast forward (model dtype={next(model.parameters()).dtype})"
        )
    else:
        print(f"  precision: pure fp32, no AMP (model dtype={next(model.parameters()).dtype})")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    eval_loader = make_loader(eval_ds, args.batch_size * 2, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # Resume state (model/optimizer/scheduler/epoch/step/wandb id) if requested.
    ckpt_path = out_dir / "checkpoint.pt"
    start_epoch = 0
    global_step = 0
    resume_wandb_id = None
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        resume_wandb_id = ckpt.get("wandb_id")
        print(f"  resumed at epoch {start_epoch + 1}/{args.epochs}, global_step {global_step}")

    wb = _init_wandb(args, model_type, resume_id=resume_wandb_id)
    wandb_id = wb.id if wb is not None else None

    def wb_log(metrics: dict, step: int | None = None) -> None:
        if wb is not None:
            wb.log(metrics, step=step)

    print(
        f"\nTraining ({total_steps} steps over {args.epochs} epochs, "
        f"{len(trainable_params)} trainable tensors) ..."
    )
    # One-time sanity: forward pass on first batch (no grad) before training starts.
    # If this produces NaN, the setup is broken; if it's clean, the bug is in the loop.
    model.train()
    diag_batch = next(iter(train_loader))
    diag_batch = {k: v.to(device) for k, v in diag_batch.items()}
    with torch.no_grad(), amp_ctx():
        diag_out = model(**diag_batch)
    print(
        f"  [diagnostic] pre-training forward: "
        f"logits[{diag_out.logits.min().item():.3g}, {diag_out.logits.max().item():.3g}], "
        f"loss={diag_out.loss.item():.4f}, "
        f"has_nan_logits={torch.isnan(diag_out.logits).any().item()}, "
        f"has_nan_loss={torch.isnan(diag_out.loss).item() if diag_out.loss.numel() == 1 else 'tensor'}"
    )
    if torch.isnan(diag_out.logits).any().item():
        raise RuntimeError(
            "Pre-training forward produced NaN logits. Aborting to avoid wasted training time."
        )

    torch.manual_seed(args.seed)
    import time as _time

    bench_t0 = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for batch in train_loader:
            # Start the benchmark clock after the first step warms up CUDA/JIT kernels.
            if args.max_steps and global_step == 1:
                _sync(device)
                bench_t0 = _time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_ctx():
                out = model(**batch)
                loss = out.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1
            wb_log(
                {
                    "train/loss": float(loss.item()),
                    "train/grad_norm": grad_norm,
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )
            if global_step < 5 or global_step % 200 == 0:
                print(
                    f"  step {global_step:5d}: loss={loss.item():.4f}  "
                    f"grad_norm={grad_norm:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )
            global_step += 1

            if args.max_steps and global_step >= args.max_steps:
                _sync(device)
                measured = global_step - 1  # exclude the warmup step before bench_t0
                dt = _time.perf_counter() - bench_t0
                spm = measured / dt * 60
                peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
                full_steps = len(train_loader) * args.epochs
                print(
                    f"\n[bench] {measured} steps in {dt:.1f}s = {spm:.1f} steps/min  "
                    f"| peak VRAM {peak:.1f} GB  | bs={args.batch_size} "
                    f"grad_ckpt={args.gradient_checkpointing} bf16={args.bf16_autocast}\n"
                    f"[bench] full run {full_steps} steps → ETA {full_steps / spm / 60:.1f} h"
                )
                return

        # End-of-epoch eval on held-out 5%
        model.eval()
        eval_correct = 0
        eval_total = 0
        eval_loss_sum = 0.0
        eval_steps = 0
        with torch.no_grad(), amp_ctx():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                eval_loss_sum += float(out.loss.item())
                eval_steps += 1
                preds = out.logits.argmax(-1)
                eval_correct += int((preds == batch["labels"]).sum().item())
                eval_total += int(batch["labels"].size(0))
        print(
            f"  Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={epoch_loss_sum / max(1, epoch_steps):.4f}  "
            f"eval_loss={eval_loss_sum / max(1, eval_steps):.4f}  "
            f"eval_acc={100 * eval_correct / max(1, eval_total):.2f}%"
        )
        wb_log(
            {
                "epoch": epoch + 1,
                "train/epoch_loss": epoch_loss_sum / max(1, epoch_steps),
                "eval/loss": eval_loss_sum / max(1, eval_steps),
                "eval/acc": eval_correct / max(1, eval_total),
            },
            step=global_step,
        )

        # Checkpoint at the epoch boundary so a crash resumes here instead of step 0.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "wandb_id": wandb_id,
            },
            ckpt_path,
        )
        print(f"  checkpoint saved → {ckpt_path} (resume point: epoch {epoch + 1})")

    # Test-split eval
    print("\nEvaluating on test split ...")
    test_ds = Dataset.from_list(test_examples).map(
        tokenize, batched=True, remove_columns=["text", "state"]
    )
    test_loader = make_loader(test_ds, args.batch_size * 2, shuffle=False)
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss_sum = 0.0
    test_steps = 0
    with torch.no_grad(), amp_ctx():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            test_loss_sum += float(out.loss.item())
            test_steps += 1
            preds = out.logits.argmax(-1)
            test_correct += int((preds == batch["labels"]).sum().item())
            test_total += int(batch["labels"].size(0))
    test_metrics = {
        "test_loss": test_loss_sum / max(1, test_steps),
        "test_acc": test_correct / max(1, test_total),
    }

    # Per-state and per-stage breakdown on test
    print("Per-state inference for breakdown ...")
    model.eval()
    device = next(model.parameters()).device
    correct_state = {s: 0 for s in ALL_STATES}
    total_state = {s: 0 for s in ALL_STATES}
    correct_stage = {s: 0 for s in "abcde"}
    total_stage = {s: 0 for s in "abcde"}
    overall_correct = 0
    bs = 64
    for i in range(0, len(test_examples), bs):
        batch = test_examples[i : i + bs]
        # Reload raw text - we need to grab from the original dicts (still have it)
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        # Tokenize
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad(), amp_ctx():
            logits = model(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()
        for p, lab, e in zip(preds, labels, batch):
            gt_state = ALL_STATES[lab]
            pred_state = ALL_STATES[p]
            total_state[gt_state] += 1
            total_stage[gt_state[0]] += 1
            if p == lab:
                overall_correct += 1
                correct_state[gt_state] += 1
            if pred_state[0] == gt_state[0]:
                correct_stage[gt_state[0]] += 1

    test_overall = overall_correct / len(test_examples)
    per_state_acc = {
        s: (correct_state[s] / total_state[s] if total_state[s] else 0.0) for s in ALL_STATES
    }
    per_stage_acc = {
        s: (correct_stage[s] / total_stage[s] if total_stage[s] else 0.0) for s in "abcde"
    }
    wb_log(
        {
            "test/state_accuracy": test_overall,
            "test/loss": test_metrics["test_loss"],
            **{f"test/stage_{s}": per_stage_acc[s] for s in "abcde"},
        }
    )
    out = {
        "n_test_turns": len(test_examples),
        "test_state_accuracy": test_overall,
        "test_stage_accuracy_via_state_pred": per_stage_acc,
        "test_per_state_accuracy": per_state_acc,
        "test_per_state_n": total_state,
        **{k: v for k, v in test_metrics.items() if isinstance(v, float)},
    }
    (out_dir / "test_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== 34-state classifier on test split ({len(test_examples)} turns) ===")
    print(f"  Overall state accuracy:        {test_overall * 100:.2f}%")
    print("  Implied stage accuracy:")
    for s in "abcde":
        print(f"    Stage {s}: {per_stage_acc[s] * 100:.2f}% (n={total_stage[s]})")
    print("  Top per-state hits (rate, n) for stages c+d:")
    for s in ALL_STATES:
        if s[0] in "cd" and total_state[s] > 0:
            print(f"    {s}: {per_state_acc[s] * 100:.2f}% (n={total_state[s]})")

    final_dir = out_dir / "final"
    if args.lora:
        # Merge LoRA adapters into the base weights so the artifact is a plain
        # HF SeqClassification checkpoint that kele.py's BERT-consultant loader
        # (AutoModelForSequenceClassification.from_pretrained) picks up unchanged.
        print("\nMerging LoRA adapters into base weights ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(str(final_dir))
    else:
        model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved to {final_dir}")
    if wb is not None:
        wb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
