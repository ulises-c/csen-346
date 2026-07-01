#!/usr/bin/env python3
"""EOS gate: verify a trained checkpoint terminates cleanly on a realistic inference prompt.

Loads the base model, applies a saved LoRA adapter, then generates on a prompt
built with a realistic multi-sentence Chinese free-form evaluation — the irreducible
residual identified in PR #101 / issue #94. PASS = model stops on its own before
max_new_tokens. FAIL = hits the cap (likely repetition collapse).

Usage (from repo root):
    make train-gemma4-31b-eos-gate        # ~30 min: trains 100-step checkpoint
    tail -f outputs/eos-gate-gemma4-31b/train.log
    make eos-gate-gemma4-31b              # run the gate once training finishes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

# Allow `import train_sft` from the same scripts/ directory.
sys.path.insert(0, str(Path(__file__).parent))


def _load_model(adapter_dir: str | None):
    import train_sft  # noqa: PLC0415

    model, tokenizer = train_sft.build_model_and_tokenizer()
    if adapter_dir:
        from peft import PeftModel  # noqa: PLC0415

        print(f"\nLoading adapter  path={adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        print("  Adapter loaded")

    import torch  # noqa: PLC0415

    model.eval()
    torch.cuda.empty_cache()
    return model, tokenizer


def _stop_ids(tokenizer) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
    if turn_end not in (None, tokenizer.unk_token_id):
        ids.append(turn_end)
    return list(set(ids))


def _generate(
    model, tokenizer, messages: list[dict], max_new_tokens: int = 512
) -> tuple[str, bool]:
    import torch  # noqa: PLC0415

    stop = _stop_ids(tokenizer)
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    input_ids = (
        enc["input_ids"]
        if hasattr(enc, "__getitem__") and not isinstance(enc, torch.Tensor)
        else enc
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=stop or None,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    gen = out[0, input_ids.shape[1] :]
    decoded = tokenizer.decode(gen, skip_special_tokens=False)
    terminated = len(gen) < max_new_tokens or "<turn|>" in decoded
    return decoded, terminated


def _capture_prompt(evaluation: str, state: str) -> tuple[str, str]:
    """Capture the (system, user) the inference path sends for a given eval + state."""
    from src.project.socratic_teaching_system import (  # noqa: PLC0415
        SocraticTeachingSystem,
        get_action_for_state,
    )

    sts = SocraticTeachingSystem(
        consultant_api_key="test",
        consultant_base_url="http://localhost:0",
        consultant_model_name="m",
        teacher_api_key="test",
        teacher_base_url="http://localhost:0",
        teacher_model_name="m",
    )
    # One prior exchange in history, then the current student turn added unpaired
    # — mirrors process_student_input's add_to_history before socrates_teacher call.
    sts.add_to_history("student", "科学实验中光发生折射是什么原因？")
    sts.add_to_history("teacher", "你知道光在不同介质中传播速度不同吗？这和方向变化有什么关系？")
    current_student = "我觉得光变慢了所以才弯曲？"
    sts.add_to_history("student", current_student)
    sts.current_state = state

    captured: dict = {}

    def _fake_call(client, model, system_prompt, user_input, predicted_state):  # noqa: ANN001
        captured["system"] = system_prompt
        captured["user"] = user_input
        return "captured"

    action = get_action_for_state(state)
    with patch("src.project.tournament_utilizations.call_teacher_wrapped", side_effect=_fake_call):
        sts.socrates_teacher(current_student, evaluation, action)

    return captured["system"], captured["user"]


def main() -> None:
    parser = argparse.ArgumentParser(description="EOS gate for a trained SFT checkpoint")
    parser.add_argument(
        "--config", metavar="PATH", help="env config file (sets TRAIN_BASE_MODEL etc.)"
    )
    parser.add_argument(
        "--adapter", metavar="PATH", help="path to saved LoRA adapter dir (checkpoint/final)"
    )
    args = parser.parse_args()

    if args.config:
        from src.project.config import load_env_file  # noqa: PLC0415

        load_env_file(Path(args.config))

    if not args.adapter:
        print(
            "WARNING: --adapter not provided — testing the base model only.\n"
            "         For a meaningful gate, train a short checkpoint first:\n"
            "           make train-gemma4-31b-eos-gate",
            file=sys.stderr,
        )

    model, tokenizer = _load_model(args.adapter)

    # ── Step 1: warm-up ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("STEP 1  warm-up (trivial prompt — verify stop tokens fire)")
    print("=" * 64)
    warmup_out, warmup_ok = _generate(
        model,
        tokenizer,
        [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
        max_new_tokens=200,
    )
    print(f"\n{warmup_out!r}")
    print(
        f"\nWarm-up: {'PASS' if warmup_ok else 'FAIL — stop tokens may not be configured correctly'}"
    )

    # ── Step 2: real EOS gate ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("STEP 2  EOS gate (realistic free-form Chinese evaluation)")
    print("=" * 64)

    # Multi-sentence realistic consultant prose — NOT the 学生处于 {state} 状态
    # template. This is the irreducible residual: training uses stored annotations,
    # inference uses live consultant output like this.
    freeform_eval = (
        "学生已经掌握了光在不同介质中传播速度变化的基本概念，能够直觉性地联系速度变化和方向改变。"
        "但对于具体的折射定律（如斯涅尔定律）以及折射角的定量计算仍存在明显困惑。"
        "建议进一步引导学生通过具体实例（如铅笔插入水中的弯折现象）来巩固其理解，"
        "并逐步引入定量分析框架。"
    )
    state = "b2"

    sys_msg, user_msg = _capture_prompt(freeform_eval, state)
    print(f"\nEvaluation ({len(freeform_eval)} chars):\n  {freeform_eval}")
    print(f"\nUser message preview:\n  {user_msg[:300]}...")

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    gate_out, gate_ok = _generate(model, tokenizer, messages, max_new_tokens=512)

    print(f"\nGeneration:\n{gate_out}")
    print("\n" + "=" * 64)
    if gate_ok:
        print("EOS GATE  PASSED — model terminated before max_new_tokens")
        print("Safe to launch: make train-gemma4-31b-stage2-unsloth")
    else:
        print("EOS GATE  FAILED — model hit max_new_tokens without terminating")
        print("Do NOT launch the full run. Check for repetition or format drift.")
    print("=" * 64)
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
