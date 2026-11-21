#!/usr/bin/env bash
# One-shot: convert THUDM/glm-4-9b-chat (HF format) → GGUF (Q8_0) for llama.cpp serving.
#
# GLM-4-9B-Chat is the BASE model that SocratTeachLLM was LoRA-fine-tuned from.
# This conversion mirrors scripts/convert_socratteachllm_to_gguf.sh exactly —
# same architecture (ChatGLM4), same tokenizer, same trust_remote_code +
# tokenizer.vocab monkey-patch needed during conversion.
#
# Purpose: enables the GLM-4-9B-Chat-base × * ablation cells that complete the
# STL contamination story — STL minus its own pre-trained base is the true
# fine-tune-contribution delta (see docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md).
#
# Why Q8_0 (not Q5/Q4): identical reasoning to the STL conversion — minimal
# quantization-induced drift. ~10 GB on a 32 GB 5090 leaves comfortable
# headroom for the consultant load.
#
# Usage:
#   ./scripts/convert_glm49b_chat_to_gguf.sh
#
# Override paths with env vars:
#   HF_DIR=/path/to/hf  GGUF_DIR=/path/to/out  QUANT=Q5_K_M ./scripts/convert_glm49b_chat_to_gguf.sh

set -euo pipefail

HF_DIR="${HF_DIR:-$HOME/hf_models/glm-4-9b-chat}"
GGUF_DIR="${GGUF_DIR:-$HOME/Documents/models/weights}"
QUANT="${QUANT:-Q8_0}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/Documents/models/llama.cpp}"
CONVERT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE="${QUANTIZE:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
if [[ ! -x "$QUANTIZE" ]] && command -v llama-quantize >/dev/null 2>&1; then
  QUANTIZE="$(command -v llama-quantize)"   # e.g. a distro llama.cpp package
fi
# Official tokenizer.json for this vocabulary (see the overlay step below).
TOKENIZER_REPO="${TOKENIZER_REPO:-zai-org/glm-4-9b-chat-hf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$(dirname "$HF_DIR")/glm-4-9b-chat-hf-tokenizer}"

F16_GGUF="$GGUF_DIR/glm-4-9b-chat-f16.gguf"
OUT_GGUF="$GGUF_DIR/glm-4-9b-chat-${QUANT}.gguf"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "$HF_DIR" ]]; then
  echo "ERROR: HF model directory not found at $HF_DIR" >&2
  echo "Download with: hf download THUDM/glm-4-9b-chat --local-dir $HF_DIR" >&2
  echo "(or set HF_DIR=/some/other/path)" >&2
  exit 1
fi

if [[ ! -f "$CONVERT" ]]; then
  echo "ERROR: convert_hf_to_gguf.py not found at $CONVERT" >&2
  echo "Build llama.cpp first or set LLAMA_CPP_DIR=/path/to/llama.cpp" >&2
  exit 1
fi

if [[ ! -x "$QUANTIZE" ]]; then
  echo "ERROR: llama-quantize binary not found at $QUANTIZE" >&2
  echo "Build llama.cpp first: cmake --build $LLAMA_CPP_DIR/build --config Release" >&2
  exit 1
fi

mkdir -p "$GGUF_DIR"

if [[ -f "$OUT_GGUF" ]]; then
  echo "glm-4-9b-chat-${QUANT}.gguf already exists at $OUT_GGUF"
  echo "Delete and re-run if you want to re-convert."
  exit 0
fi

# ── Workaround: trust_remote_code monkey-patch wrapper ───────────────────────
# Same workaround as the STL converter — ChatGLM4's config.json + tokenizer
# reference custom Python files. llama.cpp's convert_hf_to_gguf.py inconsistently
# passes trust_remote_code=True, and the chatglm4 vocab handler expects
# tokenizer.vocab (which ChatGLM4Tokenizer doesn't expose directly). Patch both.
WRAPPER=$(mktemp /tmp/convert_trust_remote_XXXXXX.py)
cat > "$WRAPPER" <<'PY'
"""Monkey-patch transformers + exec convert_hf_to_gguf.py.

For ChatGLM4 (and other custom-code models) where the upstream convert script
inconsistently passes trust_remote_code=True. Argv[1] is the convert script
path; remaining argv pass through to it.
"""
import os
import sys
import transformers

for cls_name in (
    "AutoConfig", "AutoTokenizer", "AutoModel",
    "AutoModelForCausalLM", "AutoModelForSeq2SeqLM",
    "AutoModelForSequenceClassification",
):
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        continue
    _orig = cls.from_pretrained
    is_tokenizer = cls_name == "AutoTokenizer"
    def make_wrapped(orig, fix_vocab):
        def wrapped(*args, **kwargs):
            kwargs.setdefault("trust_remote_code", True)
            obj = orig(*args, **kwargs)
            if fix_vocab:
                try:
                    _ = obj.vocab
                except AttributeError:
                    try:
                        obj.vocab = obj.get_vocab()
                    except Exception:
                        pass
            return obj
        return wrapped
    cls.from_pretrained = make_wrapped(_orig, is_tokenizer)

script = sys.argv[1]
script_dir = os.path.dirname(os.path.abspath(script))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

sys.argv = sys.argv[1:]
exec(compile(open(script).read(), script, "exec"))
PY

OVERLAY_DIR=""
restore_state() {
  rm -f "$WRAPPER"
  [[ -n "$OVERLAY_DIR" ]] && rm -rf "$OVERLAY_DIR"
  return 0
}
trap restore_state EXIT INT TERM

# ── Step 1: HF → f16 GGUF (intermediate; ~18 GB) ──────────────────────────────
if [[ ! -f "$F16_GGUF" ]]; then
  echo "=== Step 1: HF → f16 GGUF ==="
  echo "Input:  $HF_DIR"
  echo "Output: $F16_GGUF"
  echo "(this takes ~5 min on CPU; no GPU needed)"
  echo
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    echo "Set PYTHON=/path/to/python or run 'uv sync' in the repo first" >&2
    exit 1
  fi
  # ── Tokenizer overlay ───────────────────────────────────────────────────────
  # glm-4-9b-chat and its fine-tunes ship a tiktoken vocabulary (tokenizer.model
  # plus custom tokenizer code), not a tokenizer.json. With that layout the
  # converter writes raw token bytes and no BPE merges, and llama-server rejects
  # the file: "cannot find tokenizer merges in model file". The -hf repo of the
  # same model has a tokenizer.json with the identical vocabulary (all 151,329
  # ids and the 14 added tokens match) and the merge list. Convert from a
  # scratch directory that links the weights and uses those tokenizer files.
  SRC_DIR="$HF_DIR"
  if [[ ! -f "$HF_DIR/tokenizer.json" ]]; then
    if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
      echo "Fetching tokenizer files from $TOKENIZER_REPO → $TOKENIZER_DIR"
      "$(dirname "$PYTHON")/hf" download "$TOKENIZER_REPO" \
        tokenizer.json tokenizer_config.json --local-dir "$TOKENIZER_DIR"
    fi
    OVERLAY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/glm4_overlay_XXXXXX")"
    SRC_DIR="$OVERLAY_DIR"
    for f in "$HF_DIR"/*.safetensors "$HF_DIR"/model.safetensors.index.json \
             "$HF_DIR"/generation_config.json \
             "$HF_DIR"/configuration_chatglm.py "$HF_DIR"/modeling_chatglm.py; do
      [[ -e "$f" ]] && ln -s "$f" "$SRC_DIR/"
    done
    # The converter reads hparams["vocab_size"] even when padded_vocab_size is
    # present; the base glm-4-9b-chat config has only the padded key.
    "$PYTHON" - "$HF_DIR/config.json" "$SRC_DIR/config.json" <<'CFG'
import json
import sys

cfg = json.load(open(sys.argv[1]))
cfg.setdefault("vocab_size", cfg.get("padded_vocab_size"))
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
CFG
    cp "$TOKENIZER_DIR/tokenizer.json" "$TOKENIZER_DIR/tokenizer_config.json" "$SRC_DIR/"
    echo "Tokenizer overlay: $SRC_DIR"
  fi
  cd "$LLAMA_CPP_DIR"
  # PYTHONPATH: the wrapper exec()s the converter, so the converter's own
  # "add ./gguf-py to sys.path" logic resolves against the wrapper's location.
  PYTHONPATH="$LLAMA_CPP_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$WRAPPER" "$CONVERT" "$SRC_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16 \
    --model-name "glm-4-9b-chat"
  cd - >/dev/null
fi

# ── Step 2: f16 GGUF → Q8_0 ───────────────────────────────────────────────────
echo
echo "=== Step 2: f16 GGUF → $QUANT ==="
echo "Input:  $F16_GGUF"
echo "Output: $OUT_GGUF"
echo "(this takes ~1 min on CPU)"
echo
"$QUANTIZE" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

echo
echo "=== Done ==="
ls -lh "$OUT_GGUF"
echo
echo "Intermediate f16 GGUF (~18 GB) is preserved at:"
echo "  $F16_GGUF"
echo "Delete it to reclaim disk space:"
echo "  rm $F16_GGUF"
echo
echo "Next: bash scripts/serve_glm49b_chat_llamacpp.sh"
