#!/usr/bin/env bash
# One-shot: convert SocratTeachLLM (HF format) → GGUF (Q8_0) for llama.cpp serving.
#
# SocratTeachLLM is a LoRA fine-tune of THUDM/glm-4-9b-chat (ChatGLM4 architecture,
# ~9B params). The HF repo distributes the merged weights as 4× safetensors shards
# totaling ~18 GB in fp16. This script runs llama.cpp's convert_hf_to_gguf.py to
# produce a Q8_0 GGUF (~10 GB, nearly lossless) suitable for `serve_socratteachllm_llamacpp.sh`.
#
# Why Q8_0 (not Q5/Q4): we want minimal quantization-induced behavioral drift
# on this model specifically — its value is the fine-tune-specific Socratic-teaching
# behavior we're trying to characterize for the paper's overfit-hypothesis test.
# At ~10 GB on a 32 GB 5090, the extra memory cost vs Q5 is negligible.
#
# Usage:
#   ./scripts/convert_socratteachllm_to_gguf.sh
#
# Override paths with env vars:
#   HF_DIR=/path/to/hf  GGUF_DIR=/path/to/out  QUANT=Q5_K_M ./scripts/convert_socratteachllm_to_gguf.sh

set -euo pipefail

HF_DIR="${HF_DIR:-$HOME/hf_models/SocratTeachLLM}"
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

F16_GGUF="$GGUF_DIR/SocratTeachLLM-f16.gguf"
OUT_GGUF="$GGUF_DIR/SocratTeachLLM-${QUANT}.gguf"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "$HF_DIR" ]]; then
  echo "ERROR: HF model directory not found at $HF_DIR" >&2
  echo "Download with: hf download yuanpan/SocratTeachLLM --local-dir $HF_DIR" >&2
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
  echo "SocratTeachLLM-${QUANT}.gguf already exists at $OUT_GGUF"
  echo "Delete and re-run if you want to re-convert."
  exit 0
fi

# ── Workaround: trust_remote_code monkey-patch wrapper ───────────────────────
# ChatGLM4's config.json + tokenizer_config.json reference custom Python files
# (tokenization_chatglm.py, modeling_chatglm.py). When llama.cpp's
# convert_hf_to_gguf.py calls `AutoTokenizer.from_pretrained(dir)` WITHOUT
# trust_remote_code=True (base.py:1261 — a llama.cpp inconsistency: other lines
# in the same file DO pass it), transformers refuses to load. We don't actually
# need the custom modeling code for GGUF conversion (it reads safetensors
# directly + uses tokenizer.model SentencePiece file), but the transformers
# trust check fires before any of that. Workaround: monkey-patch transformers'
# from_pretrained to default trust_remote_code=True, then exec the convert
# script. The patch only lives during conversion.
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

# Default trust_remote_code=True on every from_pretrained we know about,
# AND patch loaded tokenizers to expose a `.vocab` attribute if missing
# (llama.cpp's chatglm handler unconditionally evaluates `len(tokenizer.vocab)`
# as its dict.get default, which AttributeErrors on ChatGLM4Tokenizer).
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

# Make convert_hf_to_gguf.py's relative imports work: add its dir to sys.path
# and chdir to it (the script uses `from conversion import ...` which needs
# the conversion package to be importable from cwd or sys.path).
script = sys.argv[1]
script_dir = os.path.dirname(os.path.abspath(script))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

# Now exec the script with its expected argv
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
  # Use the project's venv python directly so this script works under nohup
  # (uv-discovery breaks without an interactive shell PATH; see chain commit 6c3591a).
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
  # f16 intermediate preserves precision for the quantization step.
  # Run via the trust-remote-code monkey-patch wrapper (defined above).
  # PYTHONPATH: the wrapper exec()s the converter, so the converter's own
  # "add ./gguf-py to sys.path" logic resolves against the wrapper's location.
  PYTHONPATH="$LLAMA_CPP_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$WRAPPER" "$CONVERT" "$SRC_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16 \
    --model-name "SocratTeachLLM"
  cd - >/dev/null
fi

# ── Step 2: f16 GGUF → Q8_0 (or whatever QUANT is set to) ─────────────────────
echo
echo "=== Step 2: f16 GGUF → $QUANT ==="
echo "Input:  $F16_GGUF"
echo "Output: $OUT_GGUF"
echo "(this takes ~1 min on CPU)"
echo
"$QUANTIZE" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

# ── Optional cleanup ──────────────────────────────────────────────────────────
echo
echo "=== Done ==="
ls -lh "$OUT_GGUF"
echo
echo "Intermediate f16 GGUF (~18 GB) is preserved at:"
echo "  $F16_GGUF"
echo "Delete it to reclaim disk space:"
echo "  rm $F16_GGUF"
echo
echo "Next: bash scripts/serve_socratteachllm_llamacpp.sh"
