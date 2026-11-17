#!/usr/bin/env bash
# GPU tech-stack smoke test for ROCm / RDNA machines.
#
# Tests the full ML stack needed for inference and fine-tuning, in order:
#   1. ROCm driver + GPU visibility
#   2. torch ROCm — GPU tensor op
#   3. bitsandbytes — import + GPU detection
#   4. bitsandbytes 8-bit Linear (LLM.int8 / llm_int8 quantization)
#   5. bitsandbytes 4-bit NF4 Linear (QLoRA quantization)
#   6. transformers + BitsAndBytesConfig (QLoRA config object)
#   7. PEFT + LoraConfig (LoRA adapter config)
#   8. TRL SFTConfig (fine-tuning trainer) — skipped if not installed
#   9. PyTorch SDPA (Flash-Attention equivalent for ROCm) + flash-attn if installed
#
# No model weights are downloaded. Each step is independent.
#
# This is the single unified GPU checker for the project — clean-state, driver,
# torch, bitsandbytes, transformers, PEFT, TRL, attention, llama.cpp. Do not add
# parallel one-off GPU check scripts; extend this one.
#
# TODO: extend step 1 to also support NVIDIA/CUDA (nvidia-smi) for RTX 5090.
#   When nvidia-smi is found, skip ROCm-specific probes and use CUDA backend
#   checks instead. This allows the same script to run on both AMD and NVIDIA
#   machines without manual modification.
#
# Usage:
#   bash scripts/test_gpu_stack.sh                  # full 13-step stack test
#   bash scripts/test_gpu_stack.sh --preflight      # fast training gate: clean
#                                                     state + torch fwd/bwd (~5s)
#   bash scripts/test_gpu_stack.sh --wait-clean 120 # poll up to 120s for a clean
#                                                     GPU (used by the resume monitor)
#
# Clean-state tunables (env): GPU_PROC_PATTERN (default train_sft\.py),
#   GPU_VRAM_THRESHOLD_MB (default 1024).
#
# Exit code: 0 if all non-optional steps pass, 1 if any fail.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
pass()  { echo -e "  ${GREEN}PASS${NC}  $*"; }
fail()  { echo -e "  ${RED}FAIL${NC}  $*"; }
warn()  { echo -e "  ${YELLOW}WARN${NC}  $*"; }
step()  { echo -e "\n${BOLD}[$1/13] $2${NC}"; }

# ── GPU clean-state + fwd/bwd helpers (shared by --preflight / --wait-clean and
#    the resume monitor) ────────────────────────────────────────────────────────
# A gfx1201 page fault leaves the amdkfd dirty: orphaned procs keep a HIP context
# and stale VRAM mapped, so the next launch faults early on stale page-table
# entries (docs/GFX1201_RDNA4_TRAINING.md §10). These confirm the GPU is actually
# free and functional before a (re)launch — the gate the monitor used to skip.
GPU_PROC_PATTERN="${GPU_PROC_PATTERN:-train_sft\\.py}"
GPU_VRAM_THRESHOLD_MB="${GPU_VRAM_THRESHOLD_MB:-1024}"

gpu_vram_used_mb() {
    rocm-smi --showmeminfo vram --json 2>/dev/null | .venv/bin/python -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(-1); sys.exit(0)
best = 0
for card in d.values():
    if isinstance(card, dict):
        for k, v in card.items():
            if "used" in k.lower() and "memory" in k.lower():
                try:
                    best = max(best, int(str(v).strip()))
                except ValueError:
                    pass
print(best // (1024 * 1024))
'
}

gpu_clean_state() {
    local used pids is_dirty=0
    pids="$(rocm-smi --showpids 2>/dev/null | grep -iE "$GPU_PROC_PATTERN" || true)"
    used="$(gpu_vram_used_mb)"
    if [[ -n "$pids" ]]; then
        fail "GPU busy — process matching /$GPU_PROC_PATTERN/ still resident:"
        printf '%s\n' "$pids" | sed 's/^/         /'
        is_dirty=1
    fi
    if [[ "$used" -lt 0 ]]; then
        warn "VRAM usage unreadable (rocm-smi JSON parse failed) — inconclusive"
    elif [[ "$used" -gt "$GPU_VRAM_THRESHOLD_MB" ]]; then
        fail "VRAM used ${used}MB > ${GPU_VRAM_THRESHOLD_MB}MB threshold (stale allocation?)"
        is_dirty=1
    else
        pass "GPU idle — VRAM used ${used}MB (≤ ${GPU_VRAM_THRESHOLD_MB}MB), no training proc"
    fi
    return "$is_dirty"
}

gpu_fwd_bwd() {
    local out
    out="$(.venv/bin/python - 2>&1 <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("FAIL torch sees no GPU"); sys.exit(1)
try:
    torch.manual_seed(0)
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    ((a @ b).float().pow(2).mean()).backward()
    torch.cuda.synchronize()
    if a.grad is None or not torch.isfinite(a.grad).all():
        print("FAIL gradient missing/non-finite — GPU compute is wedged"); sys.exit(1)
    free, total = torch.cuda.mem_get_info()
    print(f"fwd+bwd OK  free={free/1024**3:.1f}/{total/1024**3:.1f}GB")
except Exception as e:  # noqa: BLE001 — any HIP fault here means the GPU is unusable
    print(f"FAIL fwd+bwd raised {type(e).__name__}: {e}"); sys.exit(1)
PY
)"
    if echo "$out" | grep -q "^FAIL"; then
        fail "$(echo "$out" | grep '^FAIL' | sed 's/^FAIL //')"
        return 1
    fi
    pass "$out"
    return 0
}

# ── Mode dispatch ──────────────────────────────────────────────────────────────
MODE=full
WAIT_SECONDS=0
case "${1:-}" in
    --preflight)  MODE=preflight ;;
    --wait-clean) MODE=wait-clean; WAIT_SECONDS="${2:-120}" ;;
    -h|--help)
        grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'
        exit 0 ;;
    "") ;;
    *) printf 'error: unknown arg %s (try --help)\n' "$1" >&2; exit 2 ;;
esac

if ! command -v rocm-smi >/dev/null 2>&1 && [[ "$MODE" != full ]]; then
    printf 'error: rocm-smi not found — not a ROCm host\n' >&2
    exit 2
fi

if [[ "$MODE" == "wait-clean" ]]; then
    echo -e "${BOLD}Waiting up to ${WAIT_SECONDS}s for a clean GPU${NC}"
    waited=0
    while true; do
        if gpu_clean_state; then
            pass "GPU clean — safe to (re)launch"
            exit 0
        fi
        if (( waited >= WAIT_SECONDS )); then
            fail "GPU still dirty after ${waited}s — refusing to launch"
            exit 1
        fi
        sleep 3
        (( waited += 3 )) || true
    done
fi

if [[ "$MODE" == "preflight" ]]; then
    echo -e "${BOLD}GPU pre-flight — clean state + torch fwd/bwd${NC}"
    PF_FAIL=0
    gpu_clean_state || PF_FAIL=$((PF_FAIL + 1))
    gpu_fwd_bwd     || PF_FAIL=$((PF_FAIL + 1))
    echo ""
    if [[ "$PF_FAIL" -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}Pre-flight passed — GPU clean and computing.${NC}"
    else
        echo -e "${RED}${BOLD}Pre-flight failed ($PF_FAIL) — do NOT launch.${NC}"
        echo "  Clear it: pkill -9 -f '$GPU_PROC_PATTERN'; then re-run. A wedged"
        echo "  context that survives the kill needs a GPU/driver reset."
    fi
    exit "$PF_FAIL"
fi

# ── Full stack test (MODE=full) ────────────────────────────────────────────────
FAILURES=0

step 0 "GPU clean state (orphan procs / stale VRAM)"
gpu_clean_state || warn "GPU not idle — fine if another job is intentionally running"

# ── 1. ROCm driver ────────────────────────────────────────────────────────────
step 1 "ROCm driver + GPU visibility"
if ! command -v rocm-smi &>/dev/null; then
    fail "rocm-smi not found — ROCm is not installed."
    FAILURES=$((FAILURES + 1))
else
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null \
        || rocminfo 2>/dev/null | awk '/ROCm Runtime/{match($0,/[0-9]+\.[0-9]+/); print substr($0,RSTART,RLENGTH); exit}' \
        || echo "unknown")
    GPU_ARCH=$(rocminfo 2>/dev/null | awk '/Name:.*gfx/{print $NF; exit}') || true
    [[ -z "${GPU_ARCH:-}" ]] && GPU_ARCH="unknown"
    pass "ROCm $ROCM_VER  arch: $GPU_ARCH"
    rocm-smi --showproductname 2>/dev/null | grep -v "^$" | sed 's/^/         /' || true
fi

# ── 2. torch ROCm ─────────────────────────────────────────────────────────────
step 2 "torch — GPU tensor op"
TORCH_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import torch
except ImportError:
    print("FAIL torch not installed")
    sys.exit(1)

print(f"torch {torch.__version__}  HIP {getattr(torch.version, 'hip', 'n/a')}")
if not torch.cuda.is_available():
    print("FAIL GPU not visible to torch (torch.cuda.is_available() returned False)")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}  {p.total_memory // 1024**3} GB")

t = torch.tensor([1.0, 2.0, 3.0]).cuda()
assert t.sum().item() == 6.0
print("tensor op on GPU: OK")
PY
)
if echo "$TORCH_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$TORCH_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$TORCH_OUT"
fi

# ── 3. bitsandbytes — import + GPU detection ──────────────────────────────────
step 3 "bitsandbytes — import + GPU detection"
BNB_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import bitsandbytes as bnb
except ImportError:
    print("FAIL bitsandbytes not installed — run: uv sync")
    sys.exit(1)

print(f"bitsandbytes {bnb.__version__}")

try:
    import torch
    if not torch.cuda.is_available():
        print("FAIL GPU not visible — bitsandbytes cannot initialise CUDA/ROCm backend")
        sys.exit(1)

    # Probe the bnb CUDA state without running a forward pass yet.
    # bnb.cuda_specs (older) or the state dict (newer) tells us if the backend loaded.
    state = None
    if hasattr(bnb, "cuda_specs"):
        state = bnb.cuda_specs
        print(f"cuda_specs: {state}")
    elif hasattr(bnb, "get_state"):
        state = bnb.get_state()
        print(f"bnb state: {state}")
    else:
        # Trigger backend load by instantiating the functional module.
        _ = bnb.functional
        print("bnb backend loaded (functional module OK)")

    # Check for common ROCm/HIP backend flags.
    hip = getattr(torch.version, "hip", None)
    if hip:
        print(f"ROCm/HIP backend: {hip}")
    else:
        print("CUDA backend (non-ROCm torch)")

except Exception as e:
    print(f"FAIL bitsandbytes GPU probe raised: {e}")
    sys.exit(1)
PY
)
if echo "$BNB_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$BNB_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$BNB_OUT"
fi

# ── 4. bitsandbytes 8-bit Linear (LLM.int8) ───────────────────────────────────
step 4 "bitsandbytes — 8-bit Linear forward pass (LLM.int8)"
BNB8_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import torch
    import bitsandbytes as bnb
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL GPU not available")
    sys.exit(1)

try:
    # Linear8bitLt is the LLM.int8() quantized linear layer.
    linear = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False, bias=False)
    linear = linear.cuda()

    x = torch.randn(2, 64, dtype=torch.float16).cuda()
    with torch.no_grad():
        y = linear(x)

    print(f"Linear8bitLt forward: input {tuple(x.shape)} → output {tuple(y.shape)}  OK")
    print(f"output dtype: {y.dtype}  device: {y.device}")
except Exception as e:
    print(f"FAIL Linear8bitLt forward raised: {e}")
    sys.exit(1)
PY
)
if echo "$BNB8_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$BNB8_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$BNB8_OUT"
fi

# ── 5. bitsandbytes 4-bit NF4 Linear (QLoRA) ──────────────────────────────────
step 5 "bitsandbytes — 4-bit NF4 Linear forward pass (QLoRA)"
BNB4_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import torch
    import bitsandbytes as bnb
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL GPU not available")
    sys.exit(1)

try:
    # Linear4bit with NF4 + bf16 compute is the exact QLoRA configuration.
    linear = bnb.nn.Linear4bit(
        64, 64,
        bias=False,
        quant_type="nf4",
        compute_dtype=torch.bfloat16,
    )
    linear = linear.cuda()

    x = torch.randn(2, 64, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        y = linear(x)

    print(f"Linear4bit (NF4, bf16) forward: input {tuple(x.shape)} → output {tuple(y.shape)}  OK")
    print(f"output dtype: {y.dtype}  device: {y.device}")
    print("QLoRA quantization: supported on this GPU")
except Exception as e:
    print(f"FAIL Linear4bit NF4 forward raised: {e}")
    sys.exit(1)
PY
)
if echo "$BNB4_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$BNB4_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$BNB4_OUT"
fi

# ── 6. transformers + BitsAndBytesConfig ──────────────────────────────────────
step 6 "transformers — BitsAndBytesConfig (QLoRA config object)"
TF_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import transformers
except ImportError:
    print("FAIL transformers not installed — run: uv sync")
    sys.exit(1)

print(f"transformers {transformers.__version__}")

try:
    import torch
    from transformers import BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # QLoRA double-quant
    )
    print(f"BitsAndBytesConfig: OK  (4-bit NF4, double-quant, bf16 compute)")

    # Also confirm 8-bit config path.
    bnb_cfg_8b = BitsAndBytesConfig(load_in_8bit=True)
    print(f"BitsAndBytesConfig: OK  (8-bit LLM.int8)")
except Exception as e:
    print(f"FAIL BitsAndBytesConfig raised: {e}")
    sys.exit(1)
PY
)
if echo "$TF_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$TF_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$TF_OUT"
fi

# ── 7. PEFT + LoraConfig ──────────────────────────────────────────────────────
step 7 "PEFT — LoraConfig (LoRA adapter config)"
PEFT_INSTALLED=$(.venv/bin/python -c "import peft; print(peft.__version__)" 2>/dev/null || echo "")
if [[ -z "$PEFT_INSTALLED" ]]; then
    warn "peft not installed — skipping.  Install with: uv add peft"
else
    PEFT_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import peft
    from peft import LoraConfig, TaskType
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

print(f"peft {peft.__version__}")

try:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    print(f"LoraConfig: OK  (r={config.r}, alpha={config.lora_alpha}, "
          f"target={config.target_modules})")
except Exception as e:
    print(f"FAIL LoraConfig raised: {e}")
    sys.exit(1)
PY
    )
    if echo "$PEFT_OUT" | grep -q "^FAIL"; then
        fail "$(echo "$PEFT_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
        FAILURES=$((FAILURES + 1))
    else
        while IFS= read -r line; do pass "$line"; done <<< "$PEFT_OUT"
    fi
fi

# ── 8. TRL SFTConfig (optional) ───────────────────────────────────────────────
step 8 "TRL — SFTConfig (fine-tuning trainer)  [optional]"
TRL_INSTALLED=$(.venv/bin/python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "")
if [[ -z "$TRL_INSTALLED" ]]; then
    warn "trl not installed — skipping.  Install with: uv add trl"
    warn "TRL is required for SFT fine-tuning (approach #4 in IMPROVEMENT_PLAN.md)."
else
    TRL_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import trl
    from trl import SFTConfig
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

print(f"trl {trl.__version__}")

try:
    cfg = SFTConfig(
        output_dir="/tmp/trl_probe",
        max_length=512,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
    )
    print(f"SFTConfig: OK  (max_length={cfg.max_length}, "
          f"grad_ckpt={cfg.gradient_checkpointing})")
except Exception as e:
    print(f"FAIL SFTConfig raised: {e}")
    sys.exit(1)
PY
    )
    if echo "$TRL_OUT" | grep -q "^FAIL"; then
        fail "$(echo "$TRL_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
        FAILURES=$((FAILURES + 1))
    else
        while IFS= read -r line; do pass "$line"; done <<< "$TRL_OUT"
    fi
fi

# ── 9. Efficient attention (PyTorch SDPA + optional flash-attn) ───────────────
step 9 "Efficient attention — PyTorch SDPA + flash-attn if installed"
SDPA_OUT=$(FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE .venv/bin/python - 2>&1 <<'PY'
import sys, warnings
warnings.filterwarnings("ignore")
try:
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL GPU not available for SDPA")
    sys.exit(1)

print(f"torch {torch.__version__}  SDPA backends available:")
flash_ok  = torch.backends.cuda.flash_sdp_enabled()
mem_ok    = torch.backends.cuda.mem_efficient_sdp_enabled()
math_ok   = torch.backends.cuda.math_sdp_enabled()
print(f"  flash={flash_ok}  mem_efficient={mem_ok}  math={math_ok}")

# Run a forward pass with all backends enabled.
B, H, S, D = 1, 4, 64, 32
q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
with sdpa_kernel(backends):
    out = scaled_dot_product_attention(q, k, v)

print(f"scaled_dot_product_attention: {tuple(q.shape)} → {tuple(out.shape)}  OK")
print(f"output dtype: {out.dtype}  device: {out.device}")
print("PyTorch SDPA (Flash-Attention equivalent for ROCm): supported")

# Probe flash_attn package if installed.
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print(f"flash_attn package {flash_attn.__version__} also installed")
    B2, S2, H2, D2 = 1, 64, 4, 32
    q2 = torch.randn(B2, S2, H2, D2, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(B2, S2, H2, D2, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(B2, S2, H2, D2, dtype=torch.bfloat16, device="cuda")
    out2 = flash_attn_func(q2, k2, v2)
    print(f"flash_attn_func: {tuple(q2.shape)} → {tuple(out2.shape)}  OK")
except ImportError:
    print("flash_attn package not installed (optional; PyTorch SDPA is sufficient on ROCm)")
except Exception as e:
    print(f"flash_attn_func raised: {e}  (non-fatal; SDPA passed above)")
PY
)
if echo "$SDPA_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$SDPA_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$SDPA_OUT"
fi

# ── llama-server discovery (shared by steps 10–13) ───────────────────────────
_LLAMA_SERVER="${LLAMA_SERVER:-}"
if [[ -z "$_LLAMA_SERVER" ]]; then
    if [[ -x "$HOME/Github/llama.cpp/build/bin/llama-server" ]]; then
        _LLAMA_SERVER="$HOME/Github/llama.cpp/build/bin/llama-server"
    elif [[ -x "$HOME/Documents/models/llama.cpp/build/bin/llama-server" ]]; then
        _LLAMA_SERVER="$HOME/Documents/models/llama.cpp/build/bin/llama-server"
    fi
fi
_LLAMA_MISSING=1
_CMAKE_CACHE=""
if [[ -n "$_LLAMA_SERVER" && -x "$_LLAMA_SERVER" ]]; then
    _LLAMA_MISSING=0
    _LLAMA_BUILD="$(dirname "$(dirname "$_LLAMA_SERVER")")"
    [[ -f "$_LLAMA_BUILD/CMakeCache.txt" ]] && _CMAKE_CACHE="$_LLAMA_BUILD/CMakeCache.txt"
fi

# ── 10. llama-server binary — GPU build verification ─────────────────────────
step 10 "llama-server — binary + ROCm build flags"
if [[ "$_LLAMA_MISSING" -eq 1 ]]; then
    warn "llama-server not found — skipping steps 10–13"
    warn "  Expected: ~/Github/llama.cpp/build/bin/llama-server"
    warn "  Build:    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 && cmake --build build -j\$(nproc)"
else
    LLAMA_VER=$("$_LLAMA_SERVER" --version 2>&1 | grep -E "^version:|ROCm|gfx|VRAM")
    while IFS= read -r line; do pass "$line"; done <<< "$LLAMA_VER"

    if [[ -n "$_CMAKE_CACHE" ]]; then
        _AMDGPU=$(grep "^AMDGPU_TARGETS"    "$_CMAKE_CACHE" | cut -d= -f2 || echo "?")
        _HIP=$(   grep "^GGML_HIP:"         "$_CMAKE_CACHE" | cut -d= -f2 || echo "?")
        _GRAPHS=$(grep "^GGML_HIP_GRAPHS:"  "$_CMAKE_CACHE" | cut -d= -f2 || echo "?")
        _MFMA=$(  grep "^GGML_HIP_MMQ_MFMA:" "$_CMAKE_CACHE" | cut -d= -f2 || echo "?")
        pass "AMDGPU_TARGETS=$_AMDGPU  GGML_HIP=$_HIP  HIP_GRAPHS=$_GRAPHS  MMQ_MFMA=$_MFMA"
        if [[ "$_HIP" != "ON" ]]; then
            fail "GGML_HIP is not ON — rebuild with -DGGML_HIP=ON"
            FAILURES=$((FAILURES + 1))
        fi
        if [[ "$_AMDGPU" != *"gfx1201"* ]]; then
            warn "AMDGPU_TARGETS does not include gfx1201 — may be using generic fallback"
        fi
    else
        warn "CMakeCache.txt not found at $_LLAMA_BUILD — cannot verify build flags"
    fi
fi

# ── 11. Flash attention — compiled + runtime status ───────────────────────────
step 11 "llama-server — flash attention (compiled-in + serve-script check)"
if [[ "$_LLAMA_MISSING" -eq 1 ]]; then
    warn "skipped (llama-server not found)"
else
    if [[ -n "$_CMAKE_CACHE" ]]; then
        _FA=$(grep "^GGML_CUDA_FA:" "$_CMAKE_CACHE" | cut -d= -f2 || echo "")
        if [[ "$_FA" == "ON" ]]; then
            pass "GGML_CUDA_FA=ON — flash attention compiled in"
        elif [[ "$_FA" == "OFF" ]]; then
            fail "GGML_CUDA_FA=OFF — flash attention not compiled; rebuild with -DGGML_CUDA_FA=ON"
            FAILURES=$((FAILURES + 1))
        else
            warn "GGML_CUDA_FA not found in CMakeCache — assumed compiled in"
        fi
    fi

    # Check whether serve scripts set -fa explicitly or rely on auto.
    _FA_SCRIPTS=$(grep -rl "\-fa\b\|--flash-attn" scripts/serve_*.sh 2>/dev/null | tr '\n' ' ' || true)
    if [[ -n "$_FA_SCRIPTS" ]]; then
        pass "Explicit -fa flag found in: $_FA_SCRIPTS"
    else
        warn "No serve script sets -fa explicitly — using default (auto)"
        warn "  'auto' activates FA for gfx1201. To confirm at runtime, start the server"
        warn "  with a model and check logs for:  llm_load_print_meta: flash attn = 1"
    fi
fi

# ── 12. Vulkan backend ────────────────────────────────────────────────────────
step 12 "llama-server — Vulkan backend (RDNA4 ~20% TG speedup)"
if [[ "$_LLAMA_MISSING" -eq 1 ]]; then
    warn "skipped (llama-server not found)"
else
    if [[ -n "$_CMAKE_CACHE" ]]; then
        _VULKAN=$(grep "^GGML_VULKAN:" "$_CMAKE_CACHE" | cut -d= -f2 || echo "")
        if [[ "$_VULKAN" == "ON" ]]; then
            pass "GGML_VULKAN=ON — Vulkan backend compiled in"
        else
            warn "GGML_VULKAN is OFF (or absent) — Vulkan not compiled in"
            warn "  Benchmarks show Vulkan is ~20% faster than ROCm HIP for TG on gfx1201"
            warn "  See ~/Github/llama.cpp/.claude/handoff.md for the full rebuild command"
        fi
    fi

    # Check Vulkan ICD availability at the OS level.
    if command -v vulkaninfo &>/dev/null; then
        _VKGPU=$(vulkaninfo 2>/dev/null | grep -E "GPU id|deviceName" | sort -u | head -4 || true)
        if [[ -n "$_VKGPU" ]]; then
            while IFS= read -r line; do pass "vulkaninfo: $line"; done <<< "$_VKGPU"
        else
            warn "vulkaninfo found but no GPU listed — check Vulkan ICD configuration"
        fi
    else
        warn "vulkaninfo not installed — cannot verify Vulkan ICD"
        warn "  Install: sudo pacman -S vulkan-tools  (or equivalent)"
    fi
fi

# ── 13. Serve script efficiency settings ─────────────────────────────────────
step 13 "llama-server — serve script efficiency (ubatch, KV quant, parallel slots)"
if [[ "$_LLAMA_MISSING" -eq 1 ]]; then
    warn "skipped (llama-server not found)"
else
    for _SCRIPT in scripts/serve_gemma4_31b.sh scripts/serve_qwen27b.sh; do
        [[ -f "$_SCRIPT" ]] || continue
        echo -e "  ${BOLD}$_SCRIPT${NC}"

        # ubatch-size
        _UBATCH=$(grep -E -- '-ub\b|--ubatch-size' "$_SCRIPT" | grep -v '^\s*#' | head -1 || true)
        if [[ -z "$_UBATCH" ]]; then
            warn "  --ubatch-size not set (default 512) — recommend 2048 for 32 GB VRAM"
        else
            pass "  ubatch: $_UBATCH"
        fi

        # KV cache quant
        _KV=$(grep 'KV_QUANT\s*=' "$_SCRIPT" | grep -v '^\s*#' | head -1 || true)
        [[ -n "$_KV" ]] && pass "  KV quant: $_KV"

        # Parallel slots
        _NP=$(grep 'PARALLEL\s*=' "$_SCRIPT" | grep -v '^\s*#' | head -1 || true)
        [[ -n "$_NP" ]] && pass "  Parallel slots: $_NP"

        # Flash attn in this specific script
        _FA_HERE=$(grep -E -- '-fa\b|--flash-attn' "$_SCRIPT" | grep -v '^\s*#' | head -1 || true)
        if [[ -z "$_FA_HERE" ]]; then
            warn "  -fa not set (inherits auto from base script — OK, see step 11)"
        else
            pass "  flash-attn: $_FA_HERE"
        fi
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Result ━━━${NC}"
if [[ "$FAILURES" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All steps passed.${NC}"
    echo ""
    echo "  GPU visibility:               steps 1–2 confirm GPU is usable."
    echo "  QLoRA fine-tuning:            steps 3–6 all passing = QLoRA is supported."
    echo "  LoRA fine-tuning:             step 7 passing = PEFT/LoRA is ready."
    echo "  SFT trainer:                  step 8 passing = TRL SFTTrainer is ready."
    echo "  Efficient attention (torch):  step 9 = PyTorch SDPA + flash-attn (if installed)"
    echo "    flash_attn install note: PyPI flash-attn 2.8.3+ works on gfx1201 via Triton JIT."
    echo "    Install (once): FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE uv pip install flash-attn --no-build-isolation"
    echo "    Set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE at training time too (Triton JIT selects AMD path)."
    echo ""
    echo "  llama-server build:           step 10 = ROCm/HIP flags verified."
    echo "  Flash attention (llama.cpp):  step 11 = FA compiled in; check runtime logs for 'flash attn = 1'."
    echo "  Vulkan backend:               step 12 = Vulkan compiled in (~20% faster TG on gfx1201)."
    echo "                                         If WARN: see ~/Github/llama.cpp/.claude/handoff.md"
    echo "  Serve script settings:        step 13 = ubatch/KV quant/slots reviewed."
    echo ""
    echo "  To check vLLM separately:  bash scripts/test_vllm_rocm.sh"
else
    echo -e "${RED}${BOLD}$FAILURES step(s) failed.${NC}"
    echo ""
    echo "  Steps 1–7 are required for QLoRA fine-tuning."
    echo "  Steps 8–9 are optional (TRL, flash-attn)."
    echo ""
    if [[ "$FAILURES" -gt 0 ]]; then
        echo "  If bitsandbytes (steps 3–5) failed on ROCm gfx1201:"
        echo "    1. Check bnb ROCm support: https://github.com/bitsandbytes-foundation/bitsandbytes"
        echo "    2. Try installing the ROCm-specific build:"
        echo "         uv pip install bitsandbytes --index-url https://pypi.org/simple/"
        echo "    3. Or pin a known-good ROCm build:"
        echo "         uv add 'bitsandbytes>=0.45.0'"
        echo "    4. If QLoRA is unavailable, fall back to bf16 LoRA (no bnb needed)."
    fi
fi
echo ""
exit "$FAILURES"
