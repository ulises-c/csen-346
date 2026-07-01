#!/usr/bin/env bash
# Collects OS / driver / ROCm / Tensile-library / stack info for the
# gfx1201 ISA1201 rocBLAS page-fault report (csen-346 #113 -> ROCm/rocm-libraries).
# Best-effort collector: intentionally NOT `set -e` so a missing tool does not abort
# the run. Tees everything to a timestamped file you can attach to the report.
set -uo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
report_dir="$(cd "$script_dir/.." && pwd)/docs/diagnostics"
mkdir -p "$report_dir"
out="$report_dir/gfx1201-report-env-$(date +%Y%m%d-%H%M%S).txt"
exec > >(tee "$out") 2>&1

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "ROCm / HIP build"
hipconfig --version 2>/dev/null || printf '(hipconfig unavailable)\n'
printf '\n'
cat "$ROCM"/.info/version* 2>/dev/null || printf '(%s/.info/version* unavailable)\n' "$ROCM"

hr "OS / kernel"
uname -a
grep -E 'PRETTY_NAME|^VERSION=' /etc/os-release 2>/dev/null || true

hr "GPU / driver / firmware / VBIOS"
rocm-smi --showdriverversion 2>/dev/null || printf '(rocm-smi unavailable)\n'
rocm-smi --showproductname --showvbios 2>/dev/null || true
printf '\n-- amdgpu module --\n'
modinfo amdgpu 2>/dev/null | grep -E '^version:|^srcversion:|^vermagic:' \
  || printf '(amdgpu version fields not exported — typical for the in-kernel/inbox driver)\n'
printf '\n-- rocminfo (gfx target) --\n'
rocminfo 2>/dev/null | grep -E 'Name:|Marketing Name:|gfx|Compute Unit:' | head -20 || printf '(rocminfo unavailable)\n'

hr "ROCm component package versions"
if command -v pacman >/dev/null 2>&1; then
  pacman -Q 2>/dev/null | grep -Ei 'rocm|rocblas|hipblas|tensile|^hip|comgr|miopen|amdgpu' || printf '(no matching pacman pkgs)\n'
elif command -v dpkg >/dev/null 2>&1; then
  dpkg -l 2>/dev/null | grep -Ei 'rocm|rocblas|hipblas|tensile|hip-|comgr|miopen|amdgpu' || printf '(no matching dpkg pkgs)\n'
else
  printf '(no pacman/dpkg found)\n'
fi

hr "Tensile library files present (issue #7192 lead)"
# Just the load-bearing files for the wrong-library-lookup question — not the full
# fallback-kernel .hsaco set (which is ~50 lines of noise).
ls -la "$ROCM"/lib/rocblas/library/TensileLibrary_lazy_gfx120[01].dat \
       "$ROCM"/lib/rocblas/library/Kernels.so-*-gfx120[01].hsaco 2>/dev/null \
  || printf '(no gfx1200/gfx1201 lazy Tensile library / Kernels.so under %s/lib/rocblas/library/)\n' "$ROCM"

hr "Calling stack (torch / hip / bitsandbytes)"
stack_probe='import torch, bitsandbytes as bnb; print("torch", torch.__version__, "| hip", torch.version.hip, "| bnb", bnb.__version__)'
if command -v uv >/dev/null 2>&1; then
  uv run --no-sync python -c "$stack_probe" 2>&1 || printf '(torch/bnb import failed)\n'
else
  python -c "$stack_probe" 2>&1 || printf '(uv not found and python import failed)\n'
fi

hr "rocblas-bench availability (for the standalone reproducer)"
if command -v rocblas-bench >/dev/null 2>&1; then
  printf 'rocblas-bench: FOUND -> %s\n' "$(command -v rocblas-bench)"
elif [[ -x "$ROCM/bin/rocblas-bench" ]]; then
  printf 'rocblas-bench: FOUND -> %s/bin/rocblas-bench\n' "$ROCM"
else
  printf 'rocblas-bench: NOT FOUND (install the rocblas test/bench package to capture the standalone repro)\n'
fi
printf 'Next (separate GPU run): add ROCBLAS_LAYER=2 to the training launch env to emit the\n'
printf 'exact "rocblas-bench --transposeA ... -m 608 -n 5376 -k 21504 --lda ... --ldb ..." line\n'
printf 'for the faulting GEMM -- that command line is the stack-independent reproducer.\n'

hr "DONE"
printf 'Saved to: %s\n' "$out"
