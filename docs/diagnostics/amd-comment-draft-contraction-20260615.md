# Draft comment for ROCm/rocm-libraries#7992 (2026-06-15)

> Status: POSTED 2026-06-15 → https://github.com/ROCm/rocm-libraries/issues/7992#issuecomment-4706745017

---

## The fault is invocation-dependent, not data-dependent — and you already have the kernel

@amd-nicknick — first, thanks: I diffed the kernel you force-dispatched against our production faulting kernel and they are **byte-identical**, down to `MT64x64x64 … DTVB1 … VWA2_VWB1`. So you're running the exact compiled binary that faults for us, with randomized inputs, and it doesn't fault. That actually narrows the problem a lot.

Same compiled kernel, fault vs. no-fault → the discriminator is the **invocation arguments**, not the kernel. The evidence that this kernel is usually fine is strong: your randomized-input stress test never faulted, and our run has ground through 2350 training steps (rotating data seeds → thousands of distinct calls) with only occasional faults. (I also captured `AMD_LOG_LEVEL=3` over 3 steps and saw the kernel dispatched cleanly across all layers — but at `batch_size=1` that's only ~3 distinct token-counts repeated per layer, so it's corroboration, not a real sweep.)

Our crash signature points the same way: the faulting access is a wild address ~1 GB outside both operands. An address that far off is hard to explain from *data values* — it looks like it comes from the **dimensions / leading dimensions / base pointers** fed to the kernel (the absolute fault address tracks the operand base across runs while the overshoot stays ~1 GB, which reads like index/offset arithmetic). So randomizing the data at a fixed shape, as I believe your stress loop did, would likely never hit it.

The arguments that change between our faulting calls are **M** (the per-step token count) and the **operand base pointers** (allocation state). Our two SYNC-deterministic captures faulted at **M = 607** and **M = 439**; N = 5376 and K = 21504 are fixed across every call, and the fault is deterministic for a fixed data seed (fixed seed → same step → same M and same allocation). M is my leading suspect, but I can't yet rule out the base-address/alignment angle — both co-vary in our runs.

## Exact production invocation

```
transA = N, transB = T          (B is the [N, K] weight, the DTVB1 path)
dtype  = bf16 in, fp32 accumulate, bf16 out
M = 607  (also seen: 439)        # per-step token count, varies with data order
N = 5376
K = 21504
lda = 21504   (A: (M, K) row-major contiguous)
ldb = 21504   (B: (21504, 5376) col-major, stride (1, 21504))
ldc = 5376    (C: (M, N) bf16)
bias = True   (fused Bias…UserArgs epilogue)
alpha = 1, beta = 0
```

## Asks

1. **Could you share your force-dispatch / stress script?** It already selects the right kernel — I'd like to plug in our exact production dims and leading dims above, rather than fight PyTorch's layout normalization (see note below). That's the fastest route to a self-contained repro for both of us.
2. **Stress the kernel by varying the invocation, not the data.** Two axes to vary (both change call-to-call in our workload, and randomizing input values at one fixed shape won't exercise either):
   - **M** — sweep M = 439, 607, 608 and neighbors (606–610), at N=5376, K=21504, transB, bf16, bias.
   - **Operand base address / preceding allocation pattern** — re-run at a faulting M with the A/B/C buffers placed at varied base offsets (e.g. churn the allocator first), since the wild address tracks the operand base.
   M is my leading suspect, but please vary both so an M-only sweep coming back clean doesn't falsely read as "not reproducible."
3. @lainlives has an independent repro with **no bitsandbytes** (pure PyTorch bf16, urvc voice-synthesis training) hitting the same `MT64x64x64` kernel on gfx1201. That's the most tractable end-to-end workload — running it under `rocprof --kernel-trace` would confirm the same kernel and hand you its real dispatch dims directly.

## Note — why I couldn't hand you a standalone PyTorch repro

I tried hard to reproduce the faulting kernel from a small PyTorch script (bitsandbytes `Linear4bit`, plain bf16 `nn.Linear`, and a raw col-major-`[K,N]` matmul). Every one of them dispatches a *different* contraction (`Cijk_Ailk_Bljk` / `Alik_Bljk`, tile `MT128x128x32`) — PyTorch normalizes the operand layout at the BLAS boundary, so I can't coax it into emitting the `Cijk_Ailk_Bjlk` contraction that production (and your force-dispatch) use. In the full workload it does appear — the faulting shape (in=21504, out=5376) is the MLP `down_proj`, run through stock HF transformers + PEFT-LoRA + bitsandbytes 4-bit + gradient checkpointing (no Unsloth runtime). This is why driving it from the rocBLAS layer with your script + the dims above is more promising than another PyTorch probe.

Probe scripts and full logs are in PR #101; glad to share any specific excerpt, or to dump the exact kernarg of the faulting dispatch if you can point me at the right way to capture it.

---

## Update (2026-06-15) — captured operand base pointers + the fault offset

@amd-nicknick — to help you point at this directly, here are the concrete pointers from the deterministic SYNC capture (`AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3`), taken from the `down_proj` operands logged immediately before the faulting `Cijk_Ailk_Bjlk … MT64x64x64 … DTVB1` dispatch:

```
A  (1, 607, 21504) bf16, row-major   base 0x7efad9200000  end 0x7efadaae5800   (M=607, K=21504)
B  (21504, 5376)   bf16, col-major   base 0x7ef978800000  end 0x7ef986480000   (stride (1,21504), the DTVB1 path)
fault address                        0x7efb33000000        (2 MB-aligned)
```

The fault is **above A** (it ends at `0x7efadaae5800`), not inside either operand:

```
fault − A = 0x59E00000 = +1.404 GiB  = exactly 719 × 2 MB   (note the round, page-aligned value)
fault − B = 0x1BA800000 = +6.914 GiB
```

`fault − A = 0x59E00000` being an exact 2 MB-aligned round number (719 pages) reads like a miscomputed offset rather than a data-dependent walk — which lines up with this being invocation-dependent, not data-dependent. I tried factoring the overshoot against the leading dims from these tensor pointers (`(fault−A)/2 ÷ K = 35059 rem 17408`, etc.) and nothing lands cleanly — but I suspect the kernel's *actual* operand bases (after any workspace/swizzle) differ from these PyTorch tensor pointers, so the clean factoring probably needs the real kernarg.

**So the most useful thing I can hand you is the literal kernarg of the faulting dispatch** — the UserArgs struct with the real base pointers, M, lda/ldb/ldc, alpha/beta, bias pointer and workspace. If you can point me at the right capture (an `AMD_LOG_LEVEL=4` filter, or an HSA/kernarg-pool dump hook), I'll grab it and post the exact bytes so you can replay the call 1:1.

Two more notes:
- **The full workload emits this kernel readily** — I don't have to coax PyTorch into it. A 3-step `AMD_LOG_LEVEL=3` capture of the real model (stock HF + PEFT-LoRA + bnb-4bit + grad-ckpt) dispatched `Cijk_Ailk_Bjlk … MT64x64x64 DTVB1` **1243×** (cleanly, since none of those calls hit a faulting M/allocation). So a repro driven from the model side is realistic; the layout-normalization problem only bites the *isolated* linear probes.
- Next on my side: a clean **M-sweep through the real model** at the `down_proj` shape (M ∈ {438–440, 606–610}, fwd+bwd, fixed allocation) and an **allocation-churn at fixed M=607** to isolate M from the base address — since those two co-vary in every capture I have. I'll report which one actually flips the fault.
