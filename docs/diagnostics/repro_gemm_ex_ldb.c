// Based on the reproducer posted to ROCm/rocm-libraries#7992 on 2026-11-17,
// with M, transB, ldb and the B allocation made selectable.
// usage: repro_gemm M transB(N|T) ldb(K|N) iters
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK_HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP FAIL line %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)
#define CHECK_RB(x) do { rocblas_status s = (x); if (s != rocblas_status_success) { printf("ROCBLAS FAIL line %d: %d\n", __LINE__, (int)s); return 1; } } while (0)
int main(int argc, char **argv) {
  const int M = atoi(argv[1]), N = 5376, K = 21504;
  const int tB = argv[2][0] == 'T';
  const int ldb = argv[3][0] == 'K' ? K : N;
  const int iters = atoi(argv[4]);
  const int original_alloc = argc > 5;  // any 5th arg: allocate K*N elements as the posted code does
  // B is K x N (transB=N) or N x K (transB=T), column-major with leading dim ldb.
  const size_t need = tB ? (size_t)ldb * (K - 1) + N : (size_t)ldb * (N - 1) + K;
  const size_t sA = (size_t)M * K, sB = original_alloc ? (size_t)K * N : need, sC = (size_t)M * N;
  printf("M=%d transB=%c ldb=%d  B allocated %zu elems, BLAS contract needs %zu%s\n", M, tB ? 'T' : 'N', ldb, sB, need, sB < need ? "  <-- UNDER-ALLOCATED" : "");
  void *dA, *dB, *dC;
  CHECK_HIP(hipMalloc(&dA, sA * 2)); CHECK_HIP(hipMalloc(&dB, sB * 2)); CHECK_HIP(hipMalloc(&dC, sC * 2));
  CHECK_HIP(hipMemset(dA, 0x3c, sA * 2)); CHECK_HIP(hipMemset(dB, 0x3c, sB * 2)); CHECK_HIP(hipMemset(dC, 0, sC * 2));
  rocblas_handle h; CHECK_RB(rocblas_create_handle(&h));
  const float alpha = 1.0f, beta = 0.0f;
  for (int it = 0; it < iters; it++) {
    CHECK_RB(rocblas_gemm_ex(h, rocblas_operation_none, tB ? rocblas_operation_transpose : rocblas_operation_none,
        M, N, K, &alpha, dA, rocblas_datatype_bf16_r, M, dB, rocblas_datatype_bf16_r, ldb, &beta,
        dC, rocblas_datatype_bf16_r, M, dC, rocblas_datatype_bf16_r, M,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    CHECK_HIP(hipDeviceSynchronize());
  }
  printf("ALL CLEAN (%d iters)\n", iters);
  return 0;
}
