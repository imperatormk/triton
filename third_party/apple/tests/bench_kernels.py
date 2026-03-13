"""
Apple MPS Triton backend -- kernel benchmarks.

Measures GFLOPS for GEMM at various tile sizes, plus element-wise throughput.
Uses wall-clock timing with torch.mps.synchronize() (Apple driver lacks
timed events needed by triton.testing.do_bench).

Run:
  python third_party/apple/tests/bench_kernels.py
"""
import torch
import time
import triton
import triton.language as tl

DEVICE = "mps"


# ============================================================================
# Kernels
# ============================================================================

@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = N // BN
    pid_m = pid // num_n
    pid_n = pid % num_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(B_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)
    tl.store(C_ptr + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def add_kernel(x, y, out, N: tl.constexpr):
    offs = tl.program_id(0) * N + tl.arange(0, N)
    tl.store(out + offs, tl.load(x + offs) + tl.load(y + offs))


# ============================================================================
# Timing helper
# ============================================================================

def bench(fn, warmup=20, iters=100):
    """Returns elapsed time in microseconds per call."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ============================================================================
# Benchmarks
# ============================================================================

def bench_gemm():
    configs = [
        # (M, N, K, BM, BN, BK)
        (256, 256, 256, 16, 16, 16),
        (256, 256, 256, 32, 32, 32),
        (512, 512, 512, 32, 32, 32),
        (1024, 1024, 1024, 32, 32, 32),
        (128, 128, 64, 128, 128, 64),
    ]

    print("GEMM (Triton)")
    print(f"  {'M':>5} {'N':>5} {'K':>5}  {'tile':>12} {'us':>9} {'GFLOPS':>8}")
    print(f"  {'-'*50}")

    for M, N, K, BM, BN, BK in configs:
        A = torch.randn(M, K, device=DEVICE)
        B = torch.randn(K, N, device=DEVICE)
        C = torch.zeros(M, N, device=DEVICE)
        grid = ((M // BM) * (N // BN),)
        tile = f"{BM}x{BN}x{BK}"

        try:
            us = bench(lambda: gemm_kernel[grid](A, B, C, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK))
            gflops = 2 * M * N * K / us / 1e3
            print(f"  {M:>5} {N:>5} {K:>5}  {tile:>12} {us:>9.1f} {gflops:>8.1f}")
        except Exception as e:
            print(f"  {M:>5} {N:>5} {K:>5}  {tile:>12} {'FAIL':>9}  {str(e)[:40]}")

    print()
    print("GEMM (torch.mm reference)")
    print(f"  {'size':>12} {'us':>9} {'GFLOPS':>8}")
    print(f"  {'-'*32}")
    for sz in [256, 512, 1024]:
        A = torch.randn(sz, sz, device=DEVICE)
        B = torch.randn(sz, sz, device=DEVICE)
        us = bench(lambda: A @ B)
        gflops = 2 * sz**3 / us / 1e3
        print(f"  {sz:>5}³     {us:>9.1f} {gflops:>8.1f}")


def bench_elementwise():
    print()
    print("Element-wise add (Triton)")
    print(f"  {'N':>10} {'us':>9} {'GB/s':>8}")
    print(f"  {'-'*30}")
    for N in [1024, 4096, 16384, 65536]:
        x = torch.randn(N, device=DEVICE)
        y = torch.randn(N, device=DEVICE)
        out = torch.empty(N, device=DEVICE)
        us = bench(lambda: add_kernel[(1,)](x, y, out, N=N))
        # 3 tensors * N * 4 bytes (read x, read y, write out)
        gbs = 3 * N * 4 / us / 1e3
        print(f"  {N:>10} {us:>9.1f} {gbs:>8.2f}")


if __name__ == "__main__":
    bench_gemm()
    bench_elementwise()
