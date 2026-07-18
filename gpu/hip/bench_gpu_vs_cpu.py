"""Benchmark: numpy (float64, CPU) vs HIP kernel (float32, GPU) for the
Dendritron RBF forward pass (integration="sum", Euclidean single chart).

Sweeps N in [1000, 10000, 100000, 1000000] with fixed B=128, D=64, O=8.
Reports:
  - CPU (numpy float64) wall time
  - GPU compute-only time (calls into the .so's rbf_forward, which itself
    times host<->device copies + kernel; we separate "compute only" by
    timing just the kernel launch+sync portion via a compute-only C entry
    point) -- see note below on how this is measured.
  - GPU incl. transfer: wall-clock time of the full rbf_forward_gpu() python
    call, which is realistic end-to-end cost (H2D copy + kernel + D2H copy +
    ctypes overhead).

To get an honest "compute only" number without modifying the .so's ABI, we
add a second entry point in the shared library, `rbf_forward_timed`, that
returns kernel-only elapsed milliseconds via hipEvent timestamps, and we
call it directly via ctypes here.
"""

import time

import numpy as np

from dendritron_hip import rbf_forward_gpu, _lib
import ctypes

# Wire up the timed variant if present (see dendritron_rbf.hip.cpp).
_HAVE_TIMED = hasattr(_lib, "rbf_forward_timed")
if _HAVE_TIMED:
    _lib.rbf_forward_timed.restype = ctypes.c_int
    _lib.rbf_forward_timed.argtypes = _lib.rbf_forward.argtypes + [
        ctypes.POINTER(ctypes.c_float)  # out: kernel_ms
    ]


def numpy_reference(X, centers, sigmas, outputs, active):
    # Memory-efficient distance: avoids materializing the [N, B, D] diff
    # tensor (which is prohibitive for large N, e.g. N=1e6, B=128, D=64 would
    # be 61 GiB). Uses the expansion ||x - c||^2 = |x|^2 + |c|^2 - 2 x.c,
    # producing only an [N, B] intermediate.
    x_sq = np.sum(X * X, axis=1, keepdims=True)        # [N, 1]
    c_sq = np.sum(centers * centers, axis=1)             # [B]
    cross = X @ centers.T                                # [N, B]
    dist_sq = np.maximum(x_sq + c_sq[None, :] - 2.0 * cross, 0.0)
    distance = np.sqrt(dist_sq)
    sigma_safe = np.maximum(sigmas, 1e-8)
    activation = np.exp(-0.5 * (distance / sigma_safe[None, :]) ** 2)
    activation = activation * active[None, :].astype(np.float64)
    return activation @ outputs


def gpu_compute_only_ms(X, centers, sigmas, outputs, active):
    """Returns GPU kernel-only elapsed ms using the timed .so entry point,
    or None if that entry point isn't available."""
    if not _HAVE_TIMED:
        return None
    N, D = X.shape
    B, O = outputs.shape

    def f32p(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    X_f, X_p = f32p(X)
    c_f, c_p = f32p(centers)
    s_f, s_p = f32p(sigmas)
    o_f, o_p = f32p(outputs)
    active_i = np.ascontiguousarray(active.astype(np.int32))
    active_p = active_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    out = np.empty((N, O), dtype=np.float32)
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ms = ctypes.c_float(0.0)

    rc = _lib.rbf_forward_timed(
        X_p, c_p, s_p, o_p, active_p, out_p, N, B, D, O, ctypes.byref(ms)
    )
    if rc != 0:
        raise RuntimeError(f"rbf_forward_timed failed rc={rc}")
    return ms.value


def bench_one(N, B, D, O, seed, cpu_repeats, gpu_repeats):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float64)
    centers = rng.normal(size=(B, D)).astype(np.float64)
    sigmas = rng.uniform(0.1, 3.0, size=B).astype(np.float64)
    outputs = rng.normal(size=(B, O)).astype(np.float64)
    active = (rng.uniform(size=B) > 0.2)

    # CPU timing (numpy float64)
    # warmup
    _ = numpy_reference(X, centers, sigmas, outputs, active)
    t0 = time.perf_counter()
    for _ in range(cpu_repeats):
        cpu_out = numpy_reference(X, centers, sigmas, outputs, active)
    t1 = time.perf_counter()
    cpu_ms = (t1 - t0) / cpu_repeats * 1000.0

    # GPU timing incl. transfer (full python call, ctypes + H2D + kernel + D2H)
    _ = rbf_forward_gpu(X, centers, sigmas, outputs, active)  # warmup (loads lib etc.)
    t0 = time.perf_counter()
    for _ in range(gpu_repeats):
        gpu_out = rbf_forward_gpu(X, centers, sigmas, outputs, active)
    t1 = time.perf_counter()
    gpu_incl_ms = (t1 - t0) / gpu_repeats * 1000.0

    # GPU compute-only (kernel launch+sync, via hipEvent timing in the .so)
    compute_ms_samples = []
    if _HAVE_TIMED:
        gpu_compute_only_ms(X, centers, sigmas, outputs, active)  # warmup
        for _ in range(gpu_repeats):
            compute_ms_samples.append(
                gpu_compute_only_ms(X, centers, sigmas, outputs, active)
            )
    gpu_compute_ms = np.mean(compute_ms_samples) if compute_ms_samples else None

    max_err = float(np.max(np.abs(gpu_out.astype(np.float64) - cpu_out)))

    return {
        "N": N,
        "cpu_ms": cpu_ms,
        "gpu_incl_ms": gpu_incl_ms,
        "gpu_compute_ms": gpu_compute_ms,
        "speedup_incl": cpu_ms / gpu_incl_ms,
        "speedup_compute": (cpu_ms / gpu_compute_ms) if gpu_compute_ms else None,
        "max_err": max_err,
    }


def main():
    B, D, O = 128, 64, 8
    sizes = [1000, 10000, 100000, 1000000]
    results = []
    for N in sizes:
        # fewer repeats for very large N to keep runtime sane
        cpu_repeats = 5 if N <= 100000 else 2
        gpu_repeats = 10 if N <= 100000 else 3
        r = bench_one(N, B, D, O, seed=N, cpu_repeats=cpu_repeats, gpu_repeats=gpu_repeats)
        results.append(r)
        print(
            f"N={r['N']:>8} | CPU(numpy f64): {r['cpu_ms']:>10.3f} ms | "
            f"GPU incl.transfer: {r['gpu_incl_ms']:>10.3f} ms "
            f"(speedup {r['speedup_incl']:.2f}x) | "
            + (
                f"GPU compute-only: {r['gpu_compute_ms']:>10.3f} ms "
                f"(speedup {r['speedup_compute']:.2f}x) | "
                if r["gpu_compute_ms"] is not None
                else "GPU compute-only: n/a | "
            )
            + f"max_err={r['max_err']:.2e}"
        )

    print("\n=== Summary table (B=128, D=64, O=8) ===")
    header = f"{'N':>10} | {'CPU ms':>10} | {'GPU incl ms':>12} | {'speedup(incl)':>14} | {'GPU compute ms':>15} | {'speedup(compute)':>17}"
    print(header)
    print("-" * len(header))
    for r in results:
        comp_ms = f"{r['gpu_compute_ms']:.3f}" if r["gpu_compute_ms"] is not None else "n/a"
        comp_sp = f"{r['speedup_compute']:.2f}x" if r["speedup_compute"] is not None else "n/a"
        print(
            f"{r['N']:>10} | {r['cpu_ms']:>10.3f} | {r['gpu_incl_ms']:>12.3f} | "
            f"{r['speedup_incl']:>13.2f}x | {comp_ms:>15} | {comp_sp:>17}"
        )


if __name__ == "__main__":
    main()
