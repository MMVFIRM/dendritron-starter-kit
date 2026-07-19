"""Correctness test for the Dendritron RBF HIP kernel vs a float64 numpy
reference implementation.

The reference reimplements the exact formula from
src/dendritron/branches.py (LocalBranch.activation / .response) and
src/dendritron/primitive.py (Dendritron.forward, integration="sum"),
restricted to the Euclidean single-chart case:

    distance[n,b]   = ||X[n,:] - center[b,:]||_2
    activation[n,b] = exp(-0.5 * (distance[n,b] / max(sigma[b], 1e-8))**2)
    out[n,o]        = sum_b active[b] * activation[n,b] * output[b,o]
"""

import numpy as np

from dendritron_hip import rbf_forward_gpu


def numpy_reference(X, centers, sigmas, outputs, active):
    X = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    active = np.asarray(active, dtype=bool)

    # [N, B, D] pairwise diffs -> [N, B] distances
    diff = X[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    sigma_safe = np.maximum(sigmas, 1e-8)
    activation = np.exp(-0.5 * (distance / sigma_safe[None, :]) ** 2)
    activation = activation * active[None, :].astype(np.float64)
    out = activation @ outputs
    return out


def run_case(name, N, B, D, O, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float64)
    centers = rng.normal(size=(B, D)).astype(np.float64)
    sigmas = rng.uniform(0.1, 3.0, size=B).astype(np.float64)
    outputs = rng.normal(size=(B, O)).astype(np.float64)
    active = (rng.uniform(size=B) > 0.2)  # ~80% active, some inactive

    ref = numpy_reference(X, centers, sigmas, outputs, active)
    gpu = rbf_forward_gpu(X, centers, sigmas, outputs, active)

    assert gpu.shape == ref.shape, f"{name}: shape mismatch {gpu.shape} vs {ref.shape}"

    ok = np.allclose(gpu, ref, atol=1e-4, rtol=1e-3)
    max_abs = np.max(np.abs(gpu.astype(np.float64) - ref))
    max_rel = np.max(np.abs(gpu.astype(np.float64) - ref) / (np.abs(ref) + 1e-8))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: N={N} B={B} D={D} O={O} "
          f"max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e}")
    assert ok, f"{name}: GPU result does not match numpy reference within tolerance"


def test_all_inactive():
    # Edge case: all branches inactive -> output should be exactly zero.
    N, B, D, O = 16, 8, 4, 3
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, D))
    centers = rng.normal(size=(B, D))
    sigmas = rng.uniform(0.1, 2.0, size=B)
    outputs = rng.normal(size=(B, O))
    active = np.zeros(B, dtype=bool)

    gpu = rbf_forward_gpu(X, centers, sigmas, outputs, active)
    assert np.allclose(gpu, 0.0), "all-inactive case should produce all-zero output"
    print("[PASS] all_inactive: N=16 B=8 D=4 O=3 -> exact zero output")


def test_exact_center_match():
    # Edge case: sample exactly at a branch center -> distance 0, activation 1.
    N, B, D, O = 1, 3, 5, 2
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(B, D))
    X = centers[0:1, :].copy()  # sample lands exactly on branch 0's center
    sigmas = np.array([1.0, 1.0, 1.0])
    outputs = rng.normal(size=(B, O))
    active = np.array([True, True, True])

    ref = numpy_reference(X, centers, sigmas, outputs, active)
    gpu = rbf_forward_gpu(X, centers, sigmas, outputs, active)
    assert np.allclose(gpu, ref, atol=1e-4, rtol=1e-3)
    print(f"[PASS] exact_center_match: gpu={gpu[0]} ref={ref[0]}")


def main():
    run_case("small", N=50, B=16, D=8, O=4, seed=42)
    run_case("medium", N=2000, B=64, D=32, O=8, seed=7)
    run_case("large_branches", N=500, B=512, D=16, O=1, seed=99)
    run_case("large_dim", N=500, B=32, D=256, O=8, seed=123)
    run_case("bench_shape", N=10000, B=128, D=64, O=8, seed=2026)
    test_all_inactive()
    test_exact_center_match()
    print("\nALL CORRECTNESS TESTS PASSED")


if __name__ == "__main__":
    main()
