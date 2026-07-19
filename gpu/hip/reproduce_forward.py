"""Closes the loop for PR #2: the HIP kernel reproduces the repository's OWN
`Dendritron.forward` (integration="sum", single Euclidean chart) on the RX 580,
not merely a standalone numpy reference.

Builds a real `Dendritron` from `src/dendritron`, populates it with random
`LocalBranch` compartments (some damaged/inactive), runs the library forward
pass (float64, CPU), then runs the same inputs through the gfx803 HIP kernel
and compares.
"""

import sys

import numpy as np

sys.path.insert(0, "gpu/hip")

from dendritron import Dendritron, LocalBranch  # noqa: E402
from dendritron_hip import rbf_forward_gpu  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(20260719)
    N, B, D, O = 5000, 96, 48, 6

    centers = rng.standard_normal((B, D))
    outputs = rng.standard_normal((B, O))
    sigmas = rng.uniform(0.3, 2.5, size=B)
    active = rng.random(B) > 0.25  # ~25% damaged branches

    dend = Dendritron(D, O, name="repro", integration="sum")
    for b in range(B):
        dend.add_branch(
            LocalBranch(
                owner="repro",
                center=centers[b],
                output=outputs[b],
                sigma=float(sigmas[b]),
                branch_id=f"b{b}",
                active=bool(active[b]),
            )
        )

    X = rng.standard_normal((N, D))

    ref = dend.forward(X)                                   # repo's own path, f64 CPU
    gpu = rbf_forward_gpu(X, centers, sigmas, outputs, active)  # RX 580 HIP kernel

    max_abs = float(np.max(np.abs(gpu.astype(np.float64) - ref)))
    denom = np.maximum(np.abs(ref), 1e-6)
    max_rel = float(np.max(np.abs(gpu.astype(np.float64) - ref) / denom))

    # Also confirm the library predict() decision is unchanged on the f32 output.
    ref_arg = ref.argmax(axis=1)
    gpu_arg = gpu.argmax(axis=1)
    argmatch = float(np.mean(ref_arg == gpu_arg))

    print(f"shapes: X={X.shape} centers={centers.shape} O={O} "
          f"active={int(active.sum())}/{B}")
    print(f"Dendritron.forward vs HIP kernel: max_abs_err={max_abs:.3e} "
          f"max_rel_err={max_rel:.3e}  argmax_agreement={argmatch:.4f}")

    ok = np.allclose(gpu.astype(np.float64), ref, atol=1e-4, rtol=1e-3)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
