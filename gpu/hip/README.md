# HIP kernel: batched RBF forward pass

Fused GPU implementation of the hot path in `Dendritron.forward` with
`integration="sum"` and a single Euclidean chart per branch — the same math as
`LocalBranch.activation` / `.response` in `src/dendritron/branches.py`.

```
distance[n,b]   = ||X[n,:] - center[b,:]||_2
activation[n,b] = exp(-0.5 * (distance[n,b] / max(sigma[b], 1e-8))^2)
out[n,o]        = sum_b active[b] * activation[n,b] * output[b,o]
```

## Requirements

- ROCm HIP (`hipcc`, `/dev/kfd`, membership in `render` group)
- Python 3.10+ with NumPy
- AMD GPU with HIP support (validated on **gfx803 / RX 580**, ROCm 6.3)

## Build

```bash
cd gpu/hip
make
# if hipcc cannot find <cmath>, point at your installed gcc libstdc++ tree:
make GCC_INSTALL_DIR=/usr/lib/gcc/x86_64-linux-gnu/11
```

Produces `libdendritron_rbf.so` beside the source.

## Test and benchmark

```bash
make test    # 7 correctness cases vs float64 numpy reference
make bench   # CPU vs GPU sweep (B=128, D=64, O=8)
```

Python API:

```python
from dendritron_hip import rbf_forward_gpu

out = rbf_forward_gpu(X, centers, sigmas, outputs, active)  # -> [N, O] float32
```

## Precision

The reference package uses float64 NumPy. The kernel uses **float32 on-device**
because Polaris (gfx803) has ~1/16 fp64 throughput. Correctness is validated at
`atol=1e-4`, `rtol=1e-3`.

## Benchmark (RX 580 / gfx803, ROCm 6.3, Ryzen 9 3900X)

Fixed `B=128, D=64, O=8`, sweeping `N`. CPU is the float64 NumPy reference
(which reduces to a BLAS `X·centersᵀ` GEMM via the `‖x−c‖² = |x|²+|c|²−2x·c`
expansion, so it saturates all 24 cores). "compute-only" excludes H2D/D2H.
Numbers re-measured on a quiet box with GPU clocks pinned high
(`rocm-smi --setperflevel high`).

| N | CPU f64 (ms) | GPU incl. transfer | speedup | GPU compute-only | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 4.37 | 2.61 | 1.67× | 2.35 | 1.86× |
| 10,000 | 41.8 | 9.10 | 4.59× | 7.81 | 5.35× |
| 100,000 | 312.4 | 87.1 | 3.59× | 77.9 | 4.01× |
| 1,000,000 | 3116 | 646 | 4.82× | 561 | **5.55×** |

**Where it wins:** the kernel is faster than the multithreaded BLAS reference
at every tested size — peak ~4.6× (end-to-end) / ~5.5× (compute-only). Small
`N` (~10³) is bounded by fixed launch + PCIe transfer (~1.7×); everything from
`N≥10⁴` is a solid 3.6–4.8× including transfer.

**How the large-`N` regime was fixed:** the first cut was a one-thread-per-
sample kernel that re-read every branch center from global memory per sample
(`O(N·B·D)` redundant loads), so at `N=10⁶` it was memory-bound and only
~1.1× over the CPU BLAS GEMM. The current kernel stages branch tiles
(`centers`/`outputs`/`sigmas`/`active`) into shared memory once per block and
reuses them across the block's samples, cutting global center traffic by
~`blockDim` (256×) and turning large `N` compute-bound: `N=10⁶` compute-only
went 3015 ms → 561 ms (**5.4× kernel speedup**, 1.1× → 5.55× vs CPU).

## Scope

This kernel accelerates the **core dendritron RBF forward pass**. It is
independent of the separate SmolLM2/LoRA experiment, which runs its own
hand-written HIP kernels precisely because stock ROCm/PyTorch wheels do **not**
execute on gfx803 (Polaris) — `hipErrorNoBinaryForGpu` / "invalid device
function". There is no `torch` fast path on this hardware; custom HIP is the
only option.
