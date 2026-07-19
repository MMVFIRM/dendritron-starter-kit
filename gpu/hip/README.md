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
| 1,000 | 4.19 | 2.57 | 1.63× | 2.30 | 1.82× |
| 10,000 | 40.4 | 9.04 | 4.47× | 7.76 | **5.21×** |
| 100,000 | 318.8 | 308.9 | 1.03× | 299.4 | 1.06× |
| 1,000,000 | 3357.6 | 3096.9 | 1.08× | 3014.6 | 1.11× |

**Where it wins:** the sweet spot is mid-range `N` (~10⁴), where the kernel is
~4.5–5× faster end-to-end. At small `N` fixed launch + PCIe transfer overhead
dominates (~1.6×). At very large `N` the CPU reference is a multithreaded BLAS
GEMM, and this straightforward one-thread-per-sample kernel (which re-reads all
branch centers from global memory per sample, with no shared-memory tiling) is
memory-bound and only marginally ahead (~1.1×). Tiling the `centers`/`outputs`
into shared memory is the obvious next step to make the large-`N` regime
compute-bound and competitive with BLAS — see the PR description.

## Scope

This kernel accelerates the **core dendritron RBF forward pass**. It is
independent of the separate SmolLM2/LoRA experiment, which runs its own
hand-written HIP kernels precisely because stock ROCm/PyTorch wheels do **not**
execute on gfx803 (Polaris) — `hipErrorNoBinaryForGpu` / "invalid device
function". There is no `torch` fast path on this hardware; custom HIP is the
only option.
