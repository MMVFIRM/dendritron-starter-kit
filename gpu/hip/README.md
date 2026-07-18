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

## Benchmark (RX 580, ROCm 6.3)

| N | CPU ms (f64) | GPU incl. transfer | speedup | GPU compute only | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 6.5 | 2.5 | 2.6× | 2.2 | 3.0× |
| 10,000 | 73 | 8.7 | 8.5× | 7.7 | 9.6× |
| 100,000 | 674 | 308 | 2.2× | 300 | 2.3× |
| 1,000,000 | 6776 | 3095 | 2.2× | 2962 | 2.3× |

Transfer overhead is small relative to compute at large N; the kernel wins at
every tested size on this hardware.

## Scope

This kernel accelerates the **core dendritron RBF forward pass**, not the
SmolLM2/LoRA Transformer benchmark (that path is already handled by PyTorch's
ROCm backend when a compatible `torch` build is installed).
