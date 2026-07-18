// Dendritron RBF forward pass HIP kernel (Euclidean single-chart, integration="sum")
//
// Math (mirrors src/dendritron/branches.py LocalBranch.activation/response and
// src/dendritron/primitive.py Dendritron.forward with integration="sum"):
//
//   distance[n,b]   = ||X[n,:] - center[b,:]||_2                       (Euclidean)
//   activation[n,b] = exp(-0.5 * (distance[n,b] / max(sigma[b], 1e-8))^2)
//   out[n,o]        = sum_b active[b] * activation[n,b] * output[b,o]
//
// Precision note: the reference (numpy) implementation is float64. This kernel
// computes in float32 for throughput, since gfx803 (Polaris / RX 580) has very
// weak double-precision throughput (~1/16 of fp32 FLOPs). Correctness is
// validated against the float64 numpy reference with atol=1e-4, rtol=1e-3,
// which float32 accumulation comfortably satisfies for the branch counts (B up
// to a few hundred) used here. If you need bit-exact float64 behavior, swap
// `float` for `double` throughout (host+device) at a significant throughput
// cost on this GPU.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                      \
  do {                                                                       \
    hipError_t _err = (expr);                                                \
    if (_err != hipSuccess) {                                                \
      fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__,        \
              __LINE__, hipGetErrorString(_err));                            \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// Max output dim supported via the register/local accumulator fast path.
// O is expected to be small (<= 64) for RBF-style branch layers.
#define MAX_O 64

__global__ void rbf_forward_kernel(const float* __restrict__ X,
                                    const float* __restrict__ centers,
                                    const float* __restrict__ sigmas,
                                    const float* __restrict__ outputs,
                                    const int* __restrict__ active_mask,
                                    float* __restrict__ out, int N, int B,
                                    int D, int O) {
  for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
       n += blockDim.x * gridDim.x) {
    float acc[MAX_O];
    for (int o = 0; o < O; ++o) acc[o] = 0.0f;

    const float* x_row = X + (size_t)n * D;

    for (int b = 0; b < B; ++b) {
      if (!active_mask[b]) continue;

      const float* c_row = centers + (size_t)b * D;
      float sumsq = 0.0f;
      for (int d = 0; d < D; ++d) {
        float diff = x_row[d] - c_row[d];
        sumsq += diff * diff;
      }
      float dist = sqrtf(sumsq);
      float sigma = sigmas[b];
      if (sigma < 1e-8f) sigma = 1e-8f;
      float ratio = dist / sigma;
      float activation = expf(-0.5f * ratio * ratio);

      const float* out_row = outputs + (size_t)b * O;
      for (int o = 0; o < O; ++o) {
        acc[o] += activation * out_row[o];
      }
    }

    float* out_row = out + (size_t)n * O;
    for (int o = 0; o < O; ++o) out_row[o] = acc[o];
  }
}

extern "C" {

// Host launcher. All array pointers are host pointers (float32 except
// active_mask which is int32, one entry per branch). Allocates device
// memory, copies in, launches, copies out, frees. Returns 0 on success,
// nonzero on failure (also prints diagnostics to stderr).
//
//   X:            [N, D] row-major
//   centers:      [B, D] row-major
//   sigmas:       [B]
//   outputs:      [B, O] row-major
//   active_mask:  [B]  (0/1 int32)
//   out:          [N, O] row-major, written by this call
int rbf_forward(const float* X, const float* centers, const float* sigmas,
                 const float* outputs, const int* active_mask, float* out,
                 int N, int B, int D, int O) {
  if (O > MAX_O) {
    fprintf(stderr, "rbf_forward: O=%d exceeds MAX_O=%d\n", O, MAX_O);
    return 1;
  }

  float *d_X = nullptr, *d_centers = nullptr, *d_sigmas = nullptr,
        *d_outputs = nullptr, *d_out = nullptr;
  int* d_active = nullptr;

  size_t bytes_X = (size_t)N * D * sizeof(float);
  size_t bytes_centers = (size_t)B * D * sizeof(float);
  size_t bytes_sigmas = (size_t)B * sizeof(float);
  size_t bytes_outputs = (size_t)B * O * sizeof(float);
  size_t bytes_active = (size_t)B * sizeof(int);
  size_t bytes_out = (size_t)N * O * sizeof(float);

  HIP_CHECK(hipMalloc(&d_X, bytes_X));
  HIP_CHECK(hipMalloc(&d_centers, bytes_centers));
  HIP_CHECK(hipMalloc(&d_sigmas, bytes_sigmas));
  HIP_CHECK(hipMalloc(&d_outputs, bytes_outputs));
  HIP_CHECK(hipMalloc(&d_active, bytes_active));
  HIP_CHECK(hipMalloc(&d_out, bytes_out));

  HIP_CHECK(hipMemcpy(d_X, X, bytes_X, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_centers, centers, bytes_centers, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sigmas, sigmas, bytes_sigmas, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_outputs, outputs, bytes_outputs, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_active, active_mask, bytes_active, hipMemcpyHostToDevice));

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;

  hipLaunchKernelGGL(rbf_forward_kernel, dim3(blocks), dim3(threads), 0, 0,
                      d_X, d_centers, d_sigmas, d_outputs, d_active, d_out, N,
                      B, D, O);

  hipError_t launch_err = hipGetLastError();
  if (launch_err != hipSuccess) {
    fprintf(stderr, "rbf_forward: kernel launch failed: %s\n",
            hipGetErrorString(launch_err));
    return 2;
  }

  hipError_t sync_err = hipDeviceSynchronize();
  if (sync_err != hipSuccess) {
    fprintf(stderr, "rbf_forward: kernel execution failed: %s\n",
            hipGetErrorString(sync_err));
    return 3;
  }

  HIP_CHECK(hipMemcpy(out, d_out, bytes_out, hipMemcpyDeviceToHost));

  hipFree(d_X);
  hipFree(d_centers);
  hipFree(d_sigmas);
  hipFree(d_outputs);
  hipFree(d_active);
  hipFree(d_out);

  return 0;
}

// Same as rbf_forward, but also reports GPU kernel-only elapsed time in
// milliseconds (via hipEvent timestamps that bracket just the launch +
// synchronize, excluding H2D/D2H copies) through the out-param kernel_ms.
// Used by bench_gpu_vs_cpu.py to separate "compute only" from
// "incl. transfer" timings.
int rbf_forward_timed(const float* X, const float* centers,
                       const float* sigmas, const float* outputs,
                       const int* active_mask, float* out, int N, int B,
                       int D, int O, float* kernel_ms) {
  if (O > MAX_O) {
    fprintf(stderr, "rbf_forward_timed: O=%d exceeds MAX_O=%d\n", O, MAX_O);
    return 1;
  }

  float *d_X = nullptr, *d_centers = nullptr, *d_sigmas = nullptr,
        *d_outputs = nullptr, *d_out = nullptr;
  int* d_active = nullptr;

  size_t bytes_X = (size_t)N * D * sizeof(float);
  size_t bytes_centers = (size_t)B * D * sizeof(float);
  size_t bytes_sigmas = (size_t)B * sizeof(float);
  size_t bytes_outputs = (size_t)B * O * sizeof(float);
  size_t bytes_active = (size_t)B * sizeof(int);
  size_t bytes_out = (size_t)N * O * sizeof(float);

  HIP_CHECK(hipMalloc(&d_X, bytes_X));
  HIP_CHECK(hipMalloc(&d_centers, bytes_centers));
  HIP_CHECK(hipMalloc(&d_sigmas, bytes_sigmas));
  HIP_CHECK(hipMalloc(&d_outputs, bytes_outputs));
  HIP_CHECK(hipMalloc(&d_active, bytes_active));
  HIP_CHECK(hipMalloc(&d_out, bytes_out));

  HIP_CHECK(hipMemcpy(d_X, X, bytes_X, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_centers, centers, bytes_centers, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sigmas, sigmas, bytes_sigmas, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_outputs, outputs, bytes_outputs, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_active, active_mask, bytes_active, hipMemcpyHostToDevice));

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, 0));
  hipLaunchKernelGGL(rbf_forward_kernel, dim3(blocks), dim3(threads), 0, 0,
                      d_X, d_centers, d_sigmas, d_outputs, d_active, d_out, N,
                      B, D, O);
  HIP_CHECK(hipEventRecord(stop, 0));

  hipError_t launch_err = hipGetLastError();
  if (launch_err != hipSuccess) {
    fprintf(stderr, "rbf_forward_timed: kernel launch failed: %s\n",
            hipGetErrorString(launch_err));
    return 2;
  }

  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  *kernel_ms = ms;

  hipEventDestroy(start);
  hipEventDestroy(stop);

  HIP_CHECK(hipMemcpy(out, d_out, bytes_out, hipMemcpyDeviceToHost));

  hipFree(d_X);
  hipFree(d_centers);
  hipFree(d_sigmas);
  hipFree(d_outputs);
  hipFree(d_active);
  hipFree(d_out);

  return 0;
}

}  // extern "C"
