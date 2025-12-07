// mamba_ssm/ops/csrc/recurrent_mamba_step_cuda.cu
//
// Real CUDA kernel for recurrent Mamba step.
//
// This implements a 1D Mamba-like SSM+conv cell per feature:
//   - Conv window: last K-1 inputs plus current x_t
//   - Conv output: u = sum_k W_conv[d,k] * w_k + b_conv[d]
//   - SSM update:  z = A[d] * s_prev + B[d] * u
//   - Output:      y = C[d] * z + D_skip[d] * u
//
// State: ssm_state is (B, D) - one scalar per feature
//        conv_state is (B, D, K-1) - rolling window

#include <torch/extension.h>

namespace mamba {

// -----------------------------------------------------------------------------
// Forward kernel: one thread per (b, d)
// -----------------------------------------------------------------------------

__global__ void recurrent_mamba_step_forward_kernel(
    const float *__restrict__ x_t,        // (B, D)
    const float *__restrict__ conv_state, // (B, D, K-1)
    const float *__restrict__ ssm_state,  // (B, D)
    const float *__restrict__ W_conv,     // (D, K)
    const float *__restrict__ b_conv,     // (D)
    const float *__restrict__ A,          // (D)
    const float *__restrict__ Bp,         // (D)
    const float *__restrict__ C,          // (D)
    const float *__restrict__ D_skip,     // (D)
    float *__restrict__ y_t,              // (B, D)
    float *__restrict__ new_conv_state,   // (B, D, K-1)
    float *__restrict__ new_ssm_state,    // (B, D)
    int B, int D, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * D;
  if (idx >= total)
    return;

  int b = idx / D;
  int d = idx % D;

  // Pointers / strides
  const int stride_x_bd = D;
  const int stride_conv_b = D * (K - 1);
  const int stride_conv_bd = (K - 1);
  const int stride_W_d = K;

  const float *x_ptr = x_t + b * stride_x_bd;
  const float *conv_bd = conv_state + b * stride_conv_b + d * stride_conv_bd;
  const float *ssm_bd = ssm_state + b * D;

  float *y_ptr = y_t + b * D;
  float *new_conv_bd = new_conv_state + b * stride_conv_b + d * stride_conv_bd;
  float *new_ssm_bd = new_ssm_state + b * D;

  // 1) Build conv window w[0..K-1]:
  //    w[0..K-2] = old conv_state[b,d,0..K-2]
  //    w[K-1]    = x_t[b,d]
  float x_val = x_ptr[d];

  // 2) Convolution: u = sum_k W_conv[d,k] * w_k + b_conv[d]
  float u = 0.0f;
  const float *Wd = W_conv + d * stride_W_d;
  for (int k = 0; k < K - 1; ++k) {
    float w_k = conv_bd[k];
    u += Wd[k] * w_k;
  }
  u += Wd[K - 1] * x_val;
  u += b_conv[d];

  // 3) SSM: z = A[d] * s_prev + B[d] * u
  float s_prev = ssm_bd[d];
  float z = A[d] * s_prev + Bp[d] * u;

  // 4) Output: y = C[d] * z + D_skip[d] * u
  float y = C[d] * z + D_skip[d] * u;

  // 5) Update conv_state: shift window and append x_t
  // new_conv_state[b,d,0..K-2] = [w_1..w_{K-1}] = last K-1 of (old conv_state +
  // x_t) i.e. new_conv[0..K-3] = old conv[1..K-2], new_conv[K-2] = x_t.
  if (K > 1) {
    for (int k = 0; k < K - 2; ++k) {
      new_conv_bd[k] = conv_bd[k + 1];
    }
    new_conv_bd[K - 2] = x_val;
  }

  // 6) Update ssm_state
  new_ssm_bd[d] = z;

  // 7) Write output
  y_ptr[d] = y;
}

// -----------------------------------------------------------------------------
// Backward kernel: one thread per (b, d)
// -----------------------------------------------------------------------------

__global__ void recurrent_mamba_step_backward_kernel(
    const float *__restrict__ grad_y,         // (B, D)
    const float *__restrict__ x_t,            // (B, D)
    const float *__restrict__ conv_state,     // (B, D, K-1)
    const float *__restrict__ ssm_state,      // (B, D)
    const float *__restrict__ W_conv,         // (D, K)
    const float *__restrict__ b_conv,         // (D)
    const float *__restrict__ A,              // (D)
    const float *__restrict__ Bp,             // (D)
    const float *__restrict__ C,              // (D)
    const float *__restrict__ D_skip,         // (D)
    const float *__restrict__ y_t,            // (B, D)
    const float *__restrict__ new_conv_state, // (B, D, K-1)  [unused in math
                                              // but kept for symmetry]
    const float *__restrict__ new_ssm_state,  // (B, D)       [= z]
    // grads:
    float *__restrict__ grad_x,          // (B, D)
    float *__restrict__ grad_conv_state, // (B, D, K-1)
    float *__restrict__ grad_ssm_state,  // (B, D)
    float *__restrict__ grad_W_conv,     // (D, K)
    float *__restrict__ grad_b_conv,     // (D)
    float *__restrict__ grad_A,          // (D)
    float *__restrict__ grad_Bp,         // (D)
    float *__restrict__ grad_C,          // (D)
    float *__restrict__ grad_D_skip,     // (D)
    int B, int D, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * D;
  if (idx >= total)
    return;

  int b = idx / D;
  int d = idx % D;

  const int stride_bd = D;
  const int stride_conv_b = D * (K - 1);
  const int stride_conv_bd = (K - 1);
  const int stride_W_d = K;

  const float *x_ptr = x_t + b * stride_bd;
  const float *conv_bd = conv_state + b * stride_conv_b + d * stride_conv_bd;
  const float *ssm_bd = ssm_state + b * D;
  const float *gy_ptr = grad_y + b * D;

  const float *Wd = W_conv + d * stride_W_d;

  float *gx_ptr = grad_x + b * D;
  float *gconv_bd = grad_conv_state + b * stride_conv_b + d * stride_conv_bd;
  float *gssm_bd = grad_ssm_state + b * D;

  // 1) Recompute local intermediates (u,z) for this (b,d)
  //    Same as in forward.
  float x_val = x_ptr[d];

  float u = 0.0f;
  for (int k = 0; k < K - 1; ++k) {
    float w_k = conv_bd[k];
    u += Wd[k] * w_k;
  }
  u += Wd[K - 1] * x_val;
  u += b_conv[d];

  float s_prev = ssm_bd[d];
  float z = A[d] * s_prev + Bp[d] * u;

  // 2) Backprop through y = C[d]*z + D_skip[d]*u
  float gy = gy_ptr[d];

  float gu = gy * D_skip[d]; // dL/du from y
  float gz = gy * C[d];      // dL/dz from y

  // accumulate param grads: C and D_skip
  atomicAdd(&grad_C[d], gy * z);
  atomicAdd(&grad_D_skip[d], gy * u);

  // 3) Backprop through z = A[d]*s_prev + B[d]*u
  //    z depends on s_prev and u.
  atomicAdd(&grad_A[d], gz * s_prev);
  atomicAdd(&grad_Bp[d], gz * u);

  float gs_prev = gz * A[d]; // dL/ds_prev
  gu += gz * Bp[d];          // accumulate into dL/du

  // 4) Backprop through u = sum_k W[d,k]*w_k + b_conv[d]
  //    u depends on W_conv[d,k], b_conv[d], w_k (window entries).
  atomicAdd(&grad_b_conv[d], gu);

  // w_k: k=0..K-2 -> conv_state[b,d,k]; k=K-1 -> x_t[b,d].
  float gx_val = 0.0f;

  for (int k = 0; k < K - 1; ++k) {
    float w_k = conv_bd[k];
    float Wdk = Wd[k];

    // grad W_conv[d,k]
    atomicAdd(&grad_W_conv[d * stride_W_d + k], gu * w_k);

    // grad w_k (goes to conv_state)
    gconv_bd[k] += gu * Wdk;
  }

  // k = K-1 (x_t)
  {
    float Wdk = Wd[K - 1];
    atomicAdd(&grad_W_conv[d * stride_W_d + (K - 1)], gu * x_val);
    gx_val += gu * Wdk;
  }

  // 5) Accumulate gradients into outputs
  gx_ptr[d] += gx_val;
  gssm_bd[d] += gs_prev;
}

// -----------------------------------------------------------------------------
// Host wrappers
// -----------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_forward_cuda(const at::Tensor &x_t,        // (B, D)
                                  const at::Tensor &conv_state, // (B, D, K-1)
                                  const at::Tensor &ssm_state,  // (B, D)
                                  const at::Tensor &W_conv,     // (D, K)
                                  const at::Tensor &b_conv,     // (D)
                                  const at::Tensor &A,          // (D)
                                  const at::Tensor &Bp,         // (D)
                                  const at::Tensor &C,          // (D)
                                  const at::Tensor &D_skip      // (D)
) {
  TORCH_CHECK(x_t.is_cuda(), "x_t must be CUDA");
  auto B = x_t.size(0);
  auto D = x_t.size(1);
  auto K = W_conv.size(1);

  at::Tensor y_t = at::zeros_like(x_t);
  at::Tensor new_conv_state = at::zeros_like(conv_state);
  at::Tensor new_ssm_state = at::zeros_like(ssm_state);

  int threads = 256;
  int blocks = (B * D + threads - 1) / threads;

  recurrent_mamba_step_forward_kernel<<<blocks, threads>>>(
      x_t.data_ptr<float>(), conv_state.data_ptr<float>(),
      ssm_state.data_ptr<float>(), W_conv.data_ptr<float>(),
      b_conv.data_ptr<float>(), A.data_ptr<float>(), Bp.data_ptr<float>(),
      C.data_ptr<float>(), D_skip.data_ptr<float>(), y_t.data_ptr<float>(),
      new_conv_state.data_ptr<float>(), new_ssm_state.data_ptr<float>(), (int)B,
      (int)D, (int)K);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch error: ", cudaGetErrorString(err));

  return std::make_tuple(y_t, new_conv_state, new_ssm_state);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_backward_cuda(
    const at::Tensor &grad_y,         // (B, D)
    const at::Tensor &x_t,            // (B, D)
    const at::Tensor &conv_state,     // (B, D, K-1)
    const at::Tensor &ssm_state,      // (B, D)
    const at::Tensor &W_conv,         // (D, K)
    const at::Tensor &b_conv,         // (D)
    const at::Tensor &A,              // (D)
    const at::Tensor &Bp,             // (D)
    const at::Tensor &C,              // (D)
    const at::Tensor &D_skip,         // (D)
    const at::Tensor &y_t,            // (B, D)
    const at::Tensor &new_conv_state, // (B, D, K-1)
    const at::Tensor &new_ssm_state   // (B, D)
) {
  TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
  auto B = x_t.size(0);
  auto D = x_t.size(1);
  auto K = W_conv.size(1);

  at::Tensor grad_x = at::zeros_like(x_t);
  at::Tensor grad_conv_state = at::zeros_like(conv_state);
  at::Tensor grad_ssm_state = at::zeros_like(ssm_state);

  at::Tensor grad_W_conv = at::zeros_like(W_conv);
  at::Tensor grad_b_conv = at::zeros_like(b_conv);
  at::Tensor grad_A = at::zeros_like(A);
  at::Tensor grad_Bp = at::zeros_like(Bp);
  at::Tensor grad_C = at::zeros_like(C);
  at::Tensor grad_D_skip = at::zeros_like(D_skip);

  int threads = 256;
  int blocks = (B * D + threads - 1) / threads;

  recurrent_mamba_step_backward_kernel<<<blocks, threads>>>(
      grad_y.data_ptr<float>(), x_t.data_ptr<float>(),
      conv_state.data_ptr<float>(), ssm_state.data_ptr<float>(),
      W_conv.data_ptr<float>(), b_conv.data_ptr<float>(), A.data_ptr<float>(),
      Bp.data_ptr<float>(), C.data_ptr<float>(), D_skip.data_ptr<float>(),
      y_t.data_ptr<float>(), new_conv_state.data_ptr<float>(),
      new_ssm_state.data_ptr<float>(), grad_x.data_ptr<float>(),
      grad_conv_state.data_ptr<float>(), grad_ssm_state.data_ptr<float>(),
      grad_W_conv.data_ptr<float>(), grad_b_conv.data_ptr<float>(),
      grad_A.data_ptr<float>(), grad_Bp.data_ptr<float>(),
      grad_C.data_ptr<float>(), grad_D_skip.data_ptr<float>(), (int)B, (int)D,
      (int)K);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch error: ", cudaGetErrorString(err));

  return std::make_tuple(grad_x, grad_conv_state, grad_ssm_state, grad_W_conv,
                         grad_b_conv, grad_A, grad_Bp, grad_C, grad_D_skip);
}

} // namespace mamba
