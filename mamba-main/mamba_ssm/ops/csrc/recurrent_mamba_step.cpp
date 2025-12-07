// mamba_ssm/ops/csrc/recurrent_mamba_step.cpp
//
// C++ dispatch layer for recurrent Mamba step kernel.
// This implements a true O(T) streaming Mamba cell.
//
// The kernel uses separate parameter tensors:
//   - W_conv, b_conv: convolution weights and bias
//   - A, B, C, D_skip: SSM parameters

#include <torch/extension.h>
#include <tuple>

namespace mamba {

// ============================================================================
// Forward declarations - CUDA kernels (implemented in .cu file)
// ============================================================================

#ifdef WITH_CUDA

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
);

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
);

#endif // WITH_CUDA

// ============================================================================
// CPU reference implementation
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_forward_cpu(const at::Tensor &x_t,        // (B, D)
                                 const at::Tensor &conv_state, // (B, D, K-1)
                                 const at::Tensor &ssm_state,  // (B, D)
                                 const at::Tensor &W_conv,     // (D, K)
                                 const at::Tensor &b_conv,     // (D)
                                 const at::Tensor &A,          // (D)
                                 const at::Tensor &Bp,         // (D)
                                 const at::Tensor &C,          // (D)
                                 const at::Tensor &D_skip      // (D)
) {
  auto B = x_t.size(0);
  auto D = x_t.size(1);
  auto K = W_conv.size(1);
  auto K_minus_1 = conv_state.size(2);

  at::Tensor y_t = at::zeros_like(x_t);
  at::Tensor new_conv_state = at::zeros_like(conv_state);
  at::Tensor new_ssm_state = at::zeros_like(ssm_state);

  // Access raw data
  auto x_a = x_t.accessor<float, 2>();
  auto conv_a = conv_state.accessor<float, 3>();
  auto ssm_a = ssm_state.accessor<float, 2>();
  auto W_a = W_conv.accessor<float, 2>();
  auto b_a = b_conv.accessor<float, 1>();
  auto A_a = A.accessor<float, 1>();
  auto Bp_a = Bp.accessor<float, 1>();
  auto C_a = C.accessor<float, 1>();
  auto D_a = D_skip.accessor<float, 1>();

  auto y_a = y_t.accessor<float, 2>();
  auto new_conv_a = new_conv_state.accessor<float, 3>();
  auto new_ssm_a = new_ssm_state.accessor<float, 2>();

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t d = 0; d < D; ++d) {
      float x_val = x_a[b][d];

      // 1) Convolution: u = sum_k W[d,k] * w_k + b[d]
      float u = 0.0f;
      for (int64_t k = 0; k < K_minus_1; ++k) {
        u += W_a[d][k] * conv_a[b][d][k];
      }
      u += W_a[d][K - 1] * x_val;
      u += b_a[d];

      // 2) SSM: z = A[d] * s_prev + B[d] * u
      float s_prev = ssm_a[b][d];
      float z = A_a[d] * s_prev + Bp_a[d] * u;

      // 3) Output: y = C[d] * z + D[d] * u
      float y = C_a[d] * z + D_a[d] * u;

      // 4) Update conv state: shift and append
      for (int64_t k = 0; k < K_minus_1 - 1; ++k) {
        new_conv_a[b][d][k] = conv_a[b][d][k + 1];
      }
      if (K_minus_1 > 0) {
        new_conv_a[b][d][K_minus_1 - 1] = x_val;
      }

      // 5) Update SSM state
      new_ssm_a[b][d] = z;

      // 6) Write output
      y_a[b][d] = y;
    }
  }

  return std::make_tuple(y_t, new_conv_state, new_ssm_state);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_backward_cpu(
    const at::Tensor &grad_y, const at::Tensor &x_t,
    const at::Tensor &conv_state, const at::Tensor &ssm_state,
    const at::Tensor &W_conv, const at::Tensor &b_conv, const at::Tensor &A,
    const at::Tensor &Bp, const at::Tensor &C, const at::Tensor &D_skip,
    const at::Tensor &y_t, const at::Tensor &new_conv_state,
    const at::Tensor &new_ssm_state) {
  auto B = x_t.size(0);
  auto D = x_t.size(1);
  auto K = W_conv.size(1);
  auto K_minus_1 = conv_state.size(2);

  at::Tensor grad_x = at::zeros_like(x_t);
  at::Tensor grad_conv_state = at::zeros_like(conv_state);
  at::Tensor grad_ssm_state = at::zeros_like(ssm_state);
  at::Tensor grad_W_conv = at::zeros_like(W_conv);
  at::Tensor grad_b_conv = at::zeros_like(b_conv);
  at::Tensor grad_A = at::zeros_like(A);
  at::Tensor grad_Bp = at::zeros_like(Bp);
  at::Tensor grad_C = at::zeros_like(C);
  at::Tensor grad_D_skip = at::zeros_like(D_skip);

  auto gy_a = grad_y.accessor<float, 2>();
  auto x_a = x_t.accessor<float, 2>();
  auto conv_a = conv_state.accessor<float, 3>();
  auto ssm_a = ssm_state.accessor<float, 2>();
  auto W_a = W_conv.accessor<float, 2>();
  auto b_a = b_conv.accessor<float, 1>();
  auto A_a = A.accessor<float, 1>();
  auto Bp_a = Bp.accessor<float, 1>();
  auto C_a = C.accessor<float, 1>();
  auto D_a = D_skip.accessor<float, 1>();

  auto gx_a = grad_x.accessor<float, 2>();
  auto gconv_a = grad_conv_state.accessor<float, 3>();
  auto gssm_a = grad_ssm_state.accessor<float, 2>();
  auto gW_a = grad_W_conv.accessor<float, 2>();
  auto gb_a = grad_b_conv.accessor<float, 1>();
  auto gA_a = grad_A.accessor<float, 1>();
  auto gBp_a = grad_Bp.accessor<float, 1>();
  auto gC_a = grad_C.accessor<float, 1>();
  auto gD_a = grad_D_skip.accessor<float, 1>();

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t d = 0; d < D; ++d) {
      float x_val = x_a[b][d];

      // Recompute forward
      float u = 0.0f;
      for (int64_t k = 0; k < K_minus_1; ++k) {
        u += W_a[d][k] * conv_a[b][d][k];
      }
      u += W_a[d][K - 1] * x_val;
      u += b_a[d];

      float s_prev = ssm_a[b][d];
      float z = A_a[d] * s_prev + Bp_a[d] * u;

      // Backward
      float gy = gy_a[b][d];

      // y = C*z + D*u
      float gu = gy * D_a[d];
      float gz = gy * C_a[d];

      gC_a[d] += gy * z;
      gD_a[d] += gy * u;

      // z = A*s_prev + B*u
      gA_a[d] += gz * s_prev;
      gBp_a[d] += gz * u;

      float gs_prev = gz * A_a[d];
      gu += gz * Bp_a[d];

      // u = sum W*w + b
      gb_a[d] += gu;

      for (int64_t k = 0; k < K_minus_1; ++k) {
        gW_a[d][k] += gu * conv_a[b][d][k];
        gconv_a[b][d][k] += gu * W_a[d][k];
      }
      gW_a[d][K - 1] += gu * x_val;
      gx_a[b][d] += gu * W_a[d][K - 1];

      gssm_a[b][d] += gs_prev;
    }
  }

  return std::make_tuple(grad_x, grad_conv_state, grad_ssm_state, grad_W_conv,
                         grad_b_conv, grad_A, grad_Bp, grad_C, grad_D_skip);
}

// ============================================================================
// Dispatch functions
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> recurrent_mamba_step_forward(
    const at::Tensor &x_t, const at::Tensor &conv_state,
    const at::Tensor &ssm_state, const at::Tensor &W_conv,
    const at::Tensor &b_conv, const at::Tensor &A, const at::Tensor &Bp,
    const at::Tensor &C, const at::Tensor &D_skip) {
  TORCH_CHECK(x_t.is_cuda() == conv_state.is_cuda(), "device mismatch");

  if (x_t.is_cuda()) {
#ifdef WITH_CUDA
    return recurrent_mamba_step_forward_cuda(x_t, conv_state, ssm_state, W_conv,
                                             b_conv, A, Bp, C, D_skip);
#else
    TORCH_CHECK(false, "recurrent_mamba_step compiled without CUDA support");
#endif
  } else {
    return recurrent_mamba_step_forward_cpu(x_t, conv_state, ssm_state, W_conv,
                                            b_conv, A, Bp, C, D_skip);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_backward(const at::Tensor &grad_y, const at::Tensor &x_t,
                              const at::Tensor &conv_state,
                              const at::Tensor &ssm_state,
                              const at::Tensor &W_conv,
                              const at::Tensor &b_conv, const at::Tensor &A,
                              const at::Tensor &Bp, const at::Tensor &C,
                              const at::Tensor &D_skip, const at::Tensor &y_t,
                              const at::Tensor &new_conv_state,
                              const at::Tensor &new_ssm_state) {
  if (grad_y.is_cuda()) {
#ifdef WITH_CUDA
    return recurrent_mamba_step_backward_cuda(
        grad_y, x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip,
        y_t, new_conv_state, new_ssm_state);
#else
    TORCH_CHECK(false, "recurrent_mamba_step compiled without CUDA support");
#endif
  } else {
    return recurrent_mamba_step_backward_cpu(
        grad_y, x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip,
        y_t, new_conv_state, new_ssm_state);
  }
}

} // namespace mamba

// ============================================================================
// PyBind11 module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Recurrent Mamba step kernel for true O(T) streaming";

  m.def("recurrent_mamba_step_forward", &mamba::recurrent_mamba_step_forward,
        "Recurrent Mamba step forward pass", py::arg("x_t"),
        py::arg("conv_state"), py::arg("ssm_state"), py::arg("W_conv"),
        py::arg("b_conv"), py::arg("A"), py::arg("Bp"), py::arg("C"),
        py::arg("D_skip"));

  m.def("recurrent_mamba_step_backward", &mamba::recurrent_mamba_step_backward,
        "Recurrent Mamba step backward pass", py::arg("grad_y"), py::arg("x_t"),
        py::arg("conv_state"), py::arg("ssm_state"), py::arg("W_conv"),
        py::arg("b_conv"), py::arg("A"), py::arg("Bp"), py::arg("C"),
        py::arg("D_skip"), py::arg("y_t"), py::arg("new_conv_state"),
        py::arg("new_ssm_state"));
}
