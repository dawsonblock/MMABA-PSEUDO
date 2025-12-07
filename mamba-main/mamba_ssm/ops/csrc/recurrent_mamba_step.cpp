// mamba_ssm/ops/csrc/recurrent_mamba_step.cpp
// 
// C++ front for CPU/CUDA dispatch of recurrent Mamba step kernel.
// This implements a true O(T) streaming Mamba cell.
//
// The kernel dev must implement:
//   - recurrent_mamba_step_forward_cuda()
//   - recurrent_mamba_step_backward_cuda()
// 
// CPU versions are optional (can call Python reference or throw).

#include <torch/extension.h>
#include <tuple>

namespace mamba {

// ============================================================================
// Forward declarations - CUDA kernels (implemented in .cu file)
// ============================================================================

#ifdef WITH_CUDA

std::tuple<at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_forward_cuda(
    const at::Tensor& x_t,          // (B, D)
    const at::Tensor& conv_state,   // (B, D, K-1)
    const at::Tensor& ssm_state,    // (B, N_state, D_inner)
    const at::Tensor& conv_weight,  // (D, K)
    const at::Tensor& A,            // SSM A matrix
    const at::Tensor& B,            // SSM B matrix
    const at::Tensor& C,            // SSM C matrix
    const at::Tensor& D_param       // SSM D matrix
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_backward_cuda(
    const at::Tensor& grad_y,        // (B, D)
    const at::Tensor& x_t,           // (B, D)
    const at::Tensor& conv_state,    // (B, D, K-1)
    const at::Tensor& ssm_state,     // (B, N_state, D_inner)
    const at::Tensor& conv_weight,   // (D, K)
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D_param,
    const at::Tensor& y_t,
    const at::Tensor& new_conv_state,
    const at::Tensor& new_ssm_state
);

#endif  // WITH_CUDA

// ============================================================================
// CPU reference implementation (fallback / testing)
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_forward_cpu(
    const at::Tensor& x_t,          // (B, D)
    const at::Tensor& conv_state,   // (B, D, K-1)
    const at::Tensor& ssm_state,    // (B, N_state, D_inner)
    const at::Tensor& conv_weight,  // (D, K)
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D_param
) {
    // CPU reference implementation
    // This is a simplified version - kernel dev should match full Mamba2 math
    
    auto B_size = x_t.size(0);
    auto D = x_t.size(1);
    auto K_minus_1 = conv_state.size(2);
    auto K = K_minus_1 + 1;
    
    // 1. Update conv state: shift left and append x_t
    // new_conv_state[:, :, :-1] = conv_state[:, :, 1:]
    // new_conv_state[:, :, -1] = x_t
    at::Tensor new_conv_state = at::zeros_like(conv_state);
    if (K_minus_1 > 0) {
        new_conv_state.slice(2, 0, K_minus_1 - 1).copy_(conv_state.slice(2, 1, K_minus_1));
    }
    // For the last position, we need x_t reshaped properly
    // x_t: (B, D) -> (B, D, 1)
    new_conv_state.select(2, K_minus_1 - 1).copy_(x_t);
    
    // 2. Compute convolution output for this step
    // Build full window: (B, D, K) = concat(conv_state, x_t.unsqueeze(-1))
    auto x_t_3d = x_t.unsqueeze(-1);  // (B, D, 1)
    auto conv_window = at::cat({conv_state, x_t_3d}, /*dim=*/2);  // (B, D, K)
    
    // Depthwise conv: sum over K dimension with weights
    // conv_out = (conv_window * conv_weight).sum(dim=-1)
    auto conv_out = (conv_window * conv_weight.unsqueeze(0)).sum(/*dim=*/2);  // (B, D)
    
    // 3. SSM recurrence (simplified)
    // z_t = A * z_{t-1} + B * u_t
    // y_t = C * z_t + D * u_t
    auto u_t = conv_out;  // Use conv output as SSM input
    
    // For now, simple linear SSM (kernel dev implements full selective scan)
    // A: (N_state, D_inner), ssm_state: (B, N_state, D_inner)
    // This is a placeholder - real impl needs proper einsum/matmul
    at::Tensor new_ssm_state = ssm_state * 0.9 + u_t.unsqueeze(1) * 0.1;  // placeholder decay
    
    // Output
    at::Tensor y_t = u_t;  // placeholder: just pass through conv output
    
    return std::make_tuple(y_t, new_conv_state, new_ssm_state);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_backward_cpu(
    const at::Tensor& grad_y,
    const at::Tensor& x_t,
    const at::Tensor& conv_state,
    const at::Tensor& ssm_state,
    const at::Tensor& conv_weight,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D_param,
    const at::Tensor& y_t,
    const at::Tensor& new_conv_state,
    const at::Tensor& new_ssm_state
) {
    // CPU backward - placeholder implementation
    // Kernel dev must match gradients from Mamba2
    
    at::Tensor grad_x = at::zeros_like(x_t);
    at::Tensor grad_conv_state = at::zeros_like(conv_state);
    at::Tensor grad_ssm_state = at::zeros_like(ssm_state);
    at::Tensor grad_conv_weight = at::zeros_like(conv_weight);
    at::Tensor grad_A = at::zeros_like(A);
    at::Tensor grad_B = at::zeros_like(B);
    at::Tensor grad_C = at::zeros_like(C);
    
    // Placeholder: identity gradient for x_t
    grad_x.copy_(grad_y);
    
    return std::make_tuple(
        grad_x, grad_conv_state, grad_ssm_state,
        grad_conv_weight, grad_A, grad_B, grad_C
    );
}

// ============================================================================
// Dispatch functions
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_forward(
    const at::Tensor& x_t,
    const at::Tensor& conv_state,
    const at::Tensor& ssm_state,
    const at::Tensor& conv_weight,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D_param
) {
    // Device checks
    TORCH_CHECK(x_t.is_cuda() == conv_state.is_cuda(), "device mismatch: x_t vs conv_state");
    TORCH_CHECK(x_t.is_cuda() == ssm_state.is_cuda(),  "device mismatch: x_t vs ssm_state");

    if (x_t.is_cuda()) {
#ifdef WITH_CUDA
        return recurrent_mamba_step_forward_cuda(
            x_t, conv_state, ssm_state, conv_weight, A, B, C, D_param
        );
#else
        TORCH_CHECK(false, "recurrent_mamba_step compiled without CUDA support");
#endif
    } else {
        return recurrent_mamba_step_forward_cpu(
            x_t, conv_state, ssm_state, conv_weight, A, B, C, D_param
        );
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recurrent_mamba_step_backward(
    const at::Tensor& grad_y,
    const at::Tensor& x_t,
    const at::Tensor& conv_state,
    const at::Tensor& ssm_state,
    const at::Tensor& conv_weight,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D_param,
    const at::Tensor& y_t,
    const at::Tensor& new_conv_state,
    const at::Tensor& new_ssm_state
) {
    if (grad_y.is_cuda()) {
#ifdef WITH_CUDA
        return recurrent_mamba_step_backward_cuda(
            grad_y, x_t, conv_state, ssm_state, conv_weight, A, B, C, D_param,
            y_t, new_conv_state, new_ssm_state
        );
#else
        TORCH_CHECK(false, "recurrent_mamba_step compiled without CUDA support");
#endif
    } else {
        return recurrent_mamba_step_backward_cpu(
            grad_y, x_t, conv_state, ssm_state, conv_weight, A, B, C, D_param,
            y_t, new_conv_state, new_ssm_state
        );
    }
}

}  // namespace mamba

// ============================================================================
// PyBind11 module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Recurrent Mamba step kernel for true O(T) streaming";
    
    m.def(
        "recurrent_mamba_step_forward",
        &mamba::recurrent_mamba_step_forward,
        "Recurrent Mamba step forward pass",
        py::arg("x_t"),
        py::arg("conv_state"),
        py::arg("ssm_state"),
        py::arg("conv_weight"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("D_param")
    );
    
    m.def(
        "recurrent_mamba_step_backward",
        &mamba::recurrent_mamba_step_backward,
        "Recurrent Mamba step backward pass",
        py::arg("grad_y"),
        py::arg("x_t"),
        py::arg("conv_state"),
        py::arg("ssm_state"),
        py::arg("conv_weight"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("D_param"),
        py::arg("y_t"),
        py::arg("new_conv_state"),
        py::arg("new_ssm_state")
    );
}
