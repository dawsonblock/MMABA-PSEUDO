import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os

def patch_mamba_for_cpu():
    """
    Patches Mamba 2 to run on CPU/MPS by redirecting CUDA calls to pure PyTorch reference implementations.
    This allows the model to run on Mac and other non-CUDA systems, albeit slower.
    """
    if torch.cuda.is_available():
        return

    print("[*] Mamba Compat: CUDA not available. Patching Mamba 2 for CPU/MPS execution...")

    # Patch torch.mps.current_device if missing (common on some Mac setups)
    if hasattr(torch, "mps") and not hasattr(torch.mps, "current_device"):
        torch.mps.current_device = lambda: 0
        print("[+] Patched torch.mps.current_device")

    # Mock triton if not available
    import sys
    import types
    from unittest.mock import MagicMock

    if "triton" not in sys.modules:
        triton = types.ModuleType("triton")
        triton.__version__ = "2.1.0"
        triton.__spec__ = MagicMock()
        triton.jit = lambda x: x
        triton.autotune = lambda *args, **kwargs: lambda x: x
        triton.heuristics = lambda *args, **kwargs: lambda x: x
        triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
        triton.cdiv = lambda x, y: (x + y - 1) // y
        
        class MockConfig:
            def __init__(self, *args, **kwargs):
                self.num_warps = 4
                self.num_stages = 3
                self.kwargs = kwargs
        
        triton.Config = MockConfig
        
        triton.language = types.ModuleType("triton.language")
        triton.language.constexpr = int
        triton.language.lib = MagicMock() # Often used
        triton.language.dtype = int # Used by torch._dynamo
        
        # Mock triton.backends
        triton.backends = types.ModuleType("triton.backends")
        triton.backends.compiler = MagicMock()
        
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = triton.language
        sys.modules["triton.backends"] = triton.backends
        sys.modules["triton.backends.compiler"] = triton.backends.compiler
        
        # Mock triton.compiler
        triton.compiler = types.ModuleType("triton.compiler")
        triton.compiler.compiler = MagicMock()
        sys.modules["triton.compiler"] = triton.compiler
        sys.modules["triton.compiler.compiler"] = triton.compiler.compiler
        
        print("[+] Mocked triton for Mamba import")

    # Mock selective_scan_cuda if not available
    if "selective_scan_cuda" not in sys.modules:
        sys.modules["selective_scan_cuda"] = MagicMock()
        print("[+] Mocked selective_scan_cuda for Mamba import")

    # Mock transformers.generation.GenerateDecoderOnlyOutput if missing
    try:
        from transformers.generation import GenerateDecoderOnlyOutput
    except ImportError:
        import transformers.generation
        transformers.generation.GenerateDecoderOnlyOutput = MagicMock
        print("[+] Mocked transformers.generation.GenerateDecoderOnlyOutput")

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
        import mamba_ssm.ops.selective_scan_interface as ssi
        
        # Patch selective_scan_fn
        def selective_scan_fn_compat(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
            return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
        
        ssi.selective_scan_fn = selective_scan_fn_compat
        print("[+] Patched selective_scan_fn")

        # Patch mamba_chunk_scan_combined
        from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref
        import mamba_ssm.ops.triton.ssd_combined as ssd
        
        def mamba_chunk_scan_combined_compat(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=True, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False, cu_seqlens=None):
             # Note: ssd_chunk_scan_combined_ref signature might differ slightly, adapting as needed
             # Ref signature: (x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False)
             # It doesn't seem to support initial_states, seq_idx, return_final_states in the simple ref.
             # However, for basic training without state passing, this might be sufficient.
             # If state passing is required, we might need a more complex patch or just accept the limitation.
             
             if initial_states is not None:
                 warnings.warn("Mamba Compat: initial_states not supported in CPU reference implementation. Ignoring.")
             
             # Pad if necessary
             seqlen = x.shape[1]
             if seqlen % chunk_size != 0:
                 pad_len = chunk_size - (seqlen % chunk_size)
                 x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
                 dt = F.pad(dt, (0, 0, 0, pad_len))
                 B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
                 C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
                 if z is not None:
                     z = F.pad(z, (0, 0, 0, 0, 0, pad_len))
             else:
                 pad_len = 0

             out = ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus)
             
             # Unpad
             if pad_len > 0:
                 out = out[:, :seqlen]

             if return_final_states:
                 # Mock final states as zeros or handle properly if needed
                 batch, _, nheads, headdim = x.shape
                 dstate = B.shape[-1]
                 final_states = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=x.dtype)
                 return out, final_states
             
             return out

        ssd.mamba_chunk_scan_combined = mamba_chunk_scan_combined_compat
        print("[+] Patched mamba_chunk_scan_combined")
        
        # Patch Mamba2 init to force non-mem-efficient path (which uses the above patched functions)
        from mamba_ssm.modules.mamba2 import Mamba2
        import mamba_ssm.modules.mamba2 as mamba2_module
        
        mamba2_module.mamba_chunk_scan_combined = mamba_chunk_scan_combined_compat
        print("[+] Patched mamba_ssm.modules.mamba2.mamba_chunk_scan_combined")
        
        # Patch RMSNorm
        try:
            from mamba_ssm.ops.triton.layernorm_gated import RMSNorm
            import mamba_ssm.ops.triton.layernorm_gated as layernorm_gated
            
            class RMSNormCompat(nn.Module):
                def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None, norm_before_gate=True, group_size=None):
                    super().__init__()
                    self.eps = eps
                    self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
                    self.norm_before_gate = norm_before_gate

                def forward(self, x, z=None, group_size=None, norm_before_gate=True):
                    # x: (..., D)
                    # z: (..., D) if present
                    
                    # If z is present, we need to handle gating
                    # Mamba2 uses RMSNorm(x, z) -> (x * silu(z)) * weight / norm
                    # Or similar. Let's check usage.
                    # Mamba2 passes z to RMSNorm.
                    
                    # Standard RMSNorm
                    # out = x * weight * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
                    
                    if z is not None:
                        if not norm_before_gate:
                            x = x * F.silu(z)
                    
                    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
                    
                    if z is not None and norm_before_gate:
                        out = out * F.silu(z)
                        
                    return out

            layernorm_gated.RMSNorm = RMSNormCompat
            mamba2_module.RMSNorm = RMSNormCompat
            mamba2_module.RMSNormGated = RMSNormCompat
            print("[+] Patched mamba_ssm.ops.triton.layernorm_gated.RMSNorm and mamba_ssm.modules.mamba2.RMSNormGated")
            
        except ImportError:
            print("[-] Could not patch RMSNorm (import failed)")
        except Exception as e:
            print(f"[-] Could not patch RMSNorm: {e}")

        original_init = Mamba2.__init__
        
        def mamba2_init_compat(self, *args, **kwargs):
            # Force use_mem_eff_path to False
            if "use_mem_eff_path" in kwargs:
                kwargs["use_mem_eff_path"] = False
            
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Force use_mem_eff_path to False (just in case)
            self.use_mem_eff_path = False
            
            # Replace self.norm with RMSNormCompat if it exists
            if hasattr(self, "norm") and isinstance(self.norm, nn.Module):
                # Check if it's the Triton RMSNorm (or just replace it blindly if we are sure)
                # We need to preserve weights if possible, but Triton RMSNorm might have different structure?
                # Triton RMSNorm has 'weight' parameter.
                # RMSNormCompat has 'weight' parameter.
                
                old_norm = self.norm
                new_norm = RMSNormCompat(old_norm.weight.shape[0], eps=old_norm.eps, device=old_norm.weight.device, dtype=old_norm.weight.dtype)
                
                # Copy weights (if they are initialized)
                with torch.no_grad():
                    new_norm.weight.copy_(old_norm.weight)
                
                self.norm = new_norm
                # print("[+] Replaced self.norm with RMSNormCompat instance")

        Mamba2.__init__ = mamba2_init_compat
        print("[+] Patched Mamba2.__init__")

    except ImportError as e:
        import traceback
        traceback.print_exc()
        print(f"[-] Mamba Compat Error: Could not import Mamba modules to patch. {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[-] Mamba Compat Error: An error occurred during patching. {e}")
