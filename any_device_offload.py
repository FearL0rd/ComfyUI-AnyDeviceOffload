import torch
import torch.nn.functional as F
import types
import comfy.model_management
import gc
import sys

# --- GLOBAL XFORMERS PATCH (CPU FIX) ---
try:
    import xformers.ops
    # Save the original function so we don't break GPU usage
    if not hasattr(xformers.ops, "original_memory_efficient_attention"):
        xformers.ops.original_memory_efficient_attention = xformers.ops.memory_efficient_attention

    def traffic_controlled_attention(query, key, value, attn_bias=None, p=0.0, scale=None):
        # 1. If on GPU, use the original xFormers (Fast)
        if query.device.type != "cpu":
            return xformers.ops.original_memory_efficient_attention(query, key, value, attn_bias, p, scale)
        
        # 2. CPU Fallback (Standard PyTorch)
        
        # [CRITICAL FIX] Force all inputs to Float32. 
        # Even if the model inputs were fixed, internal layers might still produce BF16.
        if query.dtype != torch.float32:
            query = query.to(torch.float32)
        if key.dtype != torch.float32:
            key = key.to(torch.float32)
        if value.dtype != torch.float32:
            value = value.to(torch.float32)

        # PyTorch expects (Batch, Heads, SeqLen, Dim)
        # xFormers sends (Batch, SeqLen, Heads, Dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        
        is_causal = False
        attn_mask = None
        if attn_bias is not None:
            if isinstance(attn_bias, xformers.ops.LowerTriangularMask):
                is_causal = True
                attn_mask = None
            elif isinstance(attn_bias, torch.Tensor):
                attn_mask = attn_bias
                # Ensure mask matches dtype if it's a tensor
                if attn_mask.dtype != torch.float32 and attn_mask.dtype != torch.bool:
                     attn_mask = attn_mask.to(torch.float32)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p, is_causal=is_causal, scale=scale)
        
        # Transpose back to xFormers layout
        return out.transpose(1, 2)

    # Overwrite the library function with our wrapper
    xformers.ops.memory_efficient_attention = traffic_controlled_attention
except ImportError:
    pass
# -----------------------------

def get_available_devices():
    """
    Returns a list of devices, sorted by priority.
    The first item in the list becomes the default selection in ComfyUI.
    """
    devices = []
    
    # 1. Priority: NVIDIA/AMD GPUs
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            devices.append(f"cuda:{i}")

    # 2. Priority: Mac Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    # 3. Priority: Intel XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
        
    # 4. Fallback: CPU
    devices.append("cpu")
    
    return devices

class AnyDeviceOffload:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # The list is now sorted so the best GPU is index 0
        device_list = get_available_devices()
        return {
            "required": {
                "target_device": (device_list, ),
                "vae_mode": (["Vae Patched", "Original"], ),
                "keep_in_memory": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "offload_process"
    CATEGORY = "utils/hardware"

    def offload_process(self, target_device, vae_mode, keep_in_memory, model=None, clip=None, vae=None):
        device_str = target_device.split(" ")[0]
        try:
            device = torch.device(device_str)
        except:
            device = torch.device("cpu")

        # CPU Detection
        is_cpu = device.type == 'cpu'
        offload_target = device if keep_in_memory else torch.device("cpu")
        
        print(f"[Offload] Target: {device} | Keep: {keep_in_memory}")

        # --- 1. Handle MODEL (UNet/Transformer) ---
        if model is not None:
            target_model = None
            if hasattr(model, "model"):
                target_model = model.model
            
            if target_model:
                # A. Weight Conversion
                if is_cpu:
                    target_model.to(dtype=torch.float32)

                if keep_in_memory:
                    target_model.to(device)

                # B. UNIVERSAL INPUT CASTER (Fixes 'missing argument' & 'dtype' errors)
                if is_cpu and not hasattr(target_model, "is_cpu_patched"):
                    print(" -> [CPU] Applying Universal Input Caster...")
                    
                    if hasattr(target_model, "diffusion_model"):
                        target_model.diffusion_model.original_forward_cpu = target_model.diffusion_model.forward

                        def universal_cpu_forward(self, *args, **kwargs):
                            # 1. Cast Positional Arguments
                            new_args = []
                            for arg in args:
                                if isinstance(arg, torch.Tensor) and arg.dtype != torch.float32:
                                    new_args.append(arg.to(torch.float32))
                                else:
                                    new_args.append(arg)
                            
                            # 2. Cast Keyword Arguments
                            new_kwargs = {}
                            for k, v in kwargs.items():
                                if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                                    new_kwargs[k] = v.to(torch.float32)
                                else:
                                    new_kwargs[k] = v

                            # 3. Pass exact structure to original function
                            return self.original_forward_cpu(*new_args, **new_kwargs)

                        target_model.diffusion_model.forward = types.MethodType(universal_cpu_forward, target_model.diffusion_model)
                        target_model.is_cpu_patched = True

            model.load_device = device
            model.offload_device = offload_target
            if keep_in_memory:
                model.current_device = device

        # --- 2. Handle CLIP ---
        if clip is not None:
            if hasattr(clip, "cond_stage_model"):
                target_clip = clip.cond_stage_model
                if is_cpu:
                    target_clip.to(dtype=torch.float32)
                if keep_in_memory:
                    target_clip.to(device)
            
            if hasattr(clip, "patcher"):
                clip.patcher.load_device = device
                clip.patcher.offload_device = offload_target
                if keep_in_memory:
                    clip.patcher.current_device = device

        # --- 3. Handle VAE ---
        if vae is not None:
            target_vae = None
            if hasattr(vae, "first_stage_model"):
                target_vae = vae.first_stage_model
            elif hasattr(vae, "model"):
                target_vae = vae.model

            if target_vae:
                # Setup Dynamic State
                if not hasattr(target_vae, "offload_node_state"):
                    target_vae.offload_node_state = {}
                target_vae.offload_node_state['keep'] = keep_in_memory
                target_vae.offload_node_state['device'] = device
                target_vae.offload_node_state['is_cpu'] = is_cpu
                target_vae.offload_node_state['model_ref'] = model
                target_vae.offload_node_state['clip_ref'] = clip

                # Precision
                if is_cpu or vae_mode == "Vae Patched":
                    target_vae.to(dtype=torch.float32)

                # Move
                if keep_in_memory:
                    target_vae.to(device)
                    target_vae.eval()

                # VAE Wrapper
                if not hasattr(target_vae, "is_offload_patched"):
                    target_vae.original_decode_patched = target_vae.decode

                    def dynamic_kill_switch_wrapper(self, z, *args, **kwargs):
                        state = getattr(self, "offload_node_state", {})
                        should_keep = state.get('keep', True)
                        running_on_cpu = state.get('is_cpu', False)
                        
                        try:
                            # 1. Type Safety
                            if running_on_cpu:
                                if z.dtype != torch.float32:
                                    z = z.to(torch.float32)
                            else:
                                dtype_target = next(self.parameters()).dtype
                                if z.dtype != dtype_target:
                                    z = z.to(dtype_target)
                            
                            # 2. Device Safety
                            current_weight_device = next(self.parameters()).device
                            if z.device != current_weight_device:
                                z = z.to(current_weight_device)
                            
                            return self.original_decode_patched(z, *args, **kwargs)

                        finally:
                            # 3. KILL SWITCH
                            if not should_keep:
                                self.to("cpu")
                                linked_model = state.get('model_ref')
                                linked_clip = state.get('clip_ref')

                                if linked_model and hasattr(linked_model, "model"):
                                    linked_model.model.to("cpu")
                                    linked_model.current_device = torch.device("cpu")
                                
                                if linked_clip and hasattr(linked_clip, "cond_stage_model"):
                                    linked_clip.cond_stage_model.to("cpu")
                                    if hasattr(linked_clip, "patcher"):
                                        linked_clip.patcher.current_device = torch.device("cpu")

                                gc.collect()
                                torch.cuda.empty_cache()
                                comfy.model_management.cleanup_models()
                                comfy.model_management.soft_empty_cache()

                    target_vae.decode = types.MethodType(dynamic_kill_switch_wrapper, target_vae)
                    target_vae.is_offload_patched = True

            if hasattr(vae, "device") and keep_in_memory:
                vae.device = device
            
            if hasattr(vae, "patcher"):
                vae.patcher.load_device = device
                vae.patcher.offload_device = offload_target
                if keep_in_memory:
                    vae.patcher.current_device = device

        return (model, clip, vae)

NODE_CLASS_MAPPINGS = {
    "AnyDeviceOffload": AnyDeviceOffload
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyDeviceOffload": "Offload Anything (GPU/CPU)"
}

