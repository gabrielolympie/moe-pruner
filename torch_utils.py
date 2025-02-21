import torch
from configs import PathConfig
import gc

def save_intermediate_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int, state: torch.Tensor
):
    """Save intermediate layer output to a file in FP8 format"""
    fp8_tensor = state.to(torch.bfloat16)
    torch.save(fp8_tensor, path_config.get_intermediate_path(layer_idx, batch_idx))


def save_midlayer_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int, state: torch.Tensor
):
    """Save intermediate layer output to a file in FP8 format"""
    fp8_tensor = state.to(torch.bfloat16)
    torch.save(fp8_tensor, path_config.get_exp_path(layer_idx, batch_idx))


def load_intermediate_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int
) -> torch.Tensor:
    """Load intermediate layer output from a file and upcast from FP8"""
    fp8_tensor = torch.load(path_config.get_intermediate_path(layer_idx, batch_idx))
    return fp8_tensor.to(torch.bfloat16)

def load_midlayer_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int
) -> torch.Tensor:
    """Load intermediate layer output from a file and upcast from FP8"""
    fp8_tensor = torch.load(path_config.get_exp_path(layer_idx, batch_idx))
    return fp8_tensor.to(torch.bfloat16)

def destruct_module_optimized(module: torch.nn.Module) -> torch.nn.Module:
    """Efficiently destroy module and clear memory."""
    module.to_empty(device="meta")
    gc.collect()
    torch.cuda.empty_cache()
    
def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
def count_parameters(model):
    frozen_params = 0
    non_frozen_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            non_frozen_params += param.numel()
        else:
            frozen_params += param.numel()

    total_params = frozen_params + non_frozen_params

    print(f"{'Parameter Type':<20} {'Count':<10}")
    print(f"{'='*20} {'='*10}")
    print(f"{'Frozen Parameters':<20} {frozen_params:<10,}")
    print(f"{'Non-Frozen Parameters':<20} {non_frozen_params:<10,}")
    print(f"{'Total Parameters':<20} {total_params:<10,}")