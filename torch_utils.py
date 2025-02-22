import torch
from configs import PathConfig
from fp8_linear import highest_power_of_2_divisor, act_quant, weight_dequant
import gc

def save_quant(x, base_path):
    x = x.view(-1, x.shape[-1])
    weight, weight_scale_inv = act_quant(x, 128)
    torch.save(weight, base_path.replace('.pt', 'weight.pt'))
    torch.save(weight_scale_inv, base_path.replace('pt', 'weight_scale_inv.pt'))
    
def load_quant(base_path, batch_size):
    weight = torch.load(base_path.replace('.pt', 'weight.pt'))
    weight_scale_inv = torch.load(base_path.replace('pt', 'weight_scale_inv.pt'))
    x = weight_dequant(weight, weight_scale_inv)
    x = x.view(batch_size, x.shape[0] // batch_size, x.shape[-1])
    return x
    
def save_intermediate_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int, state: torch.Tensor
):
    """Save intermediate layer output to a file in FP8 format"""
    save_quant(state, path_config.get_intermediate_path(layer_idx, batch_idx))


def save_midlayer_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int, state: torch.Tensor
):
    """Save intermediate layer output to a file in FP8 format"""
    save_quant(state, path_config.get_midlayer_path(layer_idx, batch_idx))

def load_intermediate_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int, batch_size
) -> torch.Tensor:
    """Load intermediate layer output from a file and upcast from FP8"""
    return load_quant(path_config.get_intermediate_path(layer_idx, batch_idx), batch_size)

def load_midlayer_state(
    path_config: PathConfig, layer_idx: int, batch_idx: int, batch_size
) -> torch.Tensor:
    """Load intermediate layer output from a file and upcast from FP8"""
    return load_quant(path_config.get_midlayer_path(layer_idx, batch_idx), batch_size)

def destruct_module_optimized(module: torch.nn.Module) -> torch.nn.Module:
    """Efficiently destroy module and clear memory."""
    module.to_empty(device="meta")
    gc.collect()
    torch.cuda.empty_cache()
    
def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
        
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