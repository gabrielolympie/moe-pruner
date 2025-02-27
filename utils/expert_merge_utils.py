import torch
from copy import deepcopy
from tqdm.auto import tqdm
import os
from awq.modules.linear.gemm import WQLinear_GEMM
from awq.modules.triton.gemm import awq_gemm_triton, awq_dequantize_triton

from utils.torch_utils import rsetattr, load_quant
from utils.ademamix import AdEMAMix

def calibrated_dequant(module, layer_norm, path_config, layer_idx):
    module_dequant, params=dequantize_GEMM(module, False)
    
    # optimizer=AdEMAMix(
    #     module_dequant.parameters(),
    #     lr=5e-7
    # )
    
    # criterion = torch.nn.MSELoss()
    
    # total_batches = len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}")))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches)
    # progress_bar = tqdm(range(total_batches), desc="Calibrated Dequant", leave=False)
    
    # for batch_idx in progress_bar:
    #     hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}"))
    #     hidden_states = layer_norm(hidden_states)
        
    #     module_output=module(hidden_states)
    #     module_dequant_output=module_dequant(hidden_states)
        
    #     loss = criterion(module_dequant_output, module_output)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
        
    #     progress_bar.set_postfix(loss=loss.item())
    return module_dequant

def dequantize_GEMM(model, destruct=True, dtype=torch.bfloat16):
    new_model=deepcopy(model)
    params=[]
    for name, module in new_model.named_modules():
        if isinstance(module, WQLinear_GEMM):
            linear, quant_params = dequantize_WQLinear_GEMM(module, destruct, dtype)
            params.append(quant_params)
            rsetattr(new_model, name, linear)
    
    if destruct:
        model.to_empty(device='meta')
        del model
    new_model=new_model.to(dtype=dtype)
    return new_model, params
        
def dequantize_WQLinear_GEMM(wq_linear, destruct=True, dtype=torch.bfloat16):
    
    quant_params={
        'w_bit': wq_linear.w_bit,
        'group_size': wq_linear.group_size,
        'in_features': wq_linear.in_features,
        'out_features': wq_linear.out_features,
        'bias': wq_linear.bias is not None,
        'dev':wq_linear.qweight.device,
        'zero_point':wq_linear.qzeros is not None,
    }
    
    linear=torch.nn.Linear(wq_linear.in_features, wq_linear.out_features, bias=wq_linear.bias is not None, device=wq_linear.qweight.device, dtype=dtype)
    
    linear.weight=torch.nn.Parameter(
        awq_dequantize_triton(
            wq_linear.qweight,
            wq_linear.scales,
            wq_linear.qzeros,
        ).T,
        requires_grad=True
    )
    
    if wq_linear.bias is not None:
        linear.bias = torch.nn.Parameter(wq_linear.bias, requires_grad=True)
        
    if destruct:
        wq_linear.to_empty(device="meta")
        del wq_linear
        
    return linear, quant_params

from typing import List, Union

def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a vector to unit length with a small epsilon to avoid division by zero.
    
    Args:
        v (torch.Tensor): Input vector/tensor
        eps (float): Small value to avoid division by zero
        
    Returns:
        torch.Tensor: Normalized vector
    """
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / torch.clamp(norm, min=eps)

def slerp(t: Union[float, torch.Tensor], v0: torch.Tensor, v1: torch.Tensor, 
          DOT_THRESHOLD: float = 0.9995, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two vectors in PyTorch.
    
    Args:
        t (float/torch.Tensor): Float value between 0.0 and 1.0
        v0 (torch.Tensor): Starting vector
        v1 (torch.Tensor): Final vector
        DOT_THRESHOLD (float): Threshold for considering vectors as colinear
        eps (float): Small value to avoid division by zero
        
    Returns:
        torch.Tensor: Interpolated vector between v0 and v1
    """
    # Make a copy of the original vectors
    v0_copy = v0.clone()
    v1_copy = v1.clone()
    
    # Normalize the vectors
    v0_norm = normalize(v0, eps)
    v1_norm = normalize(v1, eps)
    
    # Compute dot product
    dot = torch.sum(v0_norm * v1_norm)
    
    # If vectors are nearly parallel, use linear interpolation
    if torch.abs(dot) > DOT_THRESHOLD:
        return (1 - t) * v0_copy + t * v1_copy
    
    # Calculate initial angle between v0 and v1
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    # Compute the coefficients
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    # Compute the result
    return s0 * v0_copy + s1 * v1_copy

def nslerp_radial(
    weights: List[float],
    tensors: List[torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Alternative n-way slerp implementation using a radial approach.
    This approach uses a weighted average vector as a reference point.

    Args:
        tensors (List[torch.Tensor]): List of tensors to interpolate
        weights (List[float]): List of weights for each tensor (should sum to 1.0)
        DOT_THRESHOLD (float): Threshold for considering two vectors as colinear
        eps (float): Small value to avoid division by zero

    Returns:
        Interpolated tensor
    """
    if len(tensors) != len(weights):
        raise ValueError("Number of tensors must match number of weights")

    if abs(sum(weights) - 1.0) > eps:
        # Normalize weights to sum to 1.0
        total = sum(weights)
        weights = [w / total for w in weights]

    if len(tensors) == 1:
        return tensors[0]

    if len(tensors) == 2:
        # Use standard slerp for two tensors
        return slerp(weights[1], tensors[0], tensors[1], DOT_THRESHOLD, eps)

    # First calculate a weighted average as a center point
    center = torch.zeros_like(tensors[0])
    for i, tensor in enumerate(tensors):
        center += weights[i] * tensor

    # For each tensor, calculate slerp path from center to tensor
    # and move along that path according to its weight
    result = torch.zeros_like(center)
    for i, tensor in enumerate(tensors):
        # Skip tensors with zero weight
        if weights[i] < eps:
            continue

        # Interpolate from center to this tensor, with distance proportional to weight
        interp = slerp(weights[i], center, tensor, DOT_THRESHOLD, eps)
        result += weights[i] * interp

    # Normalize the result to maintain magnitude
    result_norm = torch.sqrt(torch.sum(result * result)) + eps
    center_norm = torch.sqrt(torch.sum(center * center)) + eps
    result = result * (center_norm / result_norm)

    return result

def uniform_multi_slerp(
    vs: List[torch.Tensor], 
    DOT_THRESHOLD: float = 0.9995, 
    eps: float = 1e-8,
) -> torch.Tensor:
    ts=[1.0/len(vs)]*len(vs)
    return nslerp_radial(
        weights=ts,
        tensors=vs,
        DOT_THRESHOLD=DOT_THRESHOLD,
        eps=eps,
    )

def calibrated_merge_experts(expert_list, layer_norm, distilled_mlp, path_config, layer_idx):
    experts_to_merge=[]
    experts_to_merge_quant=[]
    for expert_index in tqdm(expert_list):
        module = distilled_mlp.experts[expert_index].to('cuda:0')
        module_dequant = calibrated_dequant(module, layer_norm, path_config, layer_idx)
        experts_to_merge.append(module_dequant)
        experts_to_merge_quant.append(module)
    
    new_state_dict=experts_to_merge[0].state_dict()
    for k in experts_to_merge[0].state_dict().keys():
        new_state_dict[k] = uniform_multi_slerp([expert.state_dict()[k] for expert in experts_to_merge])
    
    merged_expert = deepcopy(experts_to_merge[0])
    merged_expert.load_state_dict(new_state_dict)
    
    new_state_dict=experts_to_merge[0].state_dict()
    for k in experts_to_merge[0].state_dict().keys():
        new_state_dict[k] = uniform_multi_slerp([expert.state_dict()[k] for expert in experts_to_merge])
    
    merged_expert = deepcopy(experts_to_merge[0])
    merged_expert.load_state_dict(new_state_dict)
    
    del experts_to_merge
    
    optimizer=AdEMAMix(
        merged_expert.parameters(),
        lr=5e-4
    )
    
    criterion = torch.nn.MSELoss()
    
    total_batches = len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}")))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=5e-5)
    progress_bar = tqdm(range(total_batches), desc="Calibrating merged expert")
    
    for batch_idx in progress_bar:
        hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}"))
        hidden_states=layer_norm(hidden_states)
    
        merged_expert_output=merged_expert(hidden_states)
        
        experts_to_merge_output=[]
        for expert in experts_to_merge_quant:
            experts_to_merge_output.append(expert(hidden_states))
            
        experts_to_merge_output=torch.mean(torch.stack(experts_to_merge_output), axis=0)
        
        loss = criterion(merged_expert_output, experts_to_merge_output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix(loss=loss.item())
    return merged_expert