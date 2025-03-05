from mergekit.merge_methods.sce import sce_weight, sce_mask
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)
import torch
from typing import List, Union

def sce_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    int8_mask: bool = False,
    select_topk: float = 1.0,
) -> torch.Tensor:
    if not tensors:
        return base_tensor
    mask_dtype = torch.int8 if int8_mask else base_tensor.dtype
    task_vectors = torch.stack([t - base_tensor for t in tensors], dim=0)

    if select_topk < 1:
        mask = sce_mask(task_vectors, select_topk, mask_dtype)
        
        if len(mask.shape) != len(task_vectors.shape): ## Correction should happend only when relevant, else it will bug the whole stuff
            mask = mask.unsqueeze(0)
            
        task_vectors = task_vectors * mask

    erase_mask = sign_consensus_mask(task_vectors, method="sum", mask_dtype=mask_dtype)

    tv_weights = sce_weight(task_vectors)
    
    while tv_weights.dim() < task_vectors.dim():
        tv_weights = tv_weights.unsqueeze(-1)

    erased_weights = tv_weights * erase_mask
    merged_tv = (task_vectors * erased_weights).sum(dim=0)
    
    final_tv = merged_tv / torch.sum(erased_weights, dim=0).clamp(min=1e-6)
    return base_tensor + final_tv