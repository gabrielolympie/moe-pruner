from sklearn.cluster import SpectralClustering

import torch.nn.functional as F
from typing import List, Union
from numba import njit, prange
from tqdm.auto import tqdm
from peft.tuners import lora
from copy import deepcopy
import _pickle as pickle
import numpy as np
import numba as nb

import torch
import time

from awq.modules.linear.gemm import WQLinear_GEMM
from awq.modules.triton.gemm import awq_gemm_triton, awq_dequantize_triton
from mergekit.merge_methods.sce import sce_weight, sce_mask
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)

from mergekit.merge_methods.multislerp import multislerp
from mergekit.merge_methods.slerp import slerp

from utils.torch_utils import load_quant, rsetattr, WarmupCosineAnnealingLR
from utils.ademamix import AdEMAMix
import os

## ************************ UTILS ************************ ##
@nb.njit(parallel=True)
def cooccurrence_matrix(indices_list, n_unique):
    matrix = np.zeros((n_unique, n_unique), dtype=np.int64)
    
    for i in range(len(indices_list)):
        indices = indices_list[i]  
        for idx1 in indices:
            for idx2 in indices:
                matrix[idx1, idx2] += 1
    return matrix

@njit
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit
def sigmoid_inv(x):
    # Inverse of sigmoid: log(x / (1-x))
    return np.log(x / (1.0 - x))

@njit(parallel=True)
def compute_new_weights(
    top_k_output,
    top_k_weight,
    expert_indices,
    expert_boundaries, 
    num_samples,
    num_target_experts,
    scoring_function_id
):
    new_weights = np.zeros((num_samples, num_target_experts))
    for l in prange(num_samples):
        indices = top_k_output[l]
        weights = top_k_weight[l]
        for i in range(num_target_experts):
            weight = 0.0
            start_idx = expert_boundaries[i]
            end_idx = expert_boundaries[i+1]
            for j_idx in range(start_idx, end_idx):
                j = expert_indices[j_idx]
                for k in range(len(indices)):
                    if indices[k] == j:
                        w = weights[k]
                        if scoring_function_id == 0:
                            weight += w
                        else:
                            weight += sigmoid_inv(w)
                        break
            if scoring_function_id == 1:
                weight = sigmoid(weight)
            
            new_weights[l, i] = weight
            
    return new_weights

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

def build_affinity_matrix(hidden_states):
    affinity_matrix = hidden_states.view(hidden_states.shape[0], -1, hidden_states.shape[-1])
    affinity_matrix = affinity_matrix / torch.norm(affinity_matrix, dim=-1, keepdim=True)
    affinity_matrix = affinity_matrix.transpose(0,1)
    affinity_matrix = torch.matmul(affinity_matrix, torch.transpose(affinity_matrix, 1,2))
    affinity_matrix = torch.mean(affinity_matrix, axis=0)
    affinity_matrix = affinity_matrix.to(torch.float32).detach().cpu().numpy()
    affinity_matrix = (affinity_matrix - affinity_matrix.min()) / (affinity_matrix.max() - affinity_matrix.min())
    return affinity_matrix

def pair_items_by_affinity(affinity_matrix):
    affinities = affinity_matrix.copy()
    np.fill_diagonal(affinities, -np.inf)
    pairs = []
    remaining_items = set(range(affinities.shape[0]))
    while len(remaining_items) > 1:
        if len(remaining_items) == 2:
            i, j = sorted(list(remaining_items))
        else:
            mask = np.ones_like(affinities, dtype=bool)
            for i in range(affinities.shape[0]):
                if i not in remaining_items:
                    mask[i, :] = False
                    mask[:, i] = False
            masked_affinities = np.where(mask, affinities, -np.inf)
            flat_idx = np.argmax(masked_affinities)
            i, j = np.unravel_index(flat_idx, affinities.shape)
        pairs.append((min(i, j), max(i, j)))
        remaining_items.remove(i)
        remaining_items.remove(j)
        affinities[i, :] = -np.inf
        affinities[:, i] = -np.inf
        affinities[j, :] = -np.inf
        affinities[:, j] = -np.inf
    return pairs

def expert_clustering(affinity_matrix, target_routed_expert):
    clustering = SpectralClustering(
        n_clusters=target_routed_expert,
        affinity="precomputed"
    ).fit(affinity_matrix)

    mapping_dict = dict(zip(range(len(affinity_matrix)), clustering.labels_))
    inv_mapping_dict = {i:[] for i in range(target_routed_expert)}
    for i, index in zip(clustering.labels_, range(len(affinity_matrix))):
        inv_mapping_dict[i].append(index)
    return mapping_dict, inv_mapping_dict

## ************************ FLOWS ************************ ##
def compute_gate_loss(pred_indices, pred_weights, target_indices, target_weights):
    # Convert target indices and weights to device
    target_indices = torch.tensor(target_indices, device=pred_indices.device)
    target_weights = torch.tensor(target_weights, device=pred_weights.device)
    
    # Expert matching loss - measures how well we match the right experts
    indices_match = (pred_indices.unsqueeze(-1) == target_indices.unsqueeze(1)).any(dim=1)
    
    expert_match_loss = F.binary_cross_entropy_with_logits(
        indices_match.float(), 
        torch.ones_like(indices_match, dtype=torch.float)
    )
    
    aligned_weights = torch.zeros_like(target_weights)
    for i in range(pred_indices.size(0)):
        for j in range(pred_indices.size(1)):
            idx = pred_indices[i, j]
            # Find if idx is in target_indices[i]
            mask = (target_indices[i] == idx)
            if mask.any():
                # Get the position where idx is in target_indices
                pos = torch.where(mask)[0][0]
                aligned_weights[i, pos] = pred_weights[i, j]
    
    weight_loss = F.mse_loss(aligned_weights, target_weights)
    
    # Total loss
    total_loss = expert_match_loss + weight_loss
    return total_loss, expert_match_loss, weight_loss

def create_gate(gate, inv_mapping_dict, layer_norm, path_config, distillation_config, layer_idx, scoring_func, device):
    with open(os.path.join(path_config.expert_activations, f"layer_{layer_idx}.pickle"), "rb") as f:
        (top_k_output, top_k_weight) = pickle.load(f)

    gate=gate.to(device)
    gate.train()
    gate.config.n_routed_experts=distillation_config.target_routed_expert
    gate.config.num_experts_per_tok=distillation_config.target_active_expert
    gate.n_routed_experts=gate.config.n_routed_experts
    gate.top_k=gate.config.num_experts_per_tok
    gate.weight=torch.nn.Parameter(gate.weight[:distillation_config.target_routed_expert].to(device, dtype=torch.bfloat16))

    top_k_output=top_k_output.to('cpu').numpy()
    top_k_weight=top_k_weight.to('cpu').numpy()

    expert_indices = []
    expert_boundaries = [0]
    for i in range(distillation_config.target_routed_expert):
        expert_indices.extend(inv_mapping_dict[i])
        expert_boundaries.append(len(expert_indices))
        
    expert_indices = np.array(expert_indices)
    expert_boundaries = np.array(expert_boundaries)

    # Convert to numba-friendly structures
    expert_indices = []
    expert_boundaries = [0]
    for i in range(distillation_config.target_routed_expert):
        expert_indices.extend(inv_mapping_dict[i])
        expert_boundaries.append(len(expert_indices))
    expert_indices = np.array(expert_indices)
    expert_boundaries = np.array(expert_boundaries)

    # Determine scoring function ID (either softmax or sigmoid)
    scoring_function_id = 0 if scoring_func == "softmax" else 1

    top_k_weight = compute_new_weights(
        top_k_output, 
        top_k_weight, 
        expert_indices, 
        expert_boundaries,
        top_k_output.shape[0], 
        distillation_config.target_routed_expert, 
        scoring_function_id
    )

    top_k_output = np.argsort(top_k_weight, axis=-1)
    top_k_output = top_k_output[:, ::-1]
    top_k_weight = np.take_along_axis(top_k_weight, top_k_output, axis=-1)

    # n_epochs=min(distillation_config.n_epochs,1) ## no need to make too many epochs
    # optimizer=AdEMAMix(
    #     gate.parameters(),
    #     lr=5e-4
    # )

    # criterion = torch.nn.functional.smooth_l1_loss
    # total_batches = (len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}"))) - distillation_config.eval_batches) 
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches * n_epochs)

    # for i in range(n_epochs):
    #     progress_bar = tqdm(range(total_batches), desc=f"Creating new gate, epoch {i}")
    #     for batch_idx in progress_bar:
    #         hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device)[:, distillation_config.skip_first_tokens:]
    #         hidden_states = layer_norm(hidden_states)
            
    #         topk_idx_pred, topk_weight_pred, aux_loss = gate(hidden_states)
            
    #         bs=topk_idx_pred.shape[0]
    #         topk_idx_target=torch.tensor(top_k_output[bs * batch_idx:bs * (batch_idx+1), :distillation_config.target_active_expert].copy(), device=topk_idx_pred.device)
    #         topk_weight_target=torch.tensor(top_k_weight[bs * batch_idx:bs * (batch_idx+1), :distillation_config.target_active_expert].copy(), device=topk_weight_pred.device)
            
    #         pred = torch.zeros((bs, distillation_config.target_routed_expert), device=device)
    #         pred[torch.arange(bs).unsqueeze(1), topk_idx_pred] = topk_weight_pred
            
    #         target = torch.zeros((bs, distillation_config.target_routed_expert), device=device)
    #         topk_weight_target = topk_weight_target.to(target.dtype)
    #         target[torch.arange(bs).unsqueeze(1), topk_idx_target] = topk_weight_target
            
    #         loss = criterion(pred, target)
                
    #         if aux_loss is not None:
    #             loss = loss + aux_loss
        
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         scheduler.step()
            
    #         progress_bar.set_postfix(loss=loss.item())
    return gate

def prepare_distillat_state_cl(distilled_mlp, layer_norm, scoring_func, distillation_config, path_config, layer_idx, device):
    hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{0}"))[:, distillation_config.skip_first_tokens].to(device)
    hidden_states = layer_norm(hidden_states)
    hidden_states=torch.stack([distilled_mlp.experts[elt](hidden_states) for elt in range(len(distilled_mlp.experts))])
    affinity_matrix = build_affinity_matrix(hidden_states)
    
    mapping_dict, inv_mapping_dict=expert_clustering(affinity_matrix, distillation_config.target_routed_expert)
    
    ## Build new gate
    distilled_mlp.gate=create_gate(
        distilled_mlp.gate,
        inv_mapping_dict,
        layer_norm,
        path_config,
        distillation_config,
        layer_idx,
        scoring_func=scoring_func,
        device=device
    )
    
    distilled_mlp.gate=distilled_mlp.gate.to(dtype=torch.bfloat16)
    
    
    ## Dequant and merge the experts
    new_experts=deepcopy(distilled_mlp.experts[:distillation_config.target_routed_expert])
    
    for i, expert_list in tqdm(inv_mapping_dict.items()):
        expert_to_merge=[]
        for expert_index in tqdm(expert_list, leave=False):
            module = distilled_mlp.experts[expert_index].to(device)
            module_dequant = calibrated_dequant(module, layer_norm, path_config, layer_idx)
            expert_to_merge.append(module_dequant)
        
        merged_expert_state_dict=deepcopy(expert_to_merge[0].state_dict())
        
        for k in merged_expert_state_dict.keys():
            tensors=[expert.state_dict()[k] for expert in expert_to_merge]
            merged_expert_state_dict[k]=sce_merge(
                tensors,
                merged_expert_state_dict[k],
                select_topk=len(tensors) // 2 + 1
            )
            
        merged_expert = deepcopy(expert_to_merge[0])
        merged_expert.load_state_dict(merged_expert_state_dict)
        
        for expert in expert_to_merge:
            expert.to_empty(device='meta')
            
        new_experts[i] = merged_expert
        
    distilled_mlp.experts = new_experts
    return distilled_mlp

def prepare_distillat_act_cl(distilled_mlp, layer_norm, scoring_func,  distillation_config, path_config, layer_idx, device):
    with open(os.path.join(path_config.expert_activations, f"layer_{layer_idx}.pickle"), "rb") as f:
        (top_k_output, top_k_weight) = pickle.load(f)
        
    top_k_output=top_k_output.detach().to(torch.int64).cpu().numpy()
    
    
    affinity_matrix = cooccurrence_matrix(top_k_output, len(np.unique(top_k_output)))
    
    mapping_dict, inv_mapping_dict=expert_clustering(affinity_matrix, distillation_config.target_routed_expert)
    
    ## Build new gate
    distilled_mlp.gate=create_gate(
        distilled_mlp.gate,
        inv_mapping_dict,
        layer_norm,
        path_config,
        distillation_config,
        layer_idx,
        scoring_func=scoring_func,
        device=device
    )

    distilled_mlp.gate=distilled_mlp.gate.to(dtype=torch.bfloat16)
    
    
    ## Dequant and merge the experts
    new_experts=deepcopy(distilled_mlp.experts[:distillation_config.target_routed_expert])

    for i, expert_list in tqdm(inv_mapping_dict.items()):
        expert_to_merge=[]
        for expert_index in tqdm(expert_list, leave=False):
            module = distilled_mlp.experts[expert_index].to(device)
            module_dequant = calibrated_dequant(module, layer_norm, path_config, layer_idx)
            expert_to_merge.append(module_dequant)
        
        merged_expert_state_dict=deepcopy(expert_to_merge[0].state_dict())
        
        for k in merged_expert_state_dict.keys():
            tensors=[expert.state_dict()[k] for expert in expert_to_merge]
            
            merged_expert_state_dict[k]=torch.squeeze(sce_merge(
                tensors,
                merged_expert_state_dict[k],
                select_topk=0.3
            ))
            
            # merged_expert_state_dict[k]=multislerp(
            #     tensors,
            #     weight=[1/len(tensors)] * len(tensors),
            #     base_tensor=merged_expert_state_dict[k],
            # )
            

        merged_expert = deepcopy(expert_to_merge[0])
        merged_expert.load_state_dict(merged_expert_state_dict)
        
        for expert in expert_to_merge:
            expert.to_empty(device='meta')
            
        new_experts[i] = merged_expert
        
    distilled_mlp.experts = new_experts
    return distilled_mlp

def prepare_distillat_topk(distilled_mlp, layer_norm, distillation_config, path_config, layer_idx, device):
    with open(os.path.join(path_config.expert_activations, f"layer_{layer_idx}.pickle"), "rb") as f:
        (top_k_output, top_k_weight) = pickle.load(f)
        
    top_k_output=top_k_output.detach().to(torch.int64).cpu().numpy()
    v,c=np.unique(top_k_output, return_counts=True)

    expert_list=np.argsort(c)[-distillation_config.target_routed_expert:]
    
    
    gate=distilled_mlp.gate.to(device)
    gate.train()
    gate.config.n_routed_experts=distillation_config.target_routed_expert
    gate.config.num_experts_per_tok=distillation_config.target_active_expert
    gate.n_routed_experts=gate.config.n_routed_experts
    gate.top_k=gate.config.num_experts_per_tok

    w = deepcopy(gate.weight)[:distillation_config.target_routed_expert]
    new_experts=deepcopy(distilled_mlp.experts[:distillation_config.target_routed_expert])

    expert_to_merge=[]
    for expert_index in tqdm(expert_list, leave=False):
        # print(expert_index)
        module = distilled_mlp.experts[expert_index].to(device)
        module_dequant = calibrated_dequant(module, layer_norm, path_config, layer_idx)
        expert_to_merge.append(module_dequant)

    for i, index in enumerate(expert_list):
        new_experts[i] = expert_to_merge[i]
        w[i]=gate.weight[index]
        
    gate.weight=torch.nn.Parameter(w.to(device, dtype=torch.bfloat16))
    
    distilled_mlp.gate=gate
    distilled_mlp.experts =new_experts
    return distilled_mlp

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
        
        if len(mask.shape) != task_vectors.shape: ## Correction should happend only when relevant, else it will bug the whole stuff
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

def halve_distilled_mlp(distilled_mlp, layer_norm, distillation_config, path_config, layer_idx, device):
    for parameter in distilled_mlp.parameters():
        parameter.requires_grad=False

    new_target_routed_expert=distilled_mlp.config.n_routed_experts//2
    new_target_active_expert=distilled_mlp.config.num_experts_per_tok

    distilled_mlp.config.n_routed_experts=new_target_routed_expert
    distilled_mlp.config.num_experts_per_tok=new_target_active_expert
    distilled_mlp.n_routed_experts=new_target_routed_expert
    distilled_mlp.num_experts_per_tok=new_target_active_expert

    hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{0}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]
    hidden_states = layer_norm(hidden_states)

    topk_idx, topk_weight, aux_loss = distilled_mlp.gate(hidden_states)

    gate=distilled_mlp.gate.to(device)
    gate.train()
    gate.config.n_routed_experts=new_target_routed_expert
    gate.config.num_experts_per_tok=new_target_active_expert
    gate.n_routed_experts=gate.config.n_routed_experts
    gate.top_k=gate.config.num_experts_per_tok

    w = deepcopy(gate.weight)[:new_target_routed_expert]
    new_experts=deepcopy(distilled_mlp.experts[:new_target_routed_expert])
    topk_idx=topk_idx.detach().to(torch.int64).cpu().numpy()
    
    # v,c=np.unique(topk_idx, return_counts=True)
    # expert_list=np.argsort(c)[-distillation_config.target_routed_expert:]
    
    # for i, index in enumerate(expert_list):
    #     expert_1_state_dict=distilled_mlp.experts[index].state_dict()
    #     expert_2_state_dict=distilled_mlp.experts[i + len(expert_list)].state_dict()
        
    #     for k in expert_1_state_dict.keys():
    #         # expert_1_state_dict[k] = slerp(
    #         #     0.15,
    #         #     expert_1_state_dict[k], 
    #         #     expert_2_state_dict[k],
    #         # )
    #         expert_1_state_dict[k]=sce_merge(
    #             [expert_1_state_dict[k], expert_2_state_dict[k]],
    #             expert_1_state_dict[k],
    #             select_topk=0.5
    #         )
        
    #     new_experts[i] = deepcopy(distilled_mlp.experts[index])
    #     new_experts[i].load_state_dict(expert_1_state_dict)
    #     w[i]= gate.weight[index]
    
    # affinity_matrix=cooccurrence_matrix(topk_idx, len(np.unique(topk_idx)))
    affinity_matrix=build_affinity_matrix(hidden_states)
    affinity_matrix=(affinity_matrix - affinity_matrix.min())/(affinity_matrix.max()-affinity_matrix.min())
    pairs=pair_items_by_affinity(affinity_matrix)
    
    for i, pair in enumerate(pairs):
        expert_1_state_dict=distilled_mlp.experts[pair[0]].state_dict()
        expert_2_state_dict=distilled_mlp.experts[pair[1]].state_dict()
        for k in expert_1_state_dict.keys():
            expert_1_state_dict[k]=sce_merge(
                [expert_1_state_dict[k], expert_2_state_dict[k]],
                expert_1_state_dict[k],
                select_topk=0.25
            )
            new_experts[i] = deepcopy(distilled_mlp.experts[pair[0]])
            new_experts[i].load_state_dict(expert_1_state_dict)
            w[i]= gate.weight[pair[0]]
        
    gate.weight=torch.nn.Parameter(w.to(device, dtype=torch.bfloat16))
    distilled_mlp.gate=gate
    distilled_mlp.experts =new_experts

    return distilled_mlp
    
    
def prepare_moe_for_distillation(distilled_mlp, distillation_config, path_config, layer_idx, device, dtype=torch.bfloat16):
    # distilled_mlp, _=dequantize_GEMM(distilled_mlp)
    for name, parameter in distilled_mlp.named_parameters():
        if 'gate.' in name:
            parameter.requires_grad=True
        else:
            parameter.requires_grad=False
    
    for name, module in tqdm(distilled_mlp.named_modules()):
        if isinstance(module, torch.nn.Linear):
            rsetattr(
                distilled_mlp,
                name,
                lora.Linear(
                    module,
                    adapter_name="adapter",
                    r=distillation_config.dora_rank,
                    lora_alpha=distillation_config.dora_rank,
                    lora_dropout=0.05,
                    use_dora=True,
                ).to(device=device, dtype=dtype)
            )
            
    train_batches = len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}"))) - distillation_config.eval_batches

    optimizer = AdEMAMix(
        filter(lambda p: p.requires_grad, distilled_mlp.parameters()),
        lr=distillation_config.learning_rate,
        betas=(0.7, 0.999, 0.9999),
        alpha=5
    )
    
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=distillation_config.gradient_accumulation_steps * 0, ## warmup for 32 virtual steps
        total_steps=train_batches,
        min_lr=distillation_config.learning_rate * distillation_config.end_factor
    )

    criterion = torch.nn.functional.smooth_l1_loss
    return distilled_mlp, optimizer, scheduler, criterion

def merge_and_unload(distilled_mlp):
    
    for name, module in tqdm(distilled_mlp.named_modules()):
        if isinstance(module, lora.Linear):
            module.merge()
            
            rsetattr(
                distilled_mlp,
                name,
                module.base_layer
            )
    return distilled_mlp