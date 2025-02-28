from sklearn.cluster import SpectralClustering
import torch.nn.functional as F
from typing import List, Union
from numba import njit, prange
from tqdm.auto import tqdm
from copy import deepcopy
import _pickle as pickle
import numpy as np
import numba as nb
import torch

from awq.modules.linear.gemm import WQLinear_GEMM
from awq.modules.triton.gemm import awq_gemm_triton, awq_dequantize_triton

from utils.torch_utils import load_quant, rsetattr
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

    # n_epochs=min(distillation_config.n_epochs,3) ## no need to make too many epochs
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

def calibrated_merge_experts(expert_list, layer_norm, distillation_config, distilled_mlp, path_config, layer_idx):
    """
    Iteratively merge experts using pairwise SLERP with calibration after each merge.
    
    The algorithm:
    1. Applies SLERP to experts two by two (if number is odd, one is left for the next round)
    2. Runs one epoch of calibration on each merged result based on the specific experts merged
    3. Stops when only one expert remains
    
    Args:
        expert_list: List of expert indices to merge
        layer_norm: Layer normalization module
        distillation_config: Configuration for distillation
        distilled_mlp: MLP module containing the experts
        path_config: Configuration for file paths
        layer_idx: Current layer index
        
    Returns:
        The final merged expert
    """
    # First, load all the experts
    print(f"Loading {len(expert_list)} experts for iterative merging")
    experts_map = {}  # Map to store the experts by their original index
    
    for expert_index in tqdm(expert_list):
        module = distilled_mlp.experts[expert_index].to('cuda:0')
        
        module_dequant = calibrated_dequant(module, layer_norm, path_config, layer_idx)
        
        experts_map[expert_index] = {
            'quantized': module,
            'dequantized': module_dequant,
            'original_indices': [expert_index]  # Track which original experts are part of this expert
        }
    
    # Keep track of current round's expert indices
    current_round_indices = list(expert_list)
    
    # Keep merging until only one expert remains
    round_num = 0
    criterion = torch.nn.L1Loss()
    
    while len(current_round_indices) > 1:
        round_num += 1
        print(f"\nRound {round_num}: Merging {len(current_round_indices)} experts")
        next_round_indices = []
        
        # Process experts in pairs
        i = 0
        while i < len(current_round_indices) - 1:
            idx1 = current_round_indices[i]
            idx2 = current_round_indices[i+1]
            print(f"  Merging experts {idx1} and {idx2}")
            
            # Get the expert data
            expert1 = experts_map[idx1]['dequantized']
            expert2 = experts_map[idx2]['dequantized']
            
            # Track which original experts are involved in this merge
            original_indices = experts_map[idx1]['original_indices'] + experts_map[idx2]['original_indices']
            print(f"  This merge involves original experts: {original_indices}")
            
            # Create a new expert with merged weights using SLERP
            merged_expert = deepcopy(expert1)
            # new_state_dict = expert1.state_dict()
            
            # for k in new_state_dict.keys():
            #     # Equal weights (0.5) for pairwise merging
            #     new_state_dict[k] = slerp(0.5, expert1.state_dict()[k], expert2.state_dict()[k])
            
            # merged_expert.load_state_dict(new_state_dict)
            

            # Generate a new index for the merged expert
            merged_idx = max(experts_map.keys()) + 1
            
            # Calibrate the merged expert against ONLY the experts involved in this merge
            print(f"  Calibrating merged expert {merged_idx} against experts {original_indices}")
            optimizer = AdEMAMix(
                merged_expert.parameters(),
                lr=8e-4
            )
            
            n_epochs=min(distillation_config.n_epochs,3)
            total_batches = (len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}"))) - distillation_config.eval_batches) 
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_batches*n_epochs, 
                eta_min=8e-6
            )
            
            # One epoch of calibration
            for i in range(n_epochs):
                progress_bar = tqdm(range(total_batches), desc=f"Calibration batches, epoch {i}")
                for batch_idx in progress_bar:
                    # Load hidden states
                    hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}"))
                    hidden_states = layer_norm(hidden_states)
                    
                    # Forward pass through merged expert
                    merged_output = merged_expert(hidden_states)
                    
                    # Forward pass through the original experts involved in this merge
                    involved_outputs = []
                    for orig_idx in original_indices:
                        # Use the quantized version for forward pass
                        orig_expert = distilled_mlp.experts[orig_idx].to('cuda:0')
                        involved_outputs.append(orig_expert(hidden_states))
                    
                    # Average the outputs from the involved experts
                    target_output = torch.mean(torch.stack(involved_outputs), dim=0)
                    
                    # Optimize merged expert
                    loss = criterion(merged_output, target_output)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    progress_bar.set_postfix(loss=loss.item())
                
            # Add the merged expert to the map
            experts_map[merged_idx] = {
                'quantized': None,  # We don't need the quantized version for merged experts
                'dequantized': merged_expert,
                'original_indices': original_indices
            }
            
            # Add the new merged expert index to the next round
            next_round_indices.append(merged_idx)
            i += 2
        
        # If there's an odd expert left, add it to the next round
        if i < len(current_round_indices):
            left_idx = current_round_indices[i]
            print(f"  Expert {left_idx} left for next round")
            next_round_indices.append(left_idx)
        
        # Update current round indices for next round
        current_round_indices = next_round_indices
        print(f"  Round {round_num} complete. {len(current_round_indices)} experts remaining.")
    
    # Return the final expert (only one should remain)
    assert len(current_round_indices) == 1, "Error: More than one expert remains at the end"
    final_idx = current_round_indices[0]
    return experts_map[final_idx]['dequantized']