from sklearn.cluster import SpectralClustering
import torch.nn.functional as F
from numba import njit, prange
from tqdm.auto import tqdm
import _pickle as pickle
import numpy as np
import numba as nb
import torch

from utils.torch_utils import load_quant
from utils.ademamix import AdEMAMix
import os

@nb.njit(parallel=True)
def cooccurrence_matrix(indices_list, n_unique):
    matrix = np.zeros((n_unique, n_unique), dtype=np.int64)
    
    for i in range(len(indices_list)):
        indices = indices_list[i]  
        for idx1 in indices:
            for idx2 in indices:
                matrix[idx1, idx2] += 1
    return matrix

# Define numba functions for sigmoid operations
@njit
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit
def sigmoid_inv(x):
    # Inverse of sigmoid: log(x / (1-x))
    return np.log(x / (1.0 - x))

# Numba-accelerated weight computation
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

def clusterize_experts(top_k_output, top_k_weight, target_routed_expert, scoring_func):
    v,c=np.unique(top_k_output, return_counts=True)
    matrix=cooccurrence_matrix(top_k_output, len(v))
    
    # First part remains the same (clustering)
    clustering = SpectralClustering(
        n_clusters=target_routed_expert,
        affinity="precomputed"
    ).fit(matrix)
    mapping_dict = dict(zip(v, clustering.labels_))
    
    # Create expert to indices mapping
    inv_mapping_dict = {i:[] for i in range(target_routed_expert)}
    for i, index in zip(clustering.labels_, v):
        inv_mapping_dict[i].append(index)

    # Convert to numba-friendly structures
    expert_indices = []
    expert_boundaries = [0]
    for i in range(target_routed_expert):
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
        target_routed_expert, 
        scoring_function_id
    )
    
    top_k_output = np.argsort(top_k_weight, axis=-1)
    top_k_output = top_k_output[:, ::-1]
    top_k_weight = np.take_along_axis(top_k_weight, top_k_output, axis=-1)

    return top_k_output, top_k_weight, mapping_dict, inv_mapping_dict

def create_gate(gate, layer_norm, path_config, distillation_config, layer_idx, scoring_func, device):
    
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
    
    top_k_output, top_k_weight, mapping_dict, inv_mapping_dict = clusterize_experts(
        top_k_output,
        top_k_weight,
        target_routed_expert=distillation_config.target_routed_expert,
        scoring_func=scoring_func
    )
    
    optimizer=AdEMAMix(
        gate.parameters(),
        lr=5e-3
    )
    
    criterion = torch.nn.MSELoss()
    total_batches = len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}")))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches)
    progress_bar = tqdm(range(total_batches), desc="Creating new gate")
    for batch_idx in progress_bar:
        hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device)
        hidden_states = layer_norm(hidden_states)
        
        topk_idx_pred, topk_weight_pred, aux_loss = gate(hidden_states)
        
        bs=topk_idx_pred.shape[0]
        topk_idx_target=torch.tensor(top_k_output[bs * batch_idx:bs * (batch_idx+1), :distillation_config.target_active_expert].copy(), device=topk_idx_pred.device)
        topk_weight_target=torch.tensor(top_k_weight[bs * batch_idx:bs * (batch_idx+1), :distillation_config.target_active_expert].copy(), device=topk_weight_pred.device)
        
        pred = torch.zeros((bs, distillation_config.target_routed_expert), device=device)
        pred[torch.arange(bs).unsqueeze(1), topk_idx_pred] = topk_weight_pred
        
        target = torch.zeros((bs, distillation_config.target_routed_expert), device=device)
        topk_weight_target = topk_weight_target.to(target.dtype)
        target[torch.arange(bs).unsqueeze(1), topk_idx_target] = topk_weight_target
        
        loss = criterion(pred, target)
            
        if aux_loss is not None:
            loss = loss + aux_loss
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix(loss=loss.item())
    return gate, mapping_dict, inv_mapping_dict