import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from mergekit.merge_methods.multislerp import multislerp
from tqdm.auto import tqdm

from utils.torch_utils import memory_cleanup
from utils.grokfast import gradfilter_ema

from copy import deepcopy

from utils.ademamix import AdEMAMix
# from bitsandbytes.optim.ademamix import AdEMAMix8bit as AdEMAMix
from utils.patched_sce import sce_merge
from utils.experts_merge_utils import group_items_by_affinity, dequantize_GEMM
import torch
import torch.nn as nn
import torch.nn.functional as F

# class FusedLinear(nn.Module):
#     def __init__(self, in_features, out_features, rank=8, n_fused=4, bias=False, **kwargs):
#         super().__init__()
        
#         self.in_features = in_features
#         self.out_features = out_features
#         self.rank = rank
#         self.n_fused = n_fused
        
#         # Linear layer
#         self.fused_layer = nn.Linear(in_features, out_features, bias=bias)
        
#         # Adapter weights
#         self.qa_weights = nn.Parameter(torch.randn(n_fused, rank, in_features) * 0.02)
#         self.qb_weights = nn.Parameter(torch.randn(n_fused, out_features, rank) * 0.02)
        
#         # Scaling factor
#         self.scaling_factor = nn.Parameter(torch.ones(n_fused, out_features))
        
#         # Flag to check if scaling factor has been computed
#         self.scaling_factor_computed = [False] * n_fused
    
#     def forward(self, x, i):
#         # Linear transformation
#         output = self.fused_layer(x)

#         intermediate = F.linear(x, self.qa_weights[i])
#         adapter_output = F.linear(intermediate, self.qb_weights[i])

#         if not self.scaling_factor_computed[i]:
#             fused_output_norm = torch.norm(output, dim=0)
#             lora_output_norm = torch.norm(adapter_output, dim=0)
#             lora_output_norm = lora_output_norm + 1e-8
#             scaling_factor = fused_output_norm / lora_output_norm
#             self.scaling_factor.data[i] = scaling_factor
#             self.scaling_factor_computed[i] = True
            
#         return output + 1.0 * adapter_output #* self.scaling_factor[i][None]
    
#     def fuse(self, layers, merge_method='sce', device="cuda:0"):
#         for param in self.parameters():
#             param.requires_grad = False
            
#         weights = [layer.weight for layer in layers]
#         if merge_method == 'sce':
#             fused_weight = sce_merge(weights, weights[0], select_topk=1)
#         elif merge_method == 'slerp':
#             fused_weight = multislerp(weights, [1/len(weights)]*len(weights), weights[0])
#         elif merge_method == 'mean':
#             fused_weight = torch.mean(torch.stack(weights), dim=0)
#         elif merge_method == 'greedy':
#             fused_weight = weights[0].clone()
#         else:
#             raise ValueError(f"Unknown merge method: {merge_method}")
            
#         self.fused_layer.weight = nn.Parameter(fused_weight)
        
#         # Pre-compute scaling factors based on relative norms
#         with torch.no_grad():
#             for i, weight in enumerate(weights):
#                 if i >= self.n_fused:
#                     break

#                 weight_diff = weight - fused_weight
                
#                 # Use torch.svd_lowrank for efficiency
#                 U, S, V = torch.svd_lowrank(weight_diff.to(dtype=torch.float32, device=device), q=self.rank, niter=2)
#                 scaling_factor = torch.sum(S) / self.rank
#                 sqrt_S = torch.sqrt(S / scaling_factor)
                
#                 # Update both parameter tensors and modules
#                 qa_weight = (torch.diag(sqrt_S) @ V.T).to(device=fused_weight.device, dtype=fused_weight.dtype)
#                 qb_weight = (U @ torch.diag(sqrt_S)).to(device=fused_weight.device, dtype=fused_weight.dtype)
                
#                 self.qa_weights[i].copy_(qa_weight)
#                 self.qb_weights[i].copy_(qb_weight)
        
#         for params in self.parameters():
#             params.requires_grad = True
            
# class FusedMLP(torch.nn.Module):
#     def __init__(self, config, hidden_size=None, intermediate_size=None, n_fused=4, rank=8, adapter_type='lora'):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
#         self.intermediate_size = (
#             config.moe_intermediate_size if intermediate_size is None else intermediate_size
#         )
#         self.n_fused=n_fused
#         self.gate_proj = FusedLinear(self.hidden_size, self.intermediate_size, bias=False, rank=rank, n_fused=n_fused, adapter_type=adapter_type)
#         self.up_proj = FusedLinear(self.hidden_size, self.intermediate_size, bias=False, rank=rank, n_fused=n_fused, adapter_type=adapter_type)
#         self.down_proj = FusedLinear(self.intermediate_size, self.hidden_size, bias=False, rank=rank, n_fused=n_fused, adapter_type=adapter_type)
#         self.act_fn = ACT2FN[config.hidden_act]
#         self.adapter_type=adapter_type

#     def fuse(self, experts, merge_method='sce', device='cuda:0'):
#         assert len(experts) == self.n_fused, "Warning, passed number of experts doesn't match the number of initialized fused experts"
        
#         self.gate_proj.fuse([exp.gate_proj for exp in experts],merge_method=merge_method, device=device)
#         self.up_proj.fuse([exp.up_proj for exp in experts],merge_method=merge_method, device=device)
#         self.down_proj.fuse([exp.down_proj for exp in experts],merge_method=merge_method, device=device)
        
#         self.mask_up_proj = torch.nn.Linear(self.n_fused, self.hidden_size, bias=False)
#         self.mask_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        
#     def forward(self, x, top_k_weights):
        
#         if torch.all(torch.abs(top_k_weights) < 1e-6): # in the case where no experts are activated in that specific fused experts, skip
#             return x
            
#         x = x + self.mask_up_proj(top_k_weights)
#         down_proj = x

#         for i in range(self.n_fused):
            
#             active_mask = torch.abs(top_k_weights[:, i]) > 1e-6
#             active_count = active_mask.sum().item()
            
#             if active_count > 0:
#                 # Only process if there are any active items for this adapter
#                 active_indices = torch.nonzero(active_mask).squeeze(-1)
#                 active_x = x[active_indices]
#                 active_weights = top_k_weights[active_indices, i:i+1]
                
#                 # Compute adapter output for active batch items
#                 active_output = self.down_proj(self.act_fn(self.gate_proj(active_x, i)) * self.up_proj(active_x, i), i)
#                 down_proj[active_indices] = down_proj[active_indices] + active_output * active_weights
                
#         return down_proj

class FusedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, n_fused=4, bias=False, **kwargs):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.n_fused = n_fused
        
        # Linear layer
        self.fused_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # Adapter weights
        self.qa_weights = nn.Parameter(torch.randn(n_fused, rank, in_features) * 0.02)
        self.qb_weights = nn.Parameter(torch.randn(n_fused, out_features, rank) * 0.02)
        
        # Scaling factor
        self.scaling_factor = nn.Parameter(torch.ones(n_fused, out_features))
        
        # Flag to check if scaling factor has been computed
        self.scaling_factor_computed = [False] * n_fused
    
    def forward(self, x, top_k_weights):
        # Linear transformation
        output = self.fused_layer(x)
        
        # Adapter mechanism
        if torch.all(torch.abs(top_k_weights) < 1e-6):
            return output
        
        adapter_output = torch.zeros_like(output)
        batch_size = x.shape[0]
        
        # Process each adapter separately
        
        if len(x.shape) == 2: # for some reasons there are errors here
            for i in range(self.n_fused):
                # Find batch items where this adapter is active
                active_mask = torch.abs(top_k_weights[:, i]) > 1e-6
                active_count = active_mask.sum().item()
                
                if active_count > 0:
                    # Only process if there are any active items for this adapter
                    active_indices = torch.nonzero(active_mask).squeeze(-1)
                    active_x = x[active_indices]
                    active_weights = top_k_weights[active_indices, i:i+1]
                    
                    # Compute adapter output for active batch items
                    intermediate = F.linear(active_x, self.qa_weights[i])
                    active_output = F.linear(intermediate, self.qb_weights[i])
                    
                    active_output = active_output * active_weights
                    
                    # Compute scaling factor during the first forward pass
                    if not self.scaling_factor_computed[i]:
                        fused_output_norm = torch.norm(output[active_indices], dim=0)
                        lora_output_norm = torch.norm(active_output, dim=0)
                        
                        # Avoid division by zero
                        lora_output_norm = lora_output_norm + 1e-8
                        
                        # Compute the scaling factor
                        scaling_factor = fused_output_norm / lora_output_norm
                        
                        # Update the scaling factor
                        self.scaling_factor.data[i] = scaling_factor
                        
                        # Set the flag to True to avoid recomputation
                        self.scaling_factor_computed[i] = True
                    
                    # Add to the corresponding positions in adapter_output
                    adapter_output[active_indices] = adapter_output[active_indices] + 0.1 * active_output * self.scaling_factor[i][None]
        
        return output + adapter_output
    
    def fuse(self, layers, merge_method='sce', device="cuda:0"):
        for param in self.parameters():
            param.requires_grad = False
            
        weights = [layer.weight for layer in layers]
        if merge_method == 'sce':
            fused_weight = sce_merge(weights, weights[0], select_topk=len(weights)//2)
        elif merge_method == 'slerp':
            fused_weight = multislerp(weights, [1/len(weights)]*len(weights), weights[0])
        elif merge_method == 'mean':
            fused_weight = torch.mean(torch.stack(weights), dim=0)
        elif merge_method == 'greedy':
            fused_weight = weights[0].clone()
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")
            
        self.fused_layer.weight = nn.Parameter(fused_weight)
        
        # Pre-compute scaling factors based on relative norms
        with torch.no_grad():
            for i, weight in enumerate(weights):
                if i >= self.n_fused:
                    break

                weight_diff = weight - fused_weight
                
                # Use torch.svd_lowrank for efficiency
                U, S, V = torch.svd_lowrank(weight_diff.to(dtype=torch.float32, device=device), q=self.rank, niter=2)
                scaling_factor = torch.sum(S) / self.rank
                sqrt_S = torch.sqrt(S / scaling_factor)
                
                # Update both parameter tensors and modules
                qa_weight = (torch.diag(sqrt_S) @ V.T).to(device=fused_weight.device, dtype=fused_weight.dtype)
                qb_weight = (U @ torch.diag(sqrt_S)).to(device=fused_weight.device, dtype=fused_weight.dtype)
                
                self.qa_weights[i].copy_(qa_weight)
                self.qb_weights[i].copy_(qb_weight)
        
        for params in self.parameters():
            params.requires_grad = True
            
class FusedMLP(torch.nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None, n_fused=4, rank=8, adapter_type='lora'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.moe_intermediate_size if intermediate_size is None else intermediate_size
        )
        self.n_fused=n_fused
        self.gate_proj = FusedLinear(self.hidden_size, self.intermediate_size, bias=False, rank=rank, n_fused=n_fused, adapter_type=adapter_type)
        self.up_proj = FusedLinear(self.hidden_size, self.intermediate_size, bias=False, rank=rank, n_fused=n_fused, adapter_type=adapter_type)
        self.down_proj = FusedLinear(self.intermediate_size, self.hidden_size, bias=False, rank=rank, n_fused=n_fused, adapter_type=adapter_type)
        self.act_fn = ACT2FN[config.hidden_act]
        self.adapter_type=adapter_type

    def fuse(self, experts, merge_method='sce', device='cuda:0'):
        assert len(experts) == self.n_fused, "Warning, passed number of experts doesn't match the number of initialized fused experts"
        
        self.gate_proj.fuse([exp.gate_proj for exp in experts],merge_method=merge_method, device=device)
        self.up_proj.fuse([exp.up_proj for exp in experts],merge_method=merge_method, device=device)
        self.down_proj.fuse([exp.down_proj for exp in experts],merge_method=merge_method, device=device)
        
        self.mask_up_proj = torch.nn.Linear(self.n_fused, self.hidden_size, bias=False)
        self.mask_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, x, top_k_weights):
        x = x + self.mask_up_proj(top_k_weights)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, top_k_weights)) * self.up_proj(x, top_k_weights), top_k_weights)
        return down_proj

class FusedMOE(torch.nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.config = moe.config
        self.num_experts_per_tok = moe.num_experts_per_tok

        self.ep_size = moe.ep_size
        self.experts_per_rank = moe.experts_per_rank
        self.ep_rank = moe.ep_rank
        self.experts = moe.experts
        self.gate = moe.gate
        self.ready = False
        if self.config.n_shared_experts is not None:
            self.shared_experts = moe.shared_experts
        # Register inv_mapping_dict as a buffer
        self.register_buffer('inv_mapping_dict', torch.tensor([]), persistent=True)
        
        for name, params in self.named_parameters():
            params.requires_grad = False

    def set_ready(self):
        self.experts.to_empty(device="meta")
        del self.experts
        self.ready = True

    def fuse(self, affinity_matrix, group_size, train_batches, learning_rate, device, merge_method='sce', adapter_type='mixture', rank=8, low_vram=True):
        inv_mapping_dict = group_items_by_affinity(affinity_matrix, group_size)

        # Convert the dictionary to a tensor for saving in state_dict
        inv_mapping_tensor = torch.tensor(inv_mapping_dict, dtype=torch.int64)
        self.register_buffer('inv_mapping_dict', inv_mapping_tensor, persistent=True)

        fused_experts = []
        for k, v in tqdm(enumerate(inv_mapping_dict)):
            fused_moe = FusedMLP(self.config, n_fused=group_size, rank=rank, adapter_type=adapter_type).to("cuda")
            fusion_device="cpu" if low_vram else device # significantly slower on cpu, but can save some vram
            
            fused_moe.fuse([dequantize_GEMM(self.experts[i], destruct=False, return_params=False).to(fusion_device) for i in v], merge_method=merge_method, device=device) 
            fused_moe = fused_moe.to(device, dtype=torch.bfloat16)
            fused_experts.append(fused_moe)
            memory_cleanup()
            
        self.fused_experts = torch.nn.ModuleList(fused_experts).to(device, dtype=torch.bfloat16)

    def train_mode(self, learning_rate, train_batches):
        self.train()
        self.optimizer = AdEMAMix(
            self.fused_experts.parameters(),
            # [p for p in self.fused_experts.parameters() if p.requires_grad],
            lr=learning_rate,
            betas=(0.9, 0.999, 0.9999),
            alpha=5,
            weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_batches,
            eta_min=learning_rate
        )

        self.criterion = torch.nn.functional.smooth_l1_loss
        self.step=0
        self.grads = None

    def train_step(self, hidden_states, layer_norm, temperature=5, output=None, gradient_accumulation_step=1):
        residual = deepcopy(hidden_states)
        hidden_states = layer_norm(hidden_states)
    
        pred = self.forward(hidden_states) + residual
        
        if output is None:
            true = self.forward_origin(hidden_states) + residual
        else:
            true=output
        
        T = temperature
        
        # m = torch.mean(true, dim=-1, keepdim=True)
        # s = torch.std(true, dim=-1, keepdim=True)
        # pred = (pred - m) / s
        # true = (true - m) / s
        
        loss = self.criterion(T * pred, T * true, reduction='mean')
        loss.backward()
        self.step += 1
        if self.step == gradient_accumulation_step:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # self.grads = gradfilter_ema(self, grads=self.grads, alpha=0.98, lamb=2.0) # if you want to activate grokfast, uncomment. Untested
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step=0
        
        self.scheduler.step()
        return loss

    def forward(self, hidden_states):
        identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss = self.forward_gate(hidden_states)

        y = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)

        for idx in range(self.inv_mapping_dict.size(0)):
            y += self.forward_fused_expert(idx, hidden_states, topk_idx, topk_weight)

        y = y.view(*orig_shape)
        
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def forward_gate(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        gate_outputs=self.gate(hidden_states)
        if len(gate_outputs)==3:
            topk_idx, topk_weight, aux_loss = gate_outputs
        else:
            aux_loss=0.0
            topk_idx, topk_weight = gate_outputs
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        return identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss

    def forward_origin(self, hidden_states):
        identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss = self.forward_gate(hidden_states)

        y = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for idx in range(self.inv_mapping_dict.size(0)):
            y += self.forward_non_fused_expert(idx, hidden_states, topk_idx, topk_weight)

        y = y.view(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    def forward_fused_expert(self, idx, hidden_states, topk_idx, topk_weight):
        indexes = self.inv_mapping_dict[idx].tolist()

        flat_topk_weight = torch.zeros((hidden_states.shape[0], len(indexes)), device=hidden_states.device, dtype=hidden_states.dtype)

        for i, index in enumerate(indexes):
            flat_topk_weight[:, i] = torch.sum(topk_weight * (topk_idx == index), axis=-1)

        scalar = torch.sum(flat_topk_weight, axis=-1, keepdim=True)  # keeping the total weight of the experts

        flat_topk_weight[flat_topk_weight == 0] = -1e9
        flat_topk_weight = torch.softmax(flat_topk_weight, dim=-1)
        
        output = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
        
        output[scalar.squeeze() != 0] = self.fused_experts[idx](hidden_states[scalar.squeeze() != 0], flat_topk_weight[scalar.squeeze() != 0]) # Process only if at least one weight is required, should be much faster
        
        return  scalar * output  # Weighting is already taken into account by how the multiplex is trained

    def forward_non_fused_expert(self, idx, hidden_states, topk_idx, topk_weight):
        indexes = self.inv_mapping_dict[idx].tolist()

        flat_topk_weight = torch.zeros((hidden_states.shape[0], len(indexes)), device=hidden_states.device, dtype=hidden_states.dtype)

        for i, index in enumerate(indexes):
            flat_topk_weight[:, i] = torch.sum(topk_weight * (topk_idx == index), axis=-1)

        y = torch.sum(torch.stack([self.experts[index](hidden_states) for index in indexes], dim=-1) * flat_topk_weight[:, None], dim=-1)
        return y