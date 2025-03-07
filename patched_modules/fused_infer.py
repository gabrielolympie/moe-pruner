import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import torch.distributed as dist
from utils.experts_merge_utils import group_items_by_affinity, dequantize_GEMM

class FusedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1, n_fused=4, adapter_type="mixture", bias=False, **kwargs):
        super().__init__()
        
        self.rank = rank
        self.adapter_type = adapter_type
        self.fused_layer = nn.Linear(in_features, out_features, bias=bias)
        
        if self.adapter_type == 'lora':
            self.qa_weights = nn.Parameter(torch.randn(rank, in_features) * 0.02)
            self.qb_weights = nn.Parameter(torch.randn(out_features, rank) * 0.02)
            self.mask_up_proj = nn.Parameter(torch.randn(n_fused, rank) * 0.02)
            self.scaling_factor = nn.Parameter(torch.Tensor([0.1] * out_features))
            
        if self.adapter_type == 'mixture':
            self.n_fused = n_fused
            # For efficient forward pass, create weight tensors
            self.qa_weights = nn.Parameter(torch.stack([torch.zeros(rank, in_features) for i in range(n_fused)]))
            self.qb_weights = nn.Parameter(torch.stack([torch.zeros(out_features, rank) for i in range(n_fused)]))
            self.scaling_factor = nn.Parameter(torch.Tensor([0.1] * out_features))
    
    def forward(self, x, top_k_weights):
        output = self.fused_layer(x)
        
        if self.adapter_type == 'lora': 
            x = torch.einsum('bh,rh->br', x, self.qa_weights)
            x = torch.einsum('br,brr->br', x, torch.diag_embed(torch.einsum('bk,kr -> br', top_k_weights, self.mask_up_proj)))
            x = torch.einsum('br,hr ->bh', x, self.qb_weights)
            output = output + self.scaling_factor[None] * x
            
        if self.adapter_type == 'mixture':
            x = torch.einsum('bh,krh->bkr', x, self.qa_weights)
            x = torch.einsum('bkr,khr->bkh', x, self.qb_weights)
            x = torch.einsum('bkh,bk->bkh', x, top_k_weights)
            x = torch.sum(x, dim=1)
            output=output + self.scaling_factor[None] * x
        return output

class FusedMLP(torch.nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None, n_fused=4, rank=8, adapter_type='mixture'):
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
        self.mask_up_proj = torch.nn.Linear(self.n_fused, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.adapter_type=adapter_type

    def forward(self, x, top_k_weights):
        x = x + self.mask_up_proj(top_k_weights)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, top_k_weights)) * self.up_proj(x, top_k_weights), top_k_weights)
        return down_proj

class FusedMOE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        FusedMLP(
                            config,
                            intermediate_size=config.moe_intermediate_size,
                            n_fused=config.n_routed_experts // config.n_fused_experts,
                            rank=config.fused_expert_dora_rank,
                            adapter_type=config.fused_expert_method
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_fused_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    FusedMLP(
                        config,
                        intermediate_size=config.moe_intermediate_size,
                        n_fused=config.n_routed_experts // config.n_fused_experts,
                        rank=config.fused_expert_dora_rank,
                        adapter_type=config.fused_expert_method
                    )
                    for i in range(config.n_fused_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=intermediate_size
            )

        # Register inv_mapping_dict as a buffer
        self.register_buffer('inv_mapping_dict', torch.zeros(config.n_multiplexed_routed_experts, config.n_routed_experts // config.n_multiplexed_routed_experts), persistent=True)


    def set_ready(self):
        self.experts.to_empty(device="meta")
        del self.experts
        self.ready = True

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
        
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        return identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss

    def forward_fused_expert(self, idx, hidden_states, topk_idx, topk_weight):
        indexes = self.inv_mapping_dict[idx].tolist()

        flat_topk_weight = torch.zeros((hidden_states.shape[0], len(indexes)), device=hidden_states.device, dtype=hidden_states.dtype)

        for i, index in enumerate(indexes):
            flat_topk_weight[:, i] = torch.sum(topk_weight * (topk_idx == index), axis=-1)

        scalar = torch.sum(flat_topk_weight, axis=-1, keepdim=True)  # keeping the total weight of the experts

        flat_topk_weight[flat_topk_weight == 0] = -1e9
        flat_topk_weight = torch.softmax(flat_topk_weight, dim=-1)
        
        output = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
        
        output[scalar.squeeze() != 0] = self.experts[idx](hidden_states[scalar.squeeze() != 0], flat_topk_weight[scalar.squeeze() != 0]) # Process only if at least one weight is required, should be much faster
        
        return  scalar * output  # Weighting is already taken into account by how the multiplex is trained
