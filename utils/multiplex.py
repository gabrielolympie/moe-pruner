
from utils.experts_merge_utils import group_items_by_affinity, dequantize_GEMM

from utils.ademamix import AdEMAMix
from utils.patched_sce import sce_merge
from mergekit.merge_methods.multislerp import multislerp

from transformers.activations import ACT2FN
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplexedMLP(torch.nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.moe_intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def load_experts(self, experts, merge_method='sce'):
        
        for params in self.parameters():
            params.requires_grad=False

        if len(experts)>1 and merge_method!='greedy':
            if merge_method=='sce':
                self.gate_proj.weight = torch.nn.Parameter(sce_merge([exp.gate_proj.weight for exp in experts], experts[0].gate_proj.weight, select_topk=0.15))
                self.up_proj.weight=torch.nn.Parameter(sce_merge([exp.up_proj.weight for exp in experts], experts[0].up_proj.weight, select_topk=0.15))
                self.down_proj.weight=torch.nn.Parameter(sce_merge([exp.down_proj.weight for exp in experts], experts[0].down_proj.weight, select_topk=0.15))
            elif merge_method=='slerp':
                self.gate_proj.weight =torch.nn.Parameter(multislerp([exp.gate_proj.weight for exp in experts], weight=[1/len(experts)] * len(experts), base_tensor=experts[0].gate_proj.weight))
                self.up_proj.weight=torch.nn.Parameter(multislerp([exp.up_proj.weight for exp in experts], weight=[1/len(experts)] * len(experts), base_tensor=experts[0].up_proj.weight))
                self.down_proj.weight=torch.nn.Parameter(multislerp([exp.down_proj.weight for exp in experts], weight=[1/len(experts)] * len(experts), base_tensor=experts[0].down_proj.weight))
        else  :
            self.gate_proj.weight = torch.nn.Parameter(experts[0].gate_proj.weight)
            self.up_proj.weight=torch.nn.Parameter(experts[0].up_proj.weight)
            self.down_proj.weight=torch.nn.Parameter(experts[0].down_proj.weight)
            
        self.n_sub_experts=len(experts)

        print("init done")
        self.mask_up_proj = torch.nn.Linear(self.n_sub_experts, self.hidden_size, bias=False)
        
        for name, params in self.named_parameters():
            params.requires_grad=True
        
    def forward(self, x, top_k_weights):   
        x = x + self.mask_up_proj(top_k_weights)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss
    
class MultiplexedMOE(torch.nn.Module):
    def __init__(self, moe, target_routed_experts=8):
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

    def multiplex(self, affinity_matrix, group_size, train_batches, learning_rate, device, merge_method='sce'):
        inv_mapping_dict = group_items_by_affinity(affinity_matrix, group_size)

        # Convert the dictionary to a tensor for saving in state_dict
        inv_mapping_tensor = torch.tensor(inv_mapping_dict, dtype=torch.int64)
        self.register_buffer('inv_mapping_dict', inv_mapping_tensor, persistent=True)

        multiplexed_experts = []

        for k, v in enumerate(inv_mapping_dict):
            merged_moe = MultiplexedMLP(self.config).to("cuda")
            merged_moe.load_experts([dequantize_GEMM(self.experts[i], destruct=False, return_params=False) for i in v], merge_method=merge_method)
            merged_moe = merged_moe.to(device, dtype=torch.bfloat16)
            multiplexed_experts.append(merged_moe)

        self.multiplexed_experts = torch.nn.ModuleList(multiplexed_experts).to(device, dtype=torch.bfloat16)

        self.optimizer = AdEMAMix(
            filter(lambda p: p.requires_grad, self.multiplexed_experts.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999, 0.9999),
            alpha=5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_batches,
            eta_min=learning_rate * 0.8
        )
        

        # self.optimizers = [AdEMAMix(
        #     filter(lambda p: p.requires_grad, self.multiplexed_experts[idx].parameters()),
        #     lr=learning_rate,
        #     betas=(0.9, 0.999, 0.9999),
        #     alpha=5
        # ) for idx in range(len(inv_mapping_dict))]

        # self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizers[idx],
        #     T_max=train_batches,
        #     eta_min=learning_rate * 0.1
        # ) for idx in range(len(inv_mapping_dict))]

        self.criterion = torch.nn.functional.smooth_l1_loss

    def train_step(self, hidden_states, layer_norm, temperature=5):
        residual = hidden_states
        hidden_states = layer_norm(hidden_states)
    
        pred = self.forward(hidden_states)
        true = self.forward_origin(hidden_states)
        
        T = temperature
        loss = self.criterion(T * pred, T * true, reduction='mean')
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss = self.forward_gate(hidden_states)

        # losses = 0
        # for idx in range(self.inv_mapping_dict.size(0)):
        #     pred = self.forward_multiplexed_expert(idx, hidden_states, topk_idx, topk_weight)
        #     true = self.forward_non_multiplexed_expert(idx, hidden_states, topk_idx, topk_weight)

        #     T = temperature

        #     loss = self.criterion(T * pred, T * true, reduction='mean')
        #     loss.backward()
        #     self.optimizers[idx].step()
        #     self.schedulers[idx].step()
        #     self.optimizers[idx].zero_grad()
        #     losses += loss
        return loss

    def forward(self, hidden_states):
        identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss = self.forward_gate(hidden_states)

        y = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)

        for idx in range(self.inv_mapping_dict.size(0)):
            y += self.forward_multiplexed_expert(idx, hidden_states, topk_idx, topk_weight)

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

    def forward_origin(self, hidden_states):
        identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss = self.forward_gate(hidden_states)

        y = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for idx in range(self.inv_mapping_dict.size(0)):
            y += self.forward_non_multiplexed_expert(idx, hidden_states, topk_idx, topk_weight)

        y = y.view(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    def forward_multiplexed_expert(self, idx, hidden_states, topk_idx, topk_weight):
        indexes = self.inv_mapping_dict[idx].tolist()

        flat_topk_weight = torch.zeros((hidden_states.shape[0], len(indexes)), device=hidden_states.device, dtype=hidden_states.dtype)

        for i, index in enumerate(indexes):
            flat_topk_weight[:, i] = torch.sum(topk_weight * (topk_idx == index), axis=-1)

        scalar = torch.sum(flat_topk_weight, axis=-1, keepdim=True)  # keeping the total weight of the experts

        flat_topk_weight[flat_topk_weight == 0] = -1e9
        flat_topk_weight = torch.softmax(flat_topk_weight, dim=-1)
        
        output = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
        
        output[scalar.squeeze() != 0] = self.multiplexed_experts[idx](hidden_states[scalar.squeeze() != 0], flat_topk_weight[scalar.squeeze() != 0]) # Process only if at least one weight is required, should be much faster
        
        return scalar * output  # Weighting is already taken into account by how the multiplex is trained

    def forward_non_multiplexed_expert(self, idx, hidden_states, topk_idx, topk_weight):
        indexes = self.inv_mapping_dict[idx].tolist()

        flat_topk_weight = torch.zeros((hidden_states.shape[0], len(indexes)), device=hidden_states.device, dtype=hidden_states.dtype)

        for i, index in enumerate(indexes):
            flat_topk_weight[:, i] = torch.sum(topk_weight * (topk_idx == index), axis=-1)

        y = torch.sum(torch.stack([self.experts[index](hidden_states) for index in indexes], dim=-1) * flat_topk_weight[:, None], dim=-1)
        return y

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    #     inv_mapping_tensor = state_dict.pop(prefix + 'inv_mapping_dict', None)
    #     if inv_mapping_tensor is not None:
    #         inv_mapping_dict = {int(key): val.tolist() for key, val in inv_mapping_tensor}
    #         self.register_buffer('inv_mapping_dict', inv_mapping_tensor, persistent=False)
    #     else:
    #         error_msgs.append(f"Missing key '{prefix}inv_mapping_dict' in state_dict")
    #     super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)