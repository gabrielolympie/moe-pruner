# class MultiplexedMLP(torch.nn.Module):
#     def __init__(self, config, hidden_size=None, intermediate_size=None):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
#         self.intermediate_size = (
#             config.moe_intermediate_size if intermediate_size is None else intermediate_size
#         )

#         self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[config.hidden_act]
        
#         self.n_sub_experts = config.n_routed_experts // config.n_multiplexed_routed_experts
#         self.mask_up_proj = torch.nn.Linear(self.n_sub_experts, self.hidden_size, bias=False)
        
#         for name, params in self.named_parameters():
#             params.requires_grad=True
        
#     def forward(self, x, top_k_weights):   
#         x = x + self.mask_up_proj(top_k_weights)
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#         return down_proj

# class MultiplexedMOE(torch.nn.Module):
#     def __init__(self, config, target_routed_experts=8):
#         super().__init__()
#         self.config = config
#         self.num_experts_per_tok = config.num_experts_per_tok

#         if hasattr(config, "ep_size") and config.ep_size > 1:
#             assert config.ep_size == dist.get_world_size()
#             self.ep_size = config.ep_size
#             self.experts_per_rank = config.n_routed_experts // config.ep_size
#             self.ep_rank = dist.get_rank()
#             self.experts = nn.ModuleList(
#                 [
#                     (
#                         MultiplexedMLP(
#                             config, intermediate_size=config.moe_intermediate_size
#                         )
#                         if i >= self.ep_rank * self.experts_per_rank
#                         and i < (self.ep_rank + 1) * self.experts_per_rank
#                         else None
#                     )
#                     for i in range(config.n_routed_experts)
#                 ]
#             )
#         else:
#             self.ep_size = 1
#             self.experts_per_rank = config.n_routed_experts
#             self.ep_rank = 0
#             self.experts = nn.ModuleList(
#                 [
#                     MultiplexedMLP(config, intermediate_size=config.moe_intermediate_size)
#                     for i in range(config.n_routed_experts)
#                 ]
#             )
#         self.gate = MoEGate(config)
#         if config.n_shared_experts is not None:
#             intermediate_size = config.moe_intermediate_size * config.n_shared_experts
#             self.shared_experts = DeepseekV2MLP(
#                 config=config, intermediate_size=intermediate_size

#         # Register inv_mapping_dict as a buffer
#         self.register_buffer('inv_mapping_dict', torch.tensor([]), persistent=True)

#     def forward(self, hidden_states):
#         identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss = self.forward_gate(hidden_states)

#         y = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)

#         for idx in range(self.inv_mapping_dict.size(0)):
#             y += self.forward_multiplexed_expert(idx, hidden_states, topk_idx, topk_weight)

#         y = y.view(*orig_shape)
        
#         if self.config.n_shared_experts is not None:
#             y = y + self.shared_experts(identity)
#         return y

#     def forward_gate(self, hidden_states):
#         identity = hidden_states
#         orig_shape = hidden_states.shape
        
#         topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
#         hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

#         return identity, orig_shape, hidden_states, topk_idx, topk_weight, aux_loss

#     def forward_multiplexed_expert(self, idx, hidden_states, topk_idx, topk_weight):
#         indexes = self.inv_mapping_dict[idx].tolist()

#         flat_topk_weight = torch.zeros((hidden_states.shape[0], len(indexes)), device=hidden_states.device, dtype=hidden_states.dtype)

#         for i, index in enumerate(indexes):
#             flat_topk_weight[:, i] = torch.sum(topk_weight * (topk_idx == index), axis=-1)

#         scalar = torch.sum(flat_topk_weight, axis=-1, keepdim=True)  # keeping the total weight of the experts

#         flat_topk_weight[flat_topk_weight == 0] = -1e9
#         flat_topk_weight = torch.softmax(flat_topk_weight, dim=-1)
        
#         output = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
        
#         output[scalar.squeeze() != 0] = self.experts[idx](hidden_states[scalar.squeeze() != 0], flat_topk_weight[scalar.squeeze() != 0]) # Process only if at least one weight is required, should be much faster
        
#         return scalar * output  # Weighting is already taken into account by how the multiplex is trained

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
#         inv_mapping_tensor = state_dict.pop(prefix + 'inv_mapping_dict', None)
#         if inv_mapping_tensor is not None:
#             inv_mapping_dict = {int(key): val.tolist() for key, val in inv_mapping_tensor}
#             self.register_buffer('inv_mapping_dict', inv_mapping_tensor, persistent=False)
#         else:
#             error_msgs.append(f"Missing key '{prefix}inv_mapping_dict' in state_dict")
#         super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)