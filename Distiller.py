from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3MoE, DeepseekV3ForCausalLM
import math

from transformers import AutoConfig, AutoModelForCausalLM
from memory_utils import load_module_weights_and_freeze_optimized, memory_cleanup, destruct_module_optimized  # Assuming you have this file
from tqdm.auto import tqdm
from accelerate import init_empty_weights

import torch
from copy import deepcopy
import _pickle as pickle
from typing import Optional, Tuple
import numpy as np
import json
import torch.nn as nn
import warnings
import os
from peft import LoraConfig, get_peft_model  # Import LoRA related functions
from fp8_linear import FP8Linear

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

def load_model_config(model_name: str) -> Tuple[dict, AutoConfig]:
    """Load model configuration and weight map efficiently."""
    with open(f"{model_name}/model.safetensors.index.json", 'r') as f:
        weight_map = json.load(f)['weight_map']

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return weight_map, config

def create_empty_model(config: AutoConfig):
    with init_empty_weights():
        model = DeepseekV3ForCausalLM(
            config,
        )
    return model

def create_empty_layer(config: AutoConfig, layer_idx=5) -> AutoModelForCausalLM:
    """Create an empty model with frozen parameters."""
    with init_empty_weights():
        model = DeepseekV3DecoderLayer(
            config,
            layer_idx=layer_idx
        )

    for param in model.parameters():
        param.requires_grad = False
    return model

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Get the dtype and device from the base layer
        dtype = torch.bfloat16  # Or get from base_layer.weight.dtype if it's already a tensor
        device = base_layer.weight.device

        # Initialize LoRA matrices
        self.weight = nn.Parameter(
            torch.zeros((rank, base_layer.in_features), dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((base_layer.out_features, rank), dtype=dtype, device=device)
        )

        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        device = x.device
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        base_out = self.base_layer(x).to(device)
        lora_out = self.dropout(x) @ self.weight.t() @ self.lora_B.t() * self.scaling
        
        return base_out + lora_out.to(device)


    def merge_and_unload(self):
        """Merge LoRA weights with base weights and return new layer"""
        merged_weight = self.base_layer._weight_unquantized(torch.float32).to(self.weight.device)
        merged_weight = merged_weight + (self.lora_B @ self.weight) * self.scaling

        new_layer = FP8Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.weight.device,  # Use LoRA's device
            fp8_format=self.base_layer.fp8_format
        )
        new_layer.from_weight_matrix(merged_weight)
        if self.base_layer.bias is not None:
            new_layer.bias = nn.Parameter(self.base_layer.bias.clone().to(self.weight.device)) 
        return new_layer

def prepare_distilled_moe(
    moe,
    selected_experts,
    n_routed_experts,
    n_active_experts,
    lora_rank=8,
    lora_alpha=16,
    use_fp8=True,
    learning_rate=1e-4,
    weight_decay=0.01,
    total_steps=100,
    device="cuda:1"
):
    config = deepcopy(moe.config)

    # Create new Moe
    config.n_routed_experts = n_routed_experts
    config.num_experts_per_tok = n_active_experts

    pruned_moe = DeepseekV3MoE(config)
    pruned_moe.shared_experts.gate_proj = moe.shared_experts.gate_proj
    pruned_moe.shared_experts.up_proj = moe.shared_experts.up_proj
    pruned_moe.shared_experts.down_proj = moe.shared_experts.down_proj
    
    for param in pruned_moe.parameters():
        param.requires_grad = False

    # Only the gate has full finetuning
    pruned_moe.gate.weight = nn.Parameter(moe.gate.weight[list(selected_experts)[:n_routed_experts]].detach().to(device))

    # Update Experts with custom LoRA
    for i, ind in enumerate(list(selected_experts)[:n_routed_experts]):
        expert = moe.experts[ind]
        
        pruned_moe.experts[i].to_empty(device=device)

        # Apply LoRA to each projection
        pruned_moe.experts[i].gate_proj = LoRALinear(
            expert.gate_proj, 
            rank=lora_rank, 
            alpha=lora_alpha
        )
        pruned_moe.experts[i].up_proj = LoRALinear(
            expert.up_proj, 
            rank=lora_rank, 
            alpha=lora_alpha
        )
        pruned_moe.experts[i].down_proj = LoRALinear(
            expert.down_proj, 
            rank=lora_rank, 
            alpha=lora_alpha
        )

    pruned_moe.to(device)

    # Set requires_grad for LoRA parameters only
    for name, param in pruned_moe.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for param in pruned_moe.gate.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pruned_moe.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=total_steps
    )
    criterion = torch.nn.MSELoss().to(device)

    return {
        "suffix": f"{n_routed_experts}a{n_active_experts}",
        "moe": pruned_moe,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion
    }

class MOEDistillerV3:
    def __init__(
        self,
        layer,
        layer_idx,
        model_name="deepseek",
    ):
        self.layer = layer
        self.layer_idx = layer_idx
        self.model_name=model_name
        

    def calibrate(
        self,
        calibration_batch,
        position_ids,
        mlp_params,
        use_fp8=True,
        lora_rank=8,
        lora_alpha=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        alpha=0.5,
        total_steps=100,
        temperature=10,
    ):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step_count = 0  # Counter for gradient accumulation
        
        self.layer.mlp.gate.save_path = 'temp.pickle' #This should be a class attribute
        self.temperature=temperature
        for param in self.layer.parameters():
            param.requires_grad = False

        top_k_output = []
        for batch in tqdm(calibration_batch):
            x = self.layer.forward(
                batch.to('cuda:0', dtype=torch.bfloat16),
                position_ids=position_ids
            )[0].to('cpu')
            with open(self.layer.mlp.gate.save_path, 'rb') as f: # Use class attribute
                top_k_output.append(pickle.load(f))
        top_k_output = np.concatenate(top_k_output)

        v, c = np.unique(top_k_output, return_counts=True)
        selected_experts = np.flip(np.argsort(c))

        # Prepare distillation
        self.distillats = []
        for i, (n_routed_experts, n_active_experts) in enumerate(mlp_params):
            device="cuda:1"
            self.distillats.append(prepare_distilled_moe(
                self.layer.mlp,
                selected_experts,
                n_routed_experts,
                n_active_experts,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                total_steps=total_steps,
                use_fp8=use_fp8,
                device=device
            ))


    def step(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = hidden_states.to("cuda:0", dtype=torch.bfloat16)

        hidden_states, distillat_hidden_states = self.step_forward(hidden_states, None, position_ids)
        losses = {}

        for i, distillat in enumerate(self.distillats):
            loss = distillat["criterion"](hidden_states * self.temperature, distillat_hidden_states[i] * self.temperature)
            loss = loss / self.gradient_accumulation_steps  # Scale the loss
            loss.backward()
            losses[distillat['suffix']] = loss.item() * self.gradient_accumulation_steps

        self.step_count += 1

        if self.step_count % self.gradient_accumulation_steps == 0:
            for i, distillat in enumerate(self.distillats):
                distillat["optimizer"].step()
                distillat["scheduler"].step()
                distillat["optimizer"].zero_grad()


        hidden_states = hidden_states.to("cpu", dtype=torch.bfloat16)
        # attention_mask = attention_mask.to("cpu", dtype=torch.bfloat16)
        del distillat_hidden_states  # Explicitly delete to free memory
        return hidden_states, losses

    def step_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            
        residual = hidden_states
        hidden_states = self.layer.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        # Standard part
        hidden_states = self.layer.post_attention_layernorm(hidden_states)

        hidden_states_frozen = self.layer.mlp(hidden_states)
        hidden_states_frozen = residual + hidden_states_frozen

        distillat_hidden_states = []
        
        hidden_states_frozen = hidden_states_frozen.to('cuda:1', dtype=torch.bfloat16)
        residual = residual.to('cuda:1', dtype=torch.bfloat16)

        for i, distillat in enumerate(self.distillats):
            device="cuda:1"
            hidden_states = hidden_states.to(device, dtype=torch.bfloat16)
            distillat_hidden_state = distillat['moe'](hidden_states).to('cuda:1')
            distillat_hidden_states.append(distillat_hidden_state + residual) # Reuse residual

        return hidden_states_frozen, distillat_hidden_states

    def save_distillats(self):
        for distillat in self.distillats:
            save_directory = self.model_name + "_" + distillat['suffix']
            save_name = f"_layer_{self.layer_idx}.ckpt"
            os.makedirs(save_directory, exist_ok=True)

            # Reconstruct and merge LoRA for each expert *before* saving
            merged_experts_state_dict = {}
            for i in range(distillat["moe"].config.n_routed_experts):
                
                expert = distillat["moe"].experts[i]
                #Gate
                merged_gate = expert.gate_proj.merge_and_unload()
                merged_gate_proj = merged_gate.to_linear()
                
                #UP
                merged_up = expert.up_proj.merge_and_unload()
                merged_up_proj = merged_up.to_linear()
                
                #Down
                merged_down = expert.down_proj.merge_and_unload()
                merged_down_proj = merged_down.to_linear()
                
                merged_experts_state_dict[f"experts.{i}.gate_proj.weight"] = merged_gate_proj.weight.data.cpu()
                merged_experts_state_dict[f"experts.{i}.up_proj.weight"] = merged_up_proj.weight.data.cpu()
                merged_experts_state_dict[f"experts.{i}.down_proj.weight"] = merged_down_proj.weight.data.cpu()

                #Explicit delete
                del merged_gate
                del merged_gate_proj
                del merged_up
                del merged_up_proj
                del merged_down
                del merged_down_proj
                del expert
                memory_cleanup()
                
            torch.save(merged_experts_state_dict, os.path.join(save_directory, "experts" + save_name))
            torch.save(distillat["moe"].gate.state_dict(), os.path.join(save_directory, "gate" + save_name))
            memory_cleanup()