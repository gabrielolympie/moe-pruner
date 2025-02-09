from deepseek_v3.modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3MoE, DeepseekV3ForCausalLM
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
from deepseek_v3.custom_modeling.modeling_fp8_deepseek import FP8Linear

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

def prepare_distilled_moe(
    moe,
    selected_experts,
    n_routed_experts,
    n_active_experts,
    lora_rank=8,  # Added LoRA rank
    lora_alpha=16, # Added LoRA alpha
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
    pruned_moe.shared_experts.up_proj =  moe.shared_experts.up_proj
    pruned_moe.shared_experts.down_proj =  moe.shared_experts.down_proj
    
    for param in pruned_moe.parameters():
        param.requires_grad = False

    ## Only the gate have a full finetuning
    pruned_moe.gate.weight = torch.nn.Parameter(moe.gate.weight[list(selected_experts)[:n_routed_experts]].detach().to(device))

    # Update Experts - Now with LoRA
    for i, ind in enumerate(list(selected_experts)[:n_routed_experts]):
        # Materialization.  Keep FP8 weights!
        expert = moe.experts[ind]  # Get the original expert, already in FP8
        pruned_moe.experts[i].to_empty(device=device)

        pruned_moe.experts[i].gate_proj = expert.gate_proj
        pruned_moe.experts[i].up_proj = expert.up_proj
        pruned_moe.experts[i].down_proj =  expert.down_proj

        # Create LoRA configs for each projection
        lora_config_gate = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=["weight"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        lora_config_up = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=["weight"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        lora_config_down = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=["weight"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        # print(pruned_moe.experts[i].gate_proj)
        # Apply LoRA - IMPORTANT: apply to the *FP8Linear* layers!
        pruned_moe.experts[i].gate_proj = get_peft_model(pruned_moe.experts[i].gate_proj, lora_config_gate)
        pruned_moe.experts[i].up_proj = get_peft_model(pruned_moe.experts[i].up_proj, lora_config_up)
        pruned_moe.experts[i].down_proj = get_peft_model(pruned_moe.experts[i].down_proj, lora_config_down)

    pruned_moe.to(device)

    for name,param in pruned_moe.named_parameters():
        if "lora" in name.lower():
            param.requires_grad=True
        else:
            param.requires_grad = False

    for param in pruned_moe.gate.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pruned_moe.parameters()), lr=learning_rate, weight_decay=weight_decay)  # Correct optimizer

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=total_steps
    )
    criterion = torch.nn.MSELoss().to(device)

    return {
        "suffix": f"{n_routed_experts}@{n_active_experts}",
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
        gradient_accumulation_steps=1,
        model_name="deepseek",
    ):
        self.layer = layer
        self.layer_idx = layer_idx
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model_name=model_name
        self.step_count = 0  # Counter for gradient accumulation

    def calibrate(
        self,
        calibration_batch,
        position_ids,
        mlp_params,
        use_fp8=True,
        learning_rate=1e-4,
        weight_decay=0.01,
        alpha=0.5,
        total_steps=100,
        temperature=10,
    ):
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
            destruct_module_optimized(distillat["moe"]) # Destruct after saving
            memory_cleanup()