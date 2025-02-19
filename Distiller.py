import os
import warnings
import math
import _pickle as pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model  # Import LoRA related functions
from fp8_linear import FP8Linear
from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3MoE, DeepseekV3ForCausalLM
from memory_utils import load_module_weights_and_freeze_optimized, memory_cleanup, destruct_module_optimized


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

def create_empty_layer_fp8(config, layer_idx):
    layer = create_empty_layer(config, layer_idx)
    
    # Helper function to recursively replace modules
    def replace_linear_modules(module, path=''):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            # If this is a Linear module, replace it
            if isinstance(child, torch.nn.Linear):
                new_module = FP8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device="cuda:0",
                    fp8_format="e4m3",
                    init_empty=True
                )
                # Use proper nested setattr
                parent = module
                if '.' in name:
                    *parent_path, name = name.split('.')
                    for part in parent_path:
                        parent = getattr(parent, part)
                setattr(parent, name, new_module)
            else:
                # Recursively process nested modules
                replace_linear_modules(child, child_path)
    # Start the recursive replacement
    replace_linear_modules(layer)
    return layer
    


class AdapterBase(nn.Module):
    """Base class for LoRA and DoRA adapters"""
    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)
        
        # Get the dtype and device from the base layer
        self.dtype = torch.bfloat16
        self.device = base_layer.weight.device
        
    def reset_parameters(self):
        raise NotImplementedError
        
    def forward(self, x):
        raise NotImplementedError
        
    def merge_and_unload(self):
        raise NotImplementedError

class LoRALinear(AdapterBase):
    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.05):
        super().__init__(base_layer, rank, alpha, dropout)
        
        # Initialize LoRA matrices with proper scaling
        self.lora_A = nn.Parameter(
            torch.empty((rank, base_layer.in_features), dtype=self.dtype, device=self.device)
        )
        self.lora_B = nn.Parameter(
            torch.empty((base_layer.out_features, rank), dtype=self.dtype, device=self.device)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize using scaled kaiming initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Ensure dtype consistency
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
            
        # Base layer output
        base_out = self.base_layer(x)
        
        # LoRA path with improved numerical stability
        dropout_x = self.dropout(x)
        lora_out = (dropout_x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        
        return base_out + lora_out.to(base_out.device)
    
    def merge_and_unload(self):
        """Merge LoRA weights with base weights and return new layer"""
        merged_weight = self.base_layer._weight_unquantized(torch.float32).to(self.device)
        merged_weight = merged_weight + (self.lora_B @ self.lora_A) * self.scaling
        
        new_layer = type(self.base_layer)(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.device,
            fp8_format=getattr(self.base_layer, 'fp8_format', None)
        )
        
        if hasattr(new_layer, 'from_weight_matrix'):
            new_layer.from_weight_matrix(merged_weight)
        else:
            new_layer.weight = nn.Parameter(merged_weight)
            
        if self.base_layer.bias is not None:
            new_layer.bias = nn.Parameter(self.base_layer.bias.clone())
            
        return new_layer

class DoRALinear(AdapterBase):
    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.05, dora_simple=True):
        super().__init__(base_layer, rank, alpha, dropout)
        
        self.dora_simple = dora_simple
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(
            torch.empty((rank, base_layer.in_features), dtype=self.dtype, device=self.device)
        )
        self.lora_B = nn.Parameter(
            torch.empty((base_layer.out_features, rank), dtype=self.dtype, device=self.device)
        )
        
        # Initialize magnitude decomposition
        self.weight_m = nn.Parameter(
            torch.empty((base_layer.out_features, 1), dtype=self.dtype, device=self.device)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Initialize magnitude component from base weights
        with torch.no_grad():
            base_norm = torch.linalg.norm(self.base_layer._weight_unquantized, dim=1, keepdim=True)
            self.weight_m.data.copy_(base_norm)
    
    def forward(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        
        # Get base weight and its norm
        base_weight = self.base_layer._weight_unquantized
        
        # Calculate new weight with LoRA
        new_weight = base_weight + (self.lora_B @ self.lora_A) * self.scaling
        
        # Calculate norm scales
        if self.dora_simple:
            base_norm = torch.linalg.norm(new_weight, dim=1, keepdim=True).detach()
        else:
            base_norm = torch.linalg.norm(new_weight, dim=1, keepdim=True)
            
        norm_scale = self.weight_m / base_norm
        
        # Apply dropout
        dropout_x = self.dropout(x)
        
        # Compute output with decomposed scaling
        base_out = F.linear(dropout_x, base_weight)
        lora_out = F.linear(dropout_x, self.lora_B @ self.lora_A) * self.scaling
        
        result = (base_out + lora_out) * norm_scale.t()
        
        if self.base_layer.bias is not None:
            result += self.base_layer.bias
            
        return result
    
    def merge_and_unload(self):
        """Merge DoRA weights with base weights and return new layer"""
        new_weight = self.base_layer._weight_unquantized(torch.float32).to(self.device)
        new_weight = new_weight + (self.lora_B @ self.lora_A) * self.scaling
        
        # Apply magnitude scaling
        base_norm = torch.linalg.norm(new_weight, dim=1, keepdim=True)
        norm_scale = (self.weight_m / base_norm).t()
        merged_weight = new_weight * norm_scale
        
        new_layer = type(self.base_layer)(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.device,
            fp8_format=getattr(self.base_layer, 'fp8_format', None)
        )
        
        if hasattr(new_layer, 'from_weight_matrix'):
            new_layer.from_weight_matrix(merged_weight)
        else:
            new_layer.weight = nn.Parameter(merged_weight)
            
        if self.base_layer.bias is not None:
            new_layer.bias = nn.Parameter(self.base_layer.bias.clone())
            
        return new_layer

@dataclass
class DistillationConfig:
    """Configuration for MOE distillation"""
    adapter_type: str = "lora"  # "lora" or "dora"
    adapter_rank: int = 8
    adapter_alpha: int = 16
    adapter_dropout: float = 0.05
    dora_simple: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    total_steps: int = 100
    temperature: float = 1.0
    gradient_accumulation_steps: int = 1

def prepare_distilled_moe(
    moe,
    selected_experts,
    n_routed_experts,
    n_active_experts,
    config: DistillationConfig,
    device="cuda:1"
):
    """Prepare distilled MOE with specified adapter type"""
    
    moe_config = deepcopy(moe.config)
    
    # Select adapter class based on type
    AdapterClass = DoRALinear if config.adapter_type.lower() == "dora" else LoRALinear
    
    # Create new MoE
    moe_config.n_routed_experts = n_routed_experts
    moe_config.num_experts_per_tok = n_active_experts
    
    pruned_moe = DeepseekV3MoE(moe_config)
    pruned_moe.shared_experts.gate_proj = moe.shared_experts.gate_proj
    pruned_moe.shared_experts.up_proj = moe.shared_experts.up_proj
    pruned_moe.shared_experts.down_proj = moe.shared_experts.down_proj
    
    for param in pruned_moe.parameters():
        param.requires_grad = False
        
    # Only the gate has full finetuning
    pruned_moe.gate.weight = nn.Parameter(moe.gate.weight[list(selected_experts)[:n_routed_experts]].detach().to(device))
    
    # Update Experts with chosen adapter
    for i, ind in enumerate(list(selected_experts)[:n_routed_experts]):
        expert = moe.experts[ind]
        
        pruned_moe.experts[i].to_empty(device=device)
        
        adapter_kwargs = {
            "rank": config.adapter_rank,
            "alpha": config.adapter_alpha,
            "dropout": config.adapter_dropout
        }
        
        if config.adapter_type.lower() == "dora":
            adapter_kwargs["dora_simple"] = config.dora_simple
            
        # Apply adapter to each projection
        pruned_moe.experts[i].gate_proj = AdapterClass(expert.gate_proj, **adapter_kwargs)
        pruned_moe.experts[i].up_proj = AdapterClass(expert.up_proj, **adapter_kwargs)
        pruned_moe.experts[i].down_proj = AdapterClass(expert.down_proj, **adapter_kwargs)
    
    pruned_moe.to(device)
    
    # Set requires_grad for adapter parameters only
    for name, param in pruned_moe.named_parameters():
        if any(x in name for x in ["lora_", "weight_m"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    for param in pruned_moe.gate.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pruned_moe.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=config.total_steps
    )
    
    criterion = torch.nn.MSELoss().to(device)
    
    return {
        "suffix": f"{config.adapter_type}_{n_routed_experts}a{n_active_experts}",
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
        self.model_name = model_name
        self.distillats = []
        
        # These will be set during calibration
        self.config = None
        self.temperature = None
        self.gradient_accumulation_steps = None
        self.step_count = 0

    def calibrate(
        self,
        calibration_batch,
        position_ids,
        mlp_params,
        config: DistillationConfig
    ):
        # Store the config for use in other methods
        self.config = config
        self.temperature = config.temperature
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.step_count = 0
        
        # Create temporary file path for gate outputs
        temp_path = f'temp_gate_layer_{self.layer_idx}.pickle'
        self.layer.mlp.gate.save_path = temp_path
        
        # Disable gradients for base layer
        for param in self.layer.parameters():
            param.requires_grad = False

        # Collect expert selection statistics
        top_k_output = []
        for batch in tqdm(calibration_batch):
            x = self.layer.forward(
                batch.to('cuda:0', dtype=torch.bfloat16),
                position_ids=position_ids
            )[0].to('cpu')
            with open(temp_path, 'rb') as f:
                top_k_output.append(pickle.load(f))
        top_k_output = np.concatenate(top_k_output)

        # Calculate expert usage frequencies
        v, c = np.unique(top_k_output, return_counts=True)
        selected_experts = np.flip(np.argsort(c))

        # Prepare distillation models
        self.distillats = []
        for n_routed_experts, n_active_experts in mlp_params:
            self.distillats.append(prepare_distilled_moe(
                self.layer.mlp,
                selected_experts,
                n_routed_experts,
                n_active_experts,
                self.config,
                device="cuda:1"
            ))
            
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def step(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if not self.config:
            raise RuntimeError("Must call calibrate() before step()")
            
        hidden_states = hidden_states.to("cuda:0", dtype=torch.bfloat16)

        hidden_states, distillat_hidden_states = self.step_forward(
            hidden_states, 
            attention_mask, 
            position_ids
        )
        losses = {}

        # Calculate and accumulate losses
        for i, distillat in enumerate(self.distillats):
            loss = distillat["criterion"](
                hidden_states * self.temperature, 
                distillat_hidden_states[i] * self.temperature
            )
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            # Store unscaled loss for reporting
            losses[distillat['suffix']] = loss.item() * self.gradient_accumulation_steps

        self.step_count += 1

        # Update weights after accumulating gradients
        if self.step_count % self.gradient_accumulation_steps == 0:
            for distillat in self.distillats:
                distillat["optimizer"].step()
                distillat["scheduler"].step()
                distillat["optimizer"].zero_grad()

        # Move tensors to CPU and free memory
        hidden_states = hidden_states.to("cpu", dtype=torch.bfloat16)
        del distillat_hidden_states
        memory_cleanup()
        
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
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please use `attention_mask` instead."
            )
            
        # Input normalization and residual connection
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

        # Process through MLP layers
        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)

        # Get teacher (frozen) output
        hidden_states_frozen = self.layer.mlp(hidden_states)
        hidden_states_frozen = residual + hidden_states_frozen

        # Move tensors to distillation device
        hidden_states_frozen = hidden_states_frozen.to('cuda:1', dtype=torch.bfloat16)
        residual = residual.to('cuda:1', dtype=torch.bfloat16)

        # Get student outputs
        distillat_hidden_states = []
        for distillat in self.distillats:
            hidden_states = hidden_states.to('cuda:1', dtype=torch.bfloat16)
            distillat_hidden_state = distillat['moe'](hidden_states)
            distillat_hidden_states.append(distillat_hidden_state + residual)

        return hidden_states_frozen, distillat_hidden_states

    def save_distillats(self):
        for distillat in self.distillats:
            save_directory = f"{self.model_name}_{distillat['suffix']}"
            save_name = f"_layer_{self.layer_idx}.ckpt"
            os.makedirs(save_directory, exist_ok=True)

            # Merge and save expert weights
            merged_experts_state_dict = {}
            for i in range(distillat["moe"].config.n_routed_experts):
                expert = distillat["moe"].experts[i]
                
                # Merge and convert projections
                merged_gate = expert.gate_proj.merge_and_unload()
                merged_up = expert.up_proj.merge_and_unload()
                merged_down = expert.down_proj.merge_and_unload()
                
                gate_proj = merged_gate.to_linear()
                up_proj = merged_up.to_linear()
                down_proj = merged_down.to_linear()
                
                # Store weights
                merged_experts_state_dict[f"experts.{i}.gate_proj.weight"] = gate_proj.weight.data.cpu()
                merged_experts_state_dict[f"experts.{i}.up_proj.weight"] = up_proj.weight.data.cpu()
                merged_experts_state_dict[f"experts.{i}.down_proj.weight"] = down_proj.weight.data.cpu()

                # Clean up merged layers
                del merged_gate, merged_up, merged_down
                del gate_proj, up_proj, down_proj
                del expert
                memory_cleanup()
                
            # Save merged experts and gate weights
            torch.save(merged_experts_state_dict, os.path.join(save_directory, f"experts{save_name}"))
            torch.save(distillat["moe"].gate.state_dict(), os.path.join(save_directory, f"gate{save_name}"))
            memory_cleanup()