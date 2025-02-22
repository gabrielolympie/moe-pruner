from safetensors import safe_open
import bitsandbytes as bnb
import torch
from transformers import AutoConfig, AutoModelForCausalLM
# from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3MoE, DeepseekV3ForCausalLM
from accelerate import init_empty_weights
from deepseek_v3.configuration_deepseek import DeepseekV3Config

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gc
from typing import Dict, Optional, Tuple
import os

from fp8_linear import FP8Linear

from tqdm.auto import tqdm
import logging
import torch.nn.functional as F

# Configure logging to show only errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

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

def create_empty_layer_fp8(config, layer_idx, device="cuda:0"):
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
                    device=device,
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
  
def load_weight_cached(
    weight_name: str,
    weight_file: str,
    model_name: str,
    device: int,
) -> torch.Tensor:
    """Load weight with caching for repeated access."""
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        file_path = f"{model_name}/{weight_file}"
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            tensor_slice = f.get_slice(weight_name)
            shape = tensor_slice.get_shape()
            if len(shape) > 1:
                vocab_size, hidden_dim = shape
                return tensor_slice[:, :hidden_dim]
            else:
                return tensor_slice[:]

def rebuild_from_8bit(layer: FP8Linear, trainable: bool = True, device="cuda", dtype: torch.dtype = torch.bfloat16) -> torch.nn.Linear:
    """Rebuilds a standard torch.nn.Linear layer from an FP8Linear layer."""
    new_layer = layer.to_linear(dtype=dtype)
    new_layer.weight.requires_grad=trainable
    if new_layer.bias is not None:
        new_layer.bias.requires_grad=trainable
    return new_layer



def process_module(
    module_info: Tuple[str, torch.nn.Module],
    module_path: str,
    weight_map: dict,
    model_name: str,
    fp8_format: str = "e4m3",
    ignore: list = [],
    device:str = "cuda:0"
) -> Optional[Tuple[str, torch.nn.Module]]:
    """Process individual module with optimized weight loading."""
    name, module = module_info

    if len(list(module.children())) > 0:
        return None

    if module_path is not None:
        full_name = f"{module_path}.{name}"
    else:
        full_name = f"{name}"

    if full_name[-1]==".":
        full_name=full_name[:-1]
        
    for exception in ignore:
        if exception in full_name:
            return None
    
    dtype = torch.bfloat16

    if type(module).__name__ == "Linear":
        weight_name = f"{full_name}.weight"
        weight_file = weight_map[weight_name]
        weight = load_weight_cached(weight_name, weight_file, model_name, device)
        quant_state_name = f"{full_name}.weight_scale_inv"
        if quant_state_name in weight_map:  # Check if quant_state exists
            quant_state_file = weight_map[quant_state_name]
            quant_state = load_weight_cached(quant_state_name, quant_state_file, model_name, device)
            
            weight = weight_dequant(weight, quant_state).to(dtype)

            module = FP8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=device,
                fp8_format=fp8_format
            )
            
            module.from_weight_matrix(weight)
            if module.bias is not None:
                module.bias = torch.nn.Parameter(module.bias.data.to(torch.float32)) # bias should to float32 according to FP8 line1295

        else:  # If quant_state is not found, create a regular Linear layer
            # Create a new Linear layer and load the weights
            module.to_empty(device="cuda:0")
            with torch.no_grad():  # Disable gradient tracking during weight loading
                module.weight.copy_(weight.to(dtype))
                if module.bias is not None:
                    module.bias.copy_(module.bias.data.to(torch.float32))
            
        memory_cleanup()


    else:
        module.to_empty(device=device).to(torch.bfloat16)
        for weight_name, weights in module.named_parameters():
            try:
                weights.requires_grad = False
                full_weight_name=f"{full_name}.{weight_name}"
                weight_file = weight_map[full_weight_name]  # Corrected: Use full_name
                tensor = load_weight_cached(full_weight_name, weight_file, model_name, device) # Corrected: full_name
                weights.copy_(tensor).to(dtype=torch.bfloat16, device=device)
            except Exception as e:
                logger.error(f"Failed to load weights for {full_name}: {e}")

    memory_cleanup()
    return name, module

def load_module_weights_and_freeze_optimized(
    base_module: torch.nn.Module,
    module_path: str,
    weight_map: dict,
    model_name: str,
    max_workers: int = 4,
    fp8_format: str = "e4m3",
    ignore: list = [],
    device:str = "cuda:0"
) -> torch.nn.Module:
    """Load module weights, replace Linear with FP8Linear."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.memory.set_per_process_memory_fraction(0.95)

    def update_module_recursively(module, name, new_module):
        name_parts = name.split('.')
        current = module
        for part in name_parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return False
        if hasattr(current, name_parts[-1]):
            setattr(current, name_parts[-1], new_module)
            return True
        return False

    modules = list(base_module.named_modules())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_fn = partial(
            process_module,
            module_path=module_path,
            weight_map=weight_map,
            model_name=model_name,
            fp8_format=fp8_format,
            ignore=ignore,
            device=device
        )

        with tqdm(total=len(modules), desc="Processing modules") as pbar:
            futures = []
            for module in modules:
                future = executor.submit(process_fn, module)
                future.add_done_callback(lambda p: pbar.update(1))
                futures.append(future)

            processed_modules = []
            for future in futures:
                result = future.result()
                if result is not None:
                    processed_modules.append(result)

    with tqdm(total=len(processed_modules), desc="Updating module structure") as pbar:
        for name, new_module in processed_modules:
            update_module_recursively(base_module, name, new_module)
            pbar.update(1)

    gc.collect()
    memory_cleanup()
    return base_module

