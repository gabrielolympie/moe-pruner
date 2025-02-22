import gc
import os
import pickle

import numpy as np
import torch

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig
from liger_kernel.transformers import apply_liger_kernel_to_llama
from configs import GenerationParams, PathConfig
from torch_utils import save_intermediate_state, save_midlayer_state, load_intermediate_state, load_midlayer_state, destruct_module_optimized, memory_cleanup
from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3MoE, DeepseekV3ForCausalLM
import torch
import json
from accelerate import init_empty_weights
import functools
from safetensors import safe_open

import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Semaphore

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
    


def load_model_config(model_name: str):
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

def load_weight(
    weights_location: str,
    weight_name: str,
    weight_map: dict,
    device: int,
) -> torch.Tensor:
    """Load weight with non-blocking CUDA operations."""
    weight_file = weight_map[weight_name]
    # Use non_blocking=True for CUDA operations
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        file_path = f"{weights_location}/{weight_file}"
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            tensor_slice = f.get_slice(weight_name)
            shape = tensor_slice.get_shape()
            if len(shape) > 1:
                vocab_size, hidden_dim = shape
                # Add non_blocking=True
                tensor = torch.nn.Parameter(tensor_slice[:, :hidden_dim].cuda(device), requires_grad=False)
            else:
                tensor = torch.nn.Parameter(tensor_slice[:].cuda(device), requires_grad=False)
            return tensor

def map_device(weight_name, layer_idx=-1, device="cuda:0"):
    if layer_idx == -1:
        if not("layers" in weight_name):
            return device
    else:
        if f"layers.{layer_idx}." in weight_name:
            return device
    return "meta"

def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
def assign_device(layer_idx, n_devices):
    return f"cuda:{layer_idx%n_devices}"

def get_dataset(): 
    calibration = load_dataset(
        "cognitivecomputations/dolphin-r1", "nonreasoning", cache_dir="../dolphin-r1"
    )["train"]

    def filter_function(example):
        if example["overall_quality"] is not None and example["overall_quality"] == 5:
            return True
        if example["score"] is not None and example["score"] >= 0.2:
            return True
        return False

    calibration = calibration.filter(filter_function)
    
    data = calibration["messages"][: params.batch_size * params.n_batch]
    
    train_dataset = [
        tokenizer.apply_chat_template(elt, tokenize=False, add_generation_prompt=False)
        for elt in tqdm(data, desc="Preparing dataset")
    ]
    return train_dataset

def get_device_map(layer_idx, weight_map, device):
    device_map={}
    for k in weight_map:
        d=map_device(k, layer_idx=layer_idx, device=device)
        
        if d != "meta":
            device_map[k] = d
    return device_map

if __name__ == "__main__":
    apply_liger_kernel_to_llama()
    torch.backends.cuda.max_split_size_mb = 512
    torch.set_float32_matmul_precision('medium')
    
    
    device = "cuda:0"
    weights_location='deepseek_v3/'
    
    ## Configs
    params = GenerationParams()
    path_config = PathConfig()
    
    # Load Model Config and Tokenizer
    weight_map, config = load_model_config(weights_location)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3", trust_remote_code=True
    )
    
    position_ids = torch.arange(
        0, params.max_length, dtype=torch.long, device=device
    ).unsqueeze(0)

    # Create empty model
    with init_empty_weights():
        model = DeepseekV3ForCausalLM(config)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        
    destruct_module_optimized(model)
    memory_cleanup()

    train_dataset = get_dataset()

    ## Load embedding_layer
    device_map = get_device_map(-1, weight_map, device)
    
    for i, weigh_name in enumerate(tqdm(device_map)):
        rsetattr(model, weigh_name, load_weight(weights_location, weigh_name, weight_map, device))
        if i%100 ==0:
            memory_cleanup()

    for batch_idx in tqdm(range(params.n_batch), desc="Processing embeddings"):
        batch = train_dataset[params.batch_size * batch_idx : params.batch_size * (batch_idx + 1)]
        
        inputs = tokenizer(
            batch,
            max_length=params.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = model.model.embed_tokens(inputs["input_ids"]).to(
                "cpu", dtype=torch.bfloat16
            )
            
        save_intermediate_state(path_config, -1, batch_idx, embeddings)

    destruct_module_optimized(model)
    memory_cleanup()

    # Process Each Layer
    for layer_idx in range(61):
        device_map = get_device_map(layer_idx, weight_map, device)
        
        model.model.layers[layer_idx].to_empty(device=device)
        
        for i, weigh_name in enumerate(tqdm(device_map)):
            rsetattr(model, weigh_name, load_weight(weights_location, weigh_name, weight_map, device))
            if i%100 ==0:
                memory_cleanup()
                
        if "DeepseekV3MLP" in str(model.model.layers[layer_idx].mlp.__class__):
            # Standard MLP Layer
            for batch_idx in tqdm(range(params.n_batch), desc=f"Processing MLP Layer {layer_idx}"):
                
                prev_state = load_intermediate_state(path_config, layer_idx - 1, batch_idx, batch_size=params.batch_size)
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    new_state = model.model.layers[layer_idx].forward(hidden_states=prev_state.to(device), position_ids=position_ids)[0].detach()
                    
                save_intermediate_state(path_config, layer_idx, batch_idx, new_state)

                if batch_idx % 100 == 0:
                    memory_cleanup()
        else:
            # MoE Layer
            temp_path = "temp_activation.pickle"
            model.model.layers[layer_idx].mlp.gate.save_path = temp_path
            top_k_output = []

            for batch_idx in tqdm(range(params.n_batch), desc=f"Generating Layer {layer_idx}"):
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden_states = load_intermediate_state(path_config, layer_idx - 1, batch_idx, batch_size=params.batch_size)
                    
                    residual = hidden_states
                    hidden_states = model.model.layers[layer_idx].input_layernorm(hidden_states)

                    hidden_states, _, _ = model.model.layers[layer_idx].self_attn(
                        hidden_states=hidden_states,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )

                    hidden_states = residual + hidden_states
                    residual = hidden_states
                    hidden_states = model.model.layers[layer_idx].post_attention_layernorm(hidden_states)

                    save_midlayer_state(path_config, layer_idx, batch_idx, hidden_states)

                    hidden_states = model.model.layers[layer_idx].mlp(hidden_states)
                    hidden_states = residual + hidden_states

                    save_intermediate_state(
                        path_config, layer_idx, batch_idx, hidden_states
                    )

                with open(temp_path, "rb") as f:
                    top_k_output.append(pickle.load(f))

            top_k_output = np.concatenate(top_k_output)
            
            with open(path_config.get_expert_activation_path(layer_idx), "wb") as f:
                pickle.dump(top_k_output, f)

        destruct_module_optimized(model)
        memory_cleanup()

    writer.close()