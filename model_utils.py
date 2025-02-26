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
from torch_utils import (
    save_intermediate_state,
    save_midlayer_state,
    load_intermediate_state,
    load_midlayer_state,
    destruct_module_optimized,
    memory_cleanup,
)
from modeling_deepseek import (
    DeepseekV3DecoderLayer,
    DeepseekV3MoE,
    DeepseekV3ForCausalLM,
)
import torch
import json
from accelerate import init_empty_weights
import functools
from safetensors import safe_open

import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Semaphore


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def load_model_config(model_name: str):
    """Load model configuration and weight map efficiently."""
    with open(f"{model_name}/model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]

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
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        file_path = f"{weights_location}/{weight_file}"
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            tensor_slice = f.get_slice(weight_name)
            shape = tensor_slice.get_shape()
            if len(shape) > 1:
                vocab_size, hidden_dim = shape
                # Add non_blocking=True
                tensor = torch.nn.Parameter(tensor_slice[:, :hidden_dim].to(device), requires_grad=False).to(device)
            else:
                tensor = torch.nn.Parameter(tensor_slice[:].to(device), requires_grad=False).to(device)
            return tensor


def map_device(weight_name, layer_idx=-1, device="meta"):
    if layer_idx == -1:
        if not ("layers" in weight_name):
            return device
    else:
        if f"layers.{layer_idx}." in weight_name:
            return device
    return "meta"


def assign_device(layer_idx, n_devices):
    return f"cuda:{layer_idx%n_devices}"


def get_dataset():
    calibration = load_dataset("cognitivecomputations/dolphin-r1", "nonreasoning", cache_dir="../dolphin-r1")["train"]

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
    device_map = {}
    for k in weight_map:
        d = map_device(k, layer_idx=layer_idx, device=device)

        if d != "meta":
            device_map[k] = d
    return device_map
