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
