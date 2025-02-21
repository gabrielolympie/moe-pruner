import gc
import os
import pickle
from dataclasses import dataclass
from typing import Optional, List

import bitsandbytes as bnb
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# from Distiller import create_empty_layer_fp8, load_model_config
from liger_kernel.transformers import apply_liger_kernel_to_llama
from modeling_deepseek import _prepare_4d_causal_attention_mask

from configs import GenerationParams, PathConfig
from torch_utils import save_intermediate_state, save_midlayer_state, load_intermediate_state, load_midlayer_state, destruct_module_optimized, memory_cleanup
from memory_utils import load_model_config, create_empty_layer_fp8

def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    device = "cuda"
    
    
    params = GenerationParams()
    path_config = PathConfig()
    
    # Load Model Config and Tokenizer
    weight_map, config = load_model_config("deepseek_v3")
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3", trust_remote_code=True
    )
    apply_liger_kernel_to_llama()  # Apply LLaMA kernel optimizations

    
    position_ids = torch.arange(
        0, params.max_length, dtype=torch.long, device=device
    ).unsqueeze(0)

    # Load Calibration Dataset
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

    # Initialize Embedding Layer
    embed_tokens = torch.nn.Embedding(
        config.vocab_size, config.hidden_size, config.pad_token_id, device=device
    )
    embed_tokens.weight.requires_grad = False
    embed_tokens.load_state_dict(torch.load("layers/embed_tokens.pt"))
    embed_tokens.to(device)

    # Process Embeddings
    for batch_idx in tqdm(range(params.n_batch), desc="Processing embeddings"):
        batch = train_dataset[
            params.batch_size * batch_idx : params.batch_size * (batch_idx + 1)
        ]
        inputs = tokenizer(
            batch,
            max_length=params.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = embed_tokens(inputs["input_ids"]).to(
                "cpu", dtype=torch.bfloat16
            )
        save_intermediate_state(path_config, -1, batch_idx, embeddings)

    destruct_module_optimized(embed_tokens)
    del embed_tokens
    memory_cleanup()

    writer = SummaryWriter(path_config.log_dir)

    # Process Each Layer
    for layer_idx in range(61):
        memory_cleanup()
        print(f"Loading layer {layer_idx}")
        layer = create_empty_layer_fp8(config, layer_idx=layer_idx)
        layer.load_state_dict(
            torch.load(f"layers/layer_{layer_idx}.pt", map_location=device), assign=True
        )
        print("Layer loaded")

        if "DeepseekV3MLP" in str(layer.mlp.__class__):
            # Standard MLP Layer
            for batch_idx in tqdm(
                range(params.n_batch), desc=f"Processing MLP Layer {layer_idx}"
            ):
                prev_state = load_intermediate_state(path_config, layer_idx - 1, batch_idx)
                
                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    new_state = layer.forward(
                        hidden_states=prev_state.to(device), position_ids=position_ids
                    )[0].detach()
                save_intermediate_state(path_config, layer_idx, batch_idx, new_state)

                if batch_idx % 100 == 0:
                    memory_cleanup()
        else:
            # MoE Layer
            temp_path = "temp_activation.pickle"
            layer.mlp.gate.save_path = temp_path
            top_k_output = []

            for batch_idx in tqdm(
                range(params.n_batch), desc=f"Generating Layer {layer_idx}"
            ):
                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    hidden_states = load_intermediate_state(
                        path_config, layer_idx - 1, batch_idx
                    )
                    
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)

                    hidden_states, _, _ = layer.self_attn(
                        hidden_states=hidden_states,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )

                    hidden_states = residual + hidden_states
                    residual = hidden_states
                    hidden_states = layer.post_attention_layernorm(hidden_states)

                    save_midlayer_state(path_config, layer_idx, batch_idx, hidden_states)

                    hidden_states = layer.mlp(hidden_states)
                    hidden_states = residual + hidden_states

                    save_intermediate_state(
                        path_config, layer_idx, batch_idx, hidden_states
                    )

                with open(temp_path, "rb") as f:
                    top_k_output.append(pickle.load(f))

            top_k_output = np.concatenate(top_k_output)
            with open(path_config.get_expert_activation_path(layer_idx), "wb") as f:
                pickle.dump(top_k_output, f)

        destruct_module_optimized(layer)
        del layer
        memory_cleanup()

    writer.close()