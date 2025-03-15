from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from tqdm.auto import tqdm
import numpy as np
import argparse
import pickle
import torch
import json
import os

## Custom Imports
from utils.config_utils import GenerationParams, PathConfig
from utils.torch_utils import (
    save_quant,
    load_quant,
    destruct_module_optimized,
    memory_cleanup,
    get_nonreasoning_dataset,
    load_weight,
    rsetattr,
    rgetattr,
    load_weights,
    rhasattr,
    convert_meta_model_to_awq
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

## python 2_synthetic_data_generation.py --device cuda:0 --model_name deepseek_v3_awq --n_batch 600 --local_batch_size 300 --local_batch 0 --batch_size 4 --max_length 512
## python 2_synthetic_data_generation.py --device cuda:1 --model_name deepseek_v3_awq --n_batch 600 --local_batch_size 300 --local_batch 1 --batch_size 4 --max_length 512

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Script")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--model_name", type=str, default="deepseek_v3", help="Name of the model")
    parser.add_argument("--n_batch", type=int, default=16, help="Number of batches")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    parser.add_argument("--local_batch_size", type=int, default=512, help="Device to use")
    parser.add_argument("--local_batch", type=int, default=512, help="Device to use")
    
    args = parser.parse_args()
    device=args.device
    n_batch=args.n_batch
    batch_size=args.batch_size
    max_length=args.max_length
    model_name=args.model_name
    local_batch_size=args.local_batch_size
    local_batch=args.local_batch
    
    
    dtype=torch.float16
    
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
    torch.backends.cudnn.allow_tf32 = True        # Allow TF32 on cudnn
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    
    generation_config = GenerationParams(
        n_batch=n_batch,
        batch_size=batch_size,
        max_length=max_length
    )

    path_config = PathConfig(
        model_name = model_name,
        intermediate_states = "data/intermediate_states",
        expert_states = "data/expert_states",
        expert_activations = "data/expert_activations",
    )
    
    local_batch_size=args.local_batch_size
    local_batch=args.local_batch
    
    start = local_batch_size * local_batch
    end = local_batch_size * ( local_batch + 1 )
    
    position_ids = torch.arange(0, generation_config.max_length, dtype=torch.long, device=device).unsqueeze(0)

    tokenizer=AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    train_dataset=get_nonreasoning_dataset(tokenizer, generation_config)

    with open(f"{model_name}/model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]
        
    config=AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    with init_empty_weights():
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     trust_remote_code=True,
        #     torch_dtype=dtype,
        #     attn_implementation="flash_attention_2",
        #     low_cpu_mem_usage=True
        # )
        
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            # torch_dtype=dtype,
            # attn_implementation="flash_attention_2",
            # low_cpu_mem_usage=True
        )
        
    model=convert_meta_model_to_awq(model, config, device)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False


    model.train()

    destruct_module_optimized(model)
    memory_cleanup()
    
    target_modules=[
        "model.embed_tokens.weight"
    ]

    model=load_weights(model, model_name, weight_map, target_modules, device)
    # model.model.embed_tokens=torch.compile(model.model.embed_tokens)
    
    for batch_idx in tqdm(range(start, end), desc="Processing embeddings"):
        batch = train_dataset[generation_config.batch_size * batch_idx : generation_config.batch_size * (batch_idx + 1)]
        inputs = tokenizer(
            batch,
            max_length=generation_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        hidden_states = model.model.embed_tokens(inputs["input_ids"]).to(dtype=dtype)

        os.makedirs(os.path.join(path_config.intermediate_states, f"layer_{-1}"), exist_ok=True)
        save_quant(hidden_states, os.path.join(path_config.intermediate_states, f"layer_{-1}", f"batch_{batch_idx}"))

    destruct_module_optimized(model)
    memory_cleanup()
    
    for layer_idx in range(len(model.model.layers)):
        model.eval()
        # model.model.layers[layer_idx].to_empty(device=device)
        
        target_modules=[f".layers.{layer_idx}."]
        
        model=load_weights(model, model_name, weight_map, target_modules, device)
        # model.model.layers[layer_idx]=torch.compile(model.model.layers[layer_idx])
        if rhasattr(model.model.layers[layer_idx], "mlp.gate"):
            
            top_k_output = []
            top_k_weight = []
            
            for batch_idx in tqdm(range(start, end), desc=f"Processing MLP Layer {layer_idx}"):
                hidden_states=load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx-1}", f"batch_{batch_idx}")).to(device, dtype=dtype)
                

                residual = hidden_states
                
                hidden_states = model.model.layers[layer_idx].input_layernorm(hidden_states)
                
                hidden_states, self_attn_weights, present_key_value = model.model.layers[layer_idx].self_attn(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )

                hidden_states = residual + hidden_states
                residual = hidden_states
                
                os.makedirs(os.path.join(path_config.expert_states, f"layer_{layer_idx}"), exist_ok=True)
                save_quant(hidden_states.to(device, dtype=dtype), os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}"))

                hidden_states = model.model.layers[layer_idx].post_attention_layernorm(hidden_states)

                ## For activations
                gate_output = model.model.layers[layer_idx].mlp.gate(hidden_states)
                if len(gate_output) == 3:
                    topk_idx, topk_weight, aux_loss = gate_output
                else:
                    topk_idx, topk_weight = gate_output
  
                top_k_output.append(topk_idx)
                top_k_weight.append(topk_weight)

                
                hidden_states = model.model.layers[layer_idx].mlp(hidden_states)
                hidden_states = residual + hidden_states

                os.makedirs(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}"), exist_ok=True)
                save_quant(hidden_states.to(device, dtype=dtype), os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}"))

            top_k_output=torch.cat(top_k_output, dim=0)
            top_k_weight=torch.cat(top_k_weight, dim=0)
        
            os.makedirs(os.path.join(path_config.expert_activations), exist_ok=True)
            with open(os.path.join(path_config.expert_activations, f"layer_{layer_idx}.pickle"), "wb") as f:
                pickle.dump((top_k_output, top_k_weight), f)
                
        else:
            for batch_idx in tqdm(range(start, end), desc=f"Processing MLP Layer {layer_idx}"):
                
                hidden_states=load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx-1}", f"batch_{batch_idx}")).to(device, dtype=dtype)
                
                hidden_states=model.model.layers[layer_idx](
                    hidden_states,
                    position_ids=position_ids
                )[0]

                os.makedirs(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}"), exist_ok=True)
                save_quant(hidden_states.to(device, dtype=dtype), os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}"))
                
        destruct_module_optimized(model)
        memory_cleanup()