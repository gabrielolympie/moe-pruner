from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from awq.modules.linear.gemm import WQLinear_GEMM
from torch.utils.tensorboard import SummaryWriter
from accelerate import init_empty_weights
from tqdm.auto import tqdm
from copy import deepcopy
import shutil
import numpy as np
import argparse
import pickle
import torch
import json
import os

from utils.ademamix import AdEMAMix
from utils.config_utils import GenerationParams, PathConfig, DistillationParams
from utils.experts_merge_utils import dequantize_GEMM
from utils.torch_utils import (
    destruct_module_optimized,
    memory_cleanup,
    rsetattr,
    load_weights,
    rhasattr,
)


# python 4_unhealed_aggregation.py --pruning_method progressive --device cuda:0 --model_name deepseek_coder_v2_lite_instruct_awq --start_layer 1 --calibrate_merge 1 --n_epochs 1 --end_layer 27 --target_routed_expert 8 --target_active_expert 4
# python 4_unhealed_aggregation.py --pruning_method progressive --device cuda:1 --model_name deepseek_coder_v2_lite_instruct_awq --start_layer 1 --calibrate_merge 1 --n_epochs 1 --end_layer 27 --target_routed_expert 16 --target_active_expert 6

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Two-layer distillation script.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--model_name", type=str, default="deepseek_v2_lite_awq", help="Name of the model.")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--start_layer", type=int, default=1, help="Starting layer.")
    parser.add_argument("--end_layer", type=int, default=27, help="Ending layer.")
    parser.add_argument("--target_routed_expert", type=int, default=8, help="Target routed expert.")
    parser.add_argument("--target_active_expert", type=int, default=2, help="Target active expert.")
    parser.add_argument("--dora_rank", type=int, default=16, help="Target active expert.")
    parser.add_argument("--pruning_method", type=str, default="topk", help="Target active expert.")
    parser.add_argument("--calibrate_merge", type=int, default=1, help="Target active expert.")
    
    args = parser.parse_args()
    
    base_model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    
    device = args.device
    model_name = args.model_name
    n_epochs = args.n_epochs
    start_layer = args.start_layer
    end_layer = args.end_layer
    target_routed_expert = args.target_routed_expert
    target_active_expert = args.target_active_expert
    dora_rank = args.dora_rank
    calibrate_merge= args.calibrate_merge == 1
    pruning_method= args.pruning_method
    

    

    path_config = PathConfig(
        model_name = model_name,
        intermediate_states = "data/intermediate_states",
        expert_states = "data/expert_states",
        expert_activations = "data/expert_activations",
        distillation_logs = "distillation_logs",
        moe_states="moe_states"
    )

    distillation_config = DistillationParams(
        n_epochs= n_epochs,
        target_routed_expert = target_routed_expert,
        target_active_expert = target_active_expert,
        eval_batches=16,
        gradient_accumulation_steps= 4,
        learning_rate= 3e-4,
        end_factor= 0.2,
        calibrate_merge=calibrate_merge,
        skip_first_tokens=0, ## useful to avoid tuning on early tokens that have less informations
        pruning_method=pruning_method, # topk , act_cl, state_cl
        dora_rank=dora_rank,
    )
    
    print('Loading model')
    with open(f"{model_name}/model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]

    config=AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )

    config.n_routed_experts=distillation_config.target_routed_expert
    config.num_experts_per_tok=distillation_config.target_active_expert

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False


    model.train()
    destruct_module_optimized(model)
    memory_cleanup()

    print('Emptying experts')
    for i in range(len(model.model.layers)):
        if rhasattr(model, f"model.layers.{i}.mlp.experts"):
            rsetattr(model, f"model.layers.{i}.mlp.experts", torch.nn.Module()) ## ensuring destruction of experts to avoid oom

    model=model.to_empty(device="cpu")
    
    print('Loading non experts weights')
    target_modules=[]
    for elt in weight_map:
        if not('.experts.' in elt):
            if not('gate.weight' in elt):
                target_modules.append(elt)

    model=load_weights(model, model_name, weight_map, target_modules, device)
    
    print('Creating new experts')
    for layer_idx, layer in enumerate(tqdm(model.model.layers)):
        if rhasattr(layer.mlp, "experts"):
            shared=deepcopy(layer.mlp.shared_experts) ## backup used to keep awq layers
            layer.mlp.__init__(config)
            layer.mlp.shared_experts=shared
            
            if distillation_config.pruning_method=="progressive":
                export_path=path_config.moe_states+f"/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}/layer_{layer_idx}"
            else:
                export_path=path_config.moe_states+f"/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}_{distillation_config.calibrate_merge}_{distillation_config.n_epochs}/layer_{layer_idx}"
            layer.mlp.load_state_dict(torch.load(export_path))
            
    print('Dequant Gemm layers')
    model, params = dequantize_GEMM(model, destruct=True, dtype=torch.bfloat16)
    model.to('cpu', dtype=torch.bfloat16)
    
    
    print('updating config')
    config=AutoConfig.from_pretrained(
        base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    config.n_routed_experts=distillation_config.target_routed_expert
    config.num_experts_per_tok=distillation_config.target_active_expert

    model.config=config
    
    print('Saving')
    if distillation_config.pruning_method=="progressive":
        unhealed_name=model_name+f"_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}_unhealed"
    else:
        unhealed_name=model_name+f"_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}_{distillation_config.calibrate_merge}_{distillation_config.n_epochs}_unhealed"
 
    
    unhealed_name=unhealed_name.replace('_awq', '')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.save_pretrained(unhealed_name)
    model.save_pretrained(unhealed_name)
    
    shutil.copy(os.path.join(model_name, 'modeling_deepseek.py'), os.path.join(unhealed_name, 'modeling_deepseek.py'))
    shutil.copy(os.path.join(model_name, 'configuration_deepseek.py'), os.path.join(unhealed_name, 'configuration_deepseek.py'))

    