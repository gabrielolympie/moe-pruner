from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from awq.modules.linear.gemm import WQLinear_GEMM
from torch.utils.tensorboard import SummaryWriter
from accelerate import init_empty_weights
from tqdm.auto import tqdm
from copy import deepcopy
import shutil
import numpy as np
import pickle
import torch
import json
import os

from utils.ademamix import AdEMAMix
from utils.config_utils import GenerationParams, PathConfig, DistillationParams
from utils.experts_merge_utils import dequantize_GEMM
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
    count_parameters
)

if __name__=="__main__":
    model_name = "deepseek_v2_lite_awq"
    base_model = "deepseek-ai/DeepSeek-V2-Lite"
    device="cuda:0"
    
    target_routed_expert = 8
    target_active_expert = 2

    path_config = PathConfig(
        model_name = model_name,
        intermediate_states = "data/intermediate_states",
        expert_states = "data/expert_states",
        expert_activations = "data/expert_activations",
        distillation_logs = "distillation_logs",
        moe_states="moe_states"
    )

    distillation_config = DistillationParams(
        n_epochs= 10,
        target_routed_expert = target_routed_expert,
        target_active_expert = target_active_expert,
        eval_batches=16,
        gradient_accumulation_steps= 1,
        learning_rate= 6e-4,
        end_factor= 0.05,
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
            
            export_path=path_config.moe_states+f"/distillat_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}/layer_{layer_idx}"
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
    unhealed_name=model_name+f"_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}_unhealed"
    unhealed_name=unhealed_name.replace('_awq', '')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.save_pretrained(unhealed_name)
    model.save_pretrained(unhealed_name)
    
    shutil.copy(os.path.join(model_name, 'modeling_deepseek.py'), os.path.join(unhealed_name, 'modeling_deepseek.py'))
    shutil.copy(os.path.join(model_name, 'configuration_deepseek.py'), os.path.join(unhealed_name, 'configuration_deepseek.py'))

    