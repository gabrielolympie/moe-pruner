from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.tensorboard import SummaryWriter

from accelerate import init_empty_weights
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np
import argparse
import torch
import json
import os

from utils.ademamix import AdEMAMix
from utils.config_utils import PathConfig, DistillationParams

from utils.adapters import DoRAAdapter

from utils.experts_merge_utils import (
    dequantize_GEMM,
    prepare_distillat_topk,
    prepare_distillat_state_cl,
    prepare_distillat_act_cl,
    prepare_moe_for_distillation,
    halve_distilled_mlp,
    merge_and_unload,
    calibrated_dequant,
    build_affinity_matrix,
    expert_clustering,
    cooccurrence_matrix,
    group_items_by_affinity
)

from utils.torch_utils import (
    load_quant,
    rsetattr,
    destruct_module_optimized,
    memory_cleanup,
    load_weights,
    WarmupCosineAnnealingLR
)

from utils.fused import FusedMOE
import pickle


torch.set_float32_matmul_precision('medium')

# python 3_layer_distillation_fused.py --device cuda:0 --model_name deepseek_coder_v2_lite_instruct_awq --start_layer 1 --end_layer 15 --calibrate_merge 1 --n_epochs 2 --target_routed_expert 4 --target_active_expert 1
# python 3_layer_distillation_fused.py --device cuda:1 --model_name deepseek_coder_v2_lite_instruct_awq --start_layer 15 --end_layer 27 --calibrate_merge 1 --n_epochs 2 --target_routed_expert 4 --target_active_expert 1

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Two-layer distillation script.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--model_name", type=str, default="deepseek_v2_lite_awq", help="Name of the model.")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--start_layer", type=int, default=1, help="Starting layer.")
    parser.add_argument("--end_layer", type=int, default=27, help="Ending layer.")
    parser.add_argument("--target_routed_expert", type=int, default=8, help="Target routed expert.")
    parser.add_argument("--target_active_expert", type=int, default=2, help="Target active expert.")
    parser.add_argument("--dora_rank", type=int, default=32, help="Target active expert.")
    parser.add_argument("--pruning_method", type=str, default="topk", help="Target active expert.")
    parser.add_argument("--calibrate_merge", type=int, default=1, help="Target active expert.")
    
    args = parser.parse_args()

    device = args.device
    model_name = args.model_name
    n_epochs = args.n_epochs
    start_layer = args.start_layer
    end_layer = args.end_layer
    target_routed_expert = args.target_routed_expert
    target_active_expert = args.target_active_expert
    dora_rank = args.dora_rank
    calibrate_merge= args.calibrate_merge == 1
    pruning_method= "fused"
    
    print(pruning_method)
    torch.set_float32_matmul_precision('medium')

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
        gradient_accumulation_steps= 1,
        learning_rate= 2e-4,
        end_factor= 0.3,
        calibrate_merge=calibrate_merge,
        skip_first_tokens=0, ## useful to avoid tuning on early tokens that have less informations
        pruning_method=pruning_method, # topk , act_cl, state_cl
        dora_rank=dora_rank,
    )
    
    print(distillation_config.calibrate_merge)
    
    with open(f"{model_name}/model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False

    model.train()
    destruct_module_optimized(model)
    memory_cleanup()
    
    # layer_idx=23
    for layer_idx in range(start_layer, end_layer):
        
        model.model.layers[layer_idx].to_empty(device=device)
        target_modules=[f".layers.{layer_idx}."]
        model=load_weights(model, model_name, weight_map, target_modules, device)

        distilled_mlp=deepcopy(model.model.layers[layer_idx].mlp).to(device)
        layer_norm=deepcopy(model.model.layers[layer_idx].post_attention_layernorm).to(device, dtype=torch.bfloat16)

        distilled_mlp.gate = distilled_mlp.gate.to(torch.bfloat16)
        
        for param in distilled_mlp.parameters():
            param.requires_grad=False
            
        with open(os.path.join(path_config.expert_activations, f"layer_{layer_idx}.pickle"), "rb") as f:
            (top_k_output, top_k_weight) = pickle.load(f)

        top_k_output=top_k_output.detach().to(torch.int64).cpu().numpy()
        affinity_matrix = cooccurrence_matrix(top_k_output, len(np.unique(top_k_output)))
        affinity_matrix=(affinity_matrix - affinity_matrix.min())/(affinity_matrix.max()-affinity_matrix.min())

        train_batches=2048

        group_size=affinity_matrix.shape[0] // distillation_config.target_routed_expert
        
        
        merge_method="slerp"
        eval_batches=16


        fused_moe = FusedMOE(distilled_mlp)
        fused_moe.fuse(affinity_matrix, group_size, train_batches, learning_rate=distillation_config.learning_rate, device=device, merge_method=merge_method, rank=distillation_config.dora_rank, adapter_type="mixture")
        fused_moe.train_mode(distillation_config.learning_rate, train_batches * distillation_config.n_epochs)
        fused_moe = torch.compile(fused_moe)

        writer = SummaryWriter(log_dir=f'multiplex_runs/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}_mixture/layer{layer_idx}')

        for epoch in range(distillation_config.n_epochs):
            # Training phase
            progress_bar = tqdm(range(train_batches - eval_batches), desc=f"Calibrating fused_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}_mixture_layer{layer_idx}")
            for batch_idx in progress_bar:
                hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]
                output = load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]
                loss = fused_moe.train_step(hidden_states, layer_norm, temperature=1, output=output, gradient_accumulation_step=distillation_config.gradient_accumulation_steps)
                progress_bar.set_postfix(loss=loss.item())
                writer.add_scalar(f'Loss/train', loss.item(), batch_idx + epoch * (train_batches - eval_batches))

            # Evaluation phase
            eval_progress_bar = tqdm(range(train_batches - eval_batches, train_batches), desc=f"Evaluating fused_{distillation_config.learning_rate}_{merge_method}_{distillation_config.dora_rank}_mixture")
            total_eval_loss = 0
            for batch_idx in eval_progress_bar:
                hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]
                output = load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]

                residual = deepcopy(hidden_states)
                hidden_states = layer_norm(hidden_states)
                pred = fused_moe.forward(hidden_states) + residual

                local_loss = torch.nn.functional.smooth_l1_loss(pred, output, reduction='mean')
                total_eval_loss += local_loss.item()
                eval_progress_bar.set_postfix(loss=local_loss.item())

            avg_eval_loss = total_eval_loss / eval_batches
            writer.add_scalar(f'Loss/eval', avg_eval_loss, epoch)

        # Close the writer
        writer.close()

        
        fused_moe.set_ready() ## Important, this removes the original experts to save some space

        os.makedirs(path_config.moe_states+f"/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}", exist_ok=True)
        export_path=path_config.moe_states+f"/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}/layer_{layer_idx}"
        torch.save(fused_moe.state_dict(), export_path)

                
        