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

from peft.tuners import lora
from utils.ademamix import AdEMAMix
from utils.config_utils import PathConfig, DistillationParams

from utils.adapters import DoRAAdapter

from utils.experts_merge_utils import (
    dequantize_GEMM,
    prepare_distillat_topk,
    prepare_distillat_state_cl,
    prepare_distillat_act_cl
)

from utils.torch_utils import (
    load_quant,
    rsetattr,
    destruct_module_optimized,
    memory_cleanup,
    load_weights,
    WarmupCosineAnnealingLR
)


torch.set_float32_matmul_precision('medium')

# python 3_layer_distillation.py --pruning_method topk --device cuda:0 --model_name deepseek_coder_v2_lite_instruct_awq --start_layer 1 --end_layer 15 --calibrate_merge 1 --n_epochs 1  --target_routed_expert 8 --target_active_expert 4
# python 3_layer_distillation.py --pruning_method topk --device cuda:1 --model_name deepseek_coder_v2_lite_instruct_awq --start_layer 15 --end_layer 27 --calibrate_merge 1 --n_epochs 1  --target_routed_expert 8 --target_active_expert 4

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Two-layer distillation script.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--model_name", type=str, default="deepseek_v2_lite_awq", help="Name of the model.")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--start_layer", type=int, default=1, help="Starting layer.")
    parser.add_argument("--end_layer", type=int, default=27, help="Ending layer.")
    parser.add_argument("--target_routed_expert", type=int, default=8, help="Target routed expert.")
    parser.add_argument("--target_active_expert", type=int, default=2, help="Target active expert.")
    parser.add_argument("--dora_rank", type=int, default=16, help="Target active expert.")
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
    pruning_method= args.pruning_method
    
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
        learning_rate= 4e-4,
        end_factor= 0.1,
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

        distilled_mlp.config.n_routed_experts=distillation_config.target_routed_expert
        distilled_mlp.config.num_experts_per_tok=distillation_config.target_active_expert
        distilled_mlp.n_routed_experts=distillation_config.target_routed_expert
        distilled_mlp.num_experts_per_tok=distillation_config.target_active_expert

        destruct_module_optimized(model)
        memory_cleanup()
        
        if distillation_config.pruning_method == "state_cl":
            ## Method 1 : By expert output similarity
            distillat=prepare_distillat_state_cl(distilled_mlp, layer_norm, model.config.scoring_func, distillation_config, path_config, layer_idx, device)
        elif distillation_config.pruning_method == "act_cl":
            ## Method 2 : By analysing the cooccurence of the experts
            distillat=prepare_distillat_act_cl(distilled_mlp, layer_norm, model.config.scoring_func,  distillation_config, path_config, layer_idx, device)
        else:
            ## Method 3 keep n most asctivated experts
            distilled_mlp=prepare_distillat_topk(distilled_mlp, layer_norm, distillation_config, path_config, layer_idx, device)
        
        ## Distill
        os.makedirs(path_config.moe_states, exist_ok=True)
        os.makedirs(path_config.moe_states+f"/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}_{distillation_config.calibrate_merge}_{distillation_config.n_epochs}", exist_ok=True)
        export_path=path_config.moe_states+f"/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}_{distillation_config.calibrate_merge}_{distillation_config.n_epochs}/layer_{layer_idx}"
        
        if distillation_config.calibrate_merge:
            ## Prepare for train
            ### Ensure that gates are not frozen
            for name, parameter in distilled_mlp.named_parameters():
                if 'gate.' in name:
                    parameter.requires_grad=True
                else:
                    parameter.requires_grad=False

            # distilled_mlp, _=dequantize_GEMM(distilled_mlp)
            for name, module in tqdm(distilled_mlp.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    rsetattr(
                        distilled_mlp,
                        name,
                        lora.Linear(
                            module,
                            adapter_name="adapter",
                            r=distillation_config.dora_rank,
                            lora_alpha=distillation_config.dora_rank,
                            lora_dropout=0.05,
                            use_dora=True,
                        ).to(device=device, dtype=torch.bfloat16)
                    )
        
            
            writer = SummaryWriter(log_dir=path_config.distillation_logs+f"/distillat_{distillation_config.pruning_method}_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}_{distillation_config.calibrate_merge}_{distillation_config.n_epochs}/layer_{layer_idx}")

            # distillation_config.learning_rate=3e-4

            n_epochs = distillation_config.n_epochs

            optimizer = AdEMAMix(
                distilled_mlp.parameters(),
                lr=distillation_config.learning_rate,
                betas=(0.7, 0.999, 0.9999),
                alpha=5
            )

            eval_batches = distillation_config.eval_batches

            criterion = torch.nn.functional.smooth_l1_loss

            train_batches = len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}"))) - distillation_config.eval_batches

            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                warmup_steps=distillation_config.gradient_accumulation_steps * 0, ## warmup for 32 virtual steps
                total_steps=train_batches * n_epochs ,
                min_lr=distillation_config.learning_rate * distillation_config.end_factor
            )

            patience = 2  # Number of epochs to wait for improvement
            margin = 1e-4  # Minimum improvement required
            best_loss = float('inf')
            patience_counter = 0

            # distilled_mlp=torch.compile(distilled_mlp)

            # Training and evaluation loop
            for epoch in range(n_epochs):
                distilled_mlp.train()
                progress_bar = tqdm(range(train_batches), desc=f"Calibrating merged expert, epoch {epoch}")
                for batch_idx in progress_bar:
                    with torch.amp.autocast(device):
                        hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]
                        outputs = load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]

                        residual = hidden_states
                        hidden_states = layer_norm(hidden_states)

                        pred = distilled_mlp(hidden_states)
                        
                        pred = pred + residual
                        
                        loss = criterion(pred, outputs)
                    loss.backward()
                    if (epoch * train_batches + batch_idx + 1) % distillation_config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    # Log the training loss
                    scheduler.step()
                    writer.add_scalar('Loss/train', loss.item(), epoch * train_batches + batch_idx)

                    progress_bar.set_postfix(loss=loss.item())

                # Evaluation phase at the end of each epoch
                distilled_mlp.train()
                eval_loss = 0
                
                with torch.no_grad():
                    for batch_idx in range(train_batches, train_batches + distillation_config.eval_batches):
                        hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]
                        outputs = load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)[:, distillation_config.skip_first_tokens:]

                        residual = hidden_states
                        hidden_states = layer_norm(hidden_states)

                        pred = distilled_mlp(hidden_states)
                        
                        pred = pred + residual

                        
                        loss = criterion(pred, outputs)
                        eval_loss += loss.item()

                eval_loss /= eval_batches
                writer.add_scalar('Loss/eval', eval_loss, epoch)
                print(f"Epoch {epoch + 1}/{n_epochs}, Evaluation Loss: {eval_loss}")

                if best_loss - eval_loss > margin:
                    best_loss = eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break

            writer.close()

        
        ## Merge adapter and save
        
        for name, module in tqdm(distilled_mlp.named_modules()):
            if isinstance(module, lora.Linear):
                module.merge()
                
                rsetattr(
                    distilled_mlp,
                    name,
                    module.base_layer
                )
        
        torch.save(distilled_mlp.state_dict(), export_path)

                
        