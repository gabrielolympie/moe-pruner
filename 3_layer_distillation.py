from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from accelerate import init_empty_weights
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np
import argparse
import pickle
import torch
import json
import os

from utils.experts_gate_utils import create_gate
from utils.ademamix import AdEMAMix
from utils.config_utils import GenerationParams, PathConfig, DistillationParams
from utils.expert_merge_utils import calibrated_merge_experts
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

# python 3_layer_distillation.py --device cuda:0 --model_name deepseek_v2_lite_awq --start_layer 1 --end_layer 15 --target_routed_expert 4 --target_active_expert 2
# python 3_layer_distillation.py --device cuda:1 --model_name deepseek_v2_lite_awq --start_layer 15 --end_layer 27 --target_routed_expert 4 --target_active_expert 2

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Two-layer distillation script.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--model_name", type=str, default="deepseek_v2_lite_awq", help="Name of the model.")
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of epochs.")
    parser.add_argument("--start_layer", type=int, default=1, help="Starting layer.")
    parser.add_argument("--end_layer", type=int, default=27, help="Ending layer.")
    parser.add_argument("--target_routed_expert", type=int, default=4, help="Target routed expert.")
    parser.add_argument("--target_active_expert", type=int, default=2, help="Target active expert.")

    args = parser.parse_args()

    device = args.device
    model_name = args.model_name
    n_epochs = args.n_epochs
    start_layer = args.start_layer
    end_layer = args.end_layer
    target_routed_expert = args.target_routed_expert
    target_active_expert = args.target_active_expert

    device_id = device.split(":")[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    device="cuda:0"

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
        learning_rate= 1e-4,
        end_factor= 0.05,
    )
    
    with open(f"{model_name}/model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]

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
    
    
    patience = 2  # Number of epochs to wait for improvement
    margin = 1e-4  # Minimum improvement required
    best_loss = float('inf')
    patience_counter = 0
    
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
        
        ## Adapting gating mecanism
        distilled_mlp.gate, mapping_dict, inv_mapping_dict=create_gate(
        distilled_mlp.gate,
            layer_norm,
            path_config,
            distillation_config,
            layer_idx,
            scoring_func=model.config.scoring_func,
            device=device
        )
        distilled_mlp.gate=distilled_mlp.gate.to(dtype=torch.bfloat16)
        
        
        ## Dequant and merge the experts
        new_experts=distilled_mlp.experts[:distillation_config.target_routed_expert]

        for i, expert_list in inv_mapping_dict.items():
            new_experts[i]=calibrated_merge_experts(
                expert_list,
                layer_norm,
                distilled_mlp,
                path_config,
                layer_idx
            )

        distilled_mlp.experts=new_experts
        
        ## Prepare for train
        for name, parameter in distilled_mlp.named_parameters():
            if 'experts.' in name and not("shared_experts" in name):
                print(name)
                parameter.requires_grad=True
            elif 'gate.' in name:
                parameter.requires_grad=True
            else:
                parameter.requires_grad=False
                
        ## Distill
        writer = SummaryWriter(log_dir=path_config.distillation_logs+f"/distillat_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}/layer_{layer_idx}")

        os.makedirs(path_config.moe_states, exist_ok=True)
        os.makedirs(path_config.moe_states+f"/distillat_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}", exist_ok=True)
        export_path=path_config.moe_states+f"/distillat_{distillation_config.target_routed_expert}a{distillation_config.target_active_expert}/layer_{layer_idx}"

        n_epochs = distillation_config.n_epochs

        optimizer = AdEMAMix(
            distilled_mlp.parameters(),
            lr=distillation_config.learning_rate
        )

        eval_batches = distillation_config.eval_batches

        criterion = torch.nn.MSELoss()

        train_batches = len(os.listdir(os.path.join(path_config.expert_states, f"layer_{layer_idx}"))) - distillation_config.eval_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_batches * n_epochs, eta_min=distillation_config.learning_rate * distillation_config.end_factor)

            
        # Training and evaluation loop
        for epoch in range(n_epochs):
            distilled_mlp.train()
            
            progress_bar = tqdm(range(train_batches), desc=f"Calibrating merged expert, epoch {epoch}")
            for batch_idx in progress_bar:
                with torch.amp.autocast(device):
                    hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)
                    outputs = load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)

                    residual = hidden_states
                    hidden_states = layer_norm(hidden_states)

                    pred = distilled_mlp(hidden_states)
                    pred = pred + residual

                    loss = criterion(pred, outputs)

                if (epoch * train_batches + batch_idx) % distillation_config.gradient_accumulation_steps == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Log the training loss
                scheduler.step()
                writer.add_scalar('Loss/train', loss.item(), epoch * train_batches + batch_idx)

                progress_bar.set_postfix(loss=loss.item())

            # Evaluation phase at the end of each epoch
            distilled_mlp.eval()
            eval_loss = 0
            
            with torch.no_grad():
                for batch_idx in range(train_batches, train_batches + distillation_config.eval_batches):
                    hidden_states = load_quant(os.path.join(path_config.expert_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)
                    outputs = load_quant(os.path.join(path_config.intermediate_states, f"layer_{layer_idx}", f"batch_{batch_idx}")).to(device, dtype=torch.bfloat16)

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
        
        ## Export final
        torch.save(distilled_mlp.state_dict(), export_path)

                
        