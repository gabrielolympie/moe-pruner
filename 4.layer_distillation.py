import numpy as np
import os
from configs import DistillationParams, PathConfig
import torch
from Distiller import IntermediateStateDataset, prepare_distilled_moe, MOEDistillerLightningModule
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing as mp
from ademamix import AdEMAMix
import argparse

from torch_utils import memory_cleanup, destruct_module_optimized
from model_utils import rsetattr, rgetattr, load_model_config, load_weight, map_device, assign_device, get_dataset, get_device_map
from accelerate import init_empty_weights
from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3MoE, DeepseekV3ForCausalLM
from tqdm.auto import tqdm
import pickle

# python 4.layer_distillation.py  --device cuda:0 --n_routed_experts 8 --n_active_experts 4 --min_layer 58 --max_layer 59

if __name__ == "__main__":
    print("Starting the pipeline...")
    
    
    mp.set_start_method('spawn', force=True)
    
    torch.backends.cuda.max_split_size_mb = 512
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Distillation pipeline arguments.')
    parser.add_argument('--n_routed_experts', type=int, default=8, help='Number of routed experts.')
    parser.add_argument('--n_active_experts', type=int, default=2, help='Number of active experts.')
    parser.add_argument('--learning_rate', type=float, default=8e-4, help='Learning rate.')
    parser.add_argument('--end_factor', type=float, default=0.1, help='End factor for learning rate scheduler.')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (e.g., cuda:0 or cpu).')
    parser.add_argument('--min_layer', type=int, default=3, help='Device to use (e.g., cuda:0 or cpu).')
    parser.add_argument('--max_layer', type=int, default=61, help='Device to use (e.g., cuda:0 or cpu).')
    
    
    args = parser.parse_args()
    
    path_config = PathConfig()

    n_routed_experts = args.n_routed_experts
    n_active_experts = args.n_active_experts
    learning_rate = args.learning_rate
    end_factor = args.end_factor
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    device = args.device
    weights_location='deepseek_v3/'
    min_layer = args.min_layer
    max_layer = args.max_layer
    
    params = DistillationParams(
        n_epochs=2,
        n_batch=256,
        n_train_batch=232,
        batch_size=8,
        max_length=512,
        gradient_accumulation_steps=1,
        calibration_batches=16,
        learning_rate=learning_rate,
        end_factor=end_factor,
        temperature=1.0,
        lora_type="dora",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_workers=8,
        fp8_format="e4m3",
        distiller_device=device,
    )

    # Load Model Config and Tokenizer
    weight_map, config = load_model_config(weights_location)
    # Create empty model
    with init_empty_weights():
        model = DeepseekV3ForCausalLM(config)
    
    for layer_idx in range(min_layer,max_layer):
        print("Loading dataset...")
        full_dataset = IntermediateStateDataset(path_config, layer_idx, 0, params.n_batch)
        
        train_size = params.n_train_batch
        val_size = params.n_batch - params.n_train_batch
        
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        print("Dataset loaded and split.")

            # Define dataloaders
        print("Creating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=4
        )
        print("Dataloaders created.")

        print("Initializing model...")
        

        

        device_map = get_device_map(layer_idx, weight_map, device)
        model.model.layers[layer_idx] = model.model.layers[layer_idx].to_empty(device=device)
        
        for i, weight_name in enumerate(tqdm(device_map)):
            rsetattr(model, weight_name, load_weight(weights_location, weight_name, weight_map, device))
            if i%100 ==0:
                memory_cleanup()
                
        with open(f"{path_config.expert_activation_dir}/layer_{layer_idx}.pickle", "rb") as f:
            act = pickle.load(f)

        v,c = np.unique(act, return_counts=True)
        selected_experts = np.flip(np.argsort(c))
        
        pl_model = MOEDistillerLightningModule(
            weight_map,
            path_config,
            params,
            layer_idx=layer_idx,
            n_routed_experts=n_routed_experts,
            n_active_experts=n_active_experts,
            weights_location=weights_location
        )
        
        pl_model.distillat=prepare_distilled_moe(
            model.model.layers[layer_idx].mlp,
            selected_experts,
            n_routed_experts,
            n_active_experts,
            params,
            device=device
        )

        destruct_module_optimized(model)
        memory_cleanup()
        
        print("Model initialized.")

        print("Setting up logger and checkpoint callback...")
        logger = TensorBoardLogger(
            path_config.log_dir,
            name=f"lightning_logs_layer_{layer_idx}_{n_routed_experts}a{n_active_experts}"
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=path_config.checkpoint_dir,
            filename=f"moe_distiller_layer_{layer_idx}_{n_routed_experts}a{n_active_experts}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )
        print("Logger and checkpoint callback setup.")

        # Define trainer
        print("Initializing trainer...")
        dev=int(params.distiller_device.split(':')[1])
        
        print(dev)
        trainer = pl.Trainer(
            max_epochs=params.n_epochs,
            accelerator="gpu",
            devices=[dev],
            logger=logger,
            callbacks=[checkpoint_callback],
            precision="bf16-mixed",
            gradient_clip_val=1.0,
            accumulate_grad_batches=params.gradient_accumulation_steps,
            enable_progress_bar=True,
            log_every_n_steps=1,
            # strategy="ddp"  # Add strategy for multi-gpu training
        )
        print("Trainer initialized.")

        print("Starting training...")
        trainer.fit(pl_model, train_loader, val_loader)

        print("Training finished.")
        pl_model.merge_and_save()
        print("Pipeline completed.")

        