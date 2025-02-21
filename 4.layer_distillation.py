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


# python 4.layer_distillation.py --layer_idx 13  --device cuda:0 --n_routed_experts 4 --n_active_experts 1

if __name__ == "__main__":
    print("Starting the pipeline...")
    
    
    mp.set_start_method('spawn', force=True)
    
    torch.backends.cuda.max_split_size_mb = 512
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Distillation pipeline arguments.')
    parser.add_argument('--layer_idx', type=int, default=13, help='Layer index to distill.')
    parser.add_argument('--n_routed_experts', type=int, default=8, help='Number of routed experts.')
    parser.add_argument('--n_active_experts', type=int, default=2, help='Number of active experts.')
    parser.add_argument('--learning_rate', type=float, default=8e-4, help='Learning rate.')
    parser.add_argument('--end_factor', type=float, default=0.1, help='End factor for learning rate scheduler.')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (e.g., cuda:0 or cpu).')

    
    
    args = parser.parse_args()
    
    path_config = PathConfig()

    layer_idx = args.layer_idx
    n_routed_experts = args.n_routed_experts
    n_active_experts = args.n_active_experts
    learning_rate = args.learning_rate
    end_factor = args.end_factor
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    device = args.device
    
    params = DistillationParams(
        n_epochs=1,
        n_batch=128,
        n_train_batch=116,
        batch_size=16,
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
    model = MOEDistillerLightningModule(
        path_config,
        params,
        layer_idx=layer_idx,
        n_routed_experts=n_routed_experts,
        n_active_experts=n_active_experts,
    )

    model.create()
    print("Model initialized.")

    print("Setting up logger and checkpoint callback...")
    logger = TensorBoardLogger(
        path_config.log_dir,
        name=f"lightning_logs_layer_{layer_idx}_{n_routed_experts}a{n_active_experts}"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=path_config.base_dir,
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
    trainer.fit(model, train_loader, val_loader)
    print("Training finished.")
    print("Pipeline completed.")

    print(model)
        