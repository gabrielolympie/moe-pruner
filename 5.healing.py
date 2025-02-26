import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback, SpikeDetection
from pytorch_lightning.loggers import TensorBoardLogger 
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    HqqConfig
)
from peft import LoraConfig, get_peft_model
# from ademamix import AdEMAMix
import bitsandbytes as bnb
from bitsandbytes.optim.ademamix import AdEMAMix8bit as AdEMAMix
from bitsandbytes.optim.adamw import AdamW8bit as AdamW

from datasets import load_dataset
import os
import numpy as np
from collections import deque
import gc
import argparse
from tqdm.auto import tqdm

from torch_utils import memory_cleanup, count_parameters
from liger_kernel.transformers import apply_liger_kernel_to_llama

from pytorch_lightning.strategies import FSDPStrategy
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp
from torch.distributed.fsdp import MixedPrecision

from functools import partial
from fsdp_utils import fsdp_hqq_dora_model_for_causal_lm, get_wrapping_policy
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

class HealingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

def load_and_prepare_data(tokenizer, batch_size=8, max_length=512, num_workers=os.cpu_count(), train_sample_limit=None, val_sample_limit=None):
    dataset = load_dataset(
        "cognitivecomputations/dolphin-r1", "nonreasoning", cache_dir="../dolphin-r1"
    )["train"]

    # Apply sample limits if provided
    if train_sample_limit is not None:
        train_dataset = dataset.select(range(train_sample_limit))  # Use .select for efficiency
    else:
        train_dataset = dataset

    if val_sample_limit is not None:
        val_dataset = dataset.select(range(train_sample_limit, train_sample_limit+val_sample_limit)) # Use .select for efficiency
    else:
        val_dataset = dataset

    train_dataset = train_dataset["messages"]
    val_dataset = val_dataset["messages"]
    
    train_dataset = [
        tokenizer.apply_chat_template(elt, tokenize=False, add_generation_prompt=False)
        for elt in tqdm(train_dataset, desc="Preparing dataset train")
    ]

    val_dataset = [
        tokenizer.apply_chat_template(elt, tokenize=False, add_generation_prompt=False)
        for elt in tqdm(val_dataset, desc="Preparing dataset train")
    ]

    train_dataset = HealingDataset(
        train_dataset, tokenizer, max_length=max_length
    )
    val_dataset = HealingDataset(
        val_dataset, tokenizer, max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader

class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        learning_rate=2e-5,
        warmup_steps=100, 
        total_steps=1000,
        min_lr=1e-6,
        compilation=True
    ):
        super().__init__()
        
        ## Model HP
        self.model=model
        self.compilation=compilation

        ## Training HP
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr  # Minimum learning rate for cosine annealing
        
        # Initialize EMA tracking
        self.loss_ema = None
        self.ema_alpha = 2.0 / (10 + 1)  # Alpha for 10-step EMA

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        memory_cleanup()
        outputs = self(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        loss = outputs.loss
        
        # Update EMA
        if self.loss_ema is None:
            self.loss_ema = loss.item()
        else:
            self.loss_ema = (1 - self.ema_alpha) * self.loss_ema + self.ema_alpha * loss.item()
        
        # Replace loss with EMA if it's more than 2.5x the EMA
        original_loss = loss.item()
        if original_loss > 1.5 * self.loss_ema:
            # Create a new tensor with the EMA value, maintaining gradients
            loss = loss * (self.loss_ema / original_loss)
            self.log("loss_clipped", True, on_step=True, prog_bar=True, sync_dist=True)
        else:
            self.log("loss_clipped", False, on_step=True, prog_bar=True, sync_dist=True)
        
        # Log losses
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_ema", self.loss_ema, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_original", original_loss, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # AdamW
        optimizer = AdEMAMix(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            # foreach=False
        )
        # Use CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps,  # Adjusted total steps
            eta_min=self.min_lr  # Minimum learning rate
        )
        
        if self.warmup_steps > 0:
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                return scheduler.get_lr()[0] / self.learning_rate  # Scale by initial LR
                
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return [optimizer], [
                {"scheduler": warmup_scheduler, "interval": "step"}, # Warmup scheduler
                {"scheduler": scheduler, "interval": "step", 'start_epoch':self.warmup_steps},
            ]
            
        return {
           "optimizer": optimizer,
           "lr_scheduler": {
               "scheduler": scheduler,
               "interval": "step",
           },
        }
        
        
if __name__=="__main__":
    
        ## Set to FSDP

    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCHDYNAMO_DISABLE_GRAPH_CAPTURE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    torch.set_float32_matmul_precision('medium')
    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    weights_location="deepseek_v3"
    n_routed_experts=8
    n_active_experts=4
    epochs=1
    batch_size=1
    max_length=32
    
    learning_rate=3e-5
    train_sample_limit=32000
    val_sample_limit=512
    warmup_steps=128
    compilation=False
    checkpoint_every_n_steps=1024
    accumulate_grad_batches=1
    

    model_name=f"/home/golympie/ai-toolbox/{weights_location}_{n_routed_experts}a{n_active_experts}" ## i displaced the model on a faster disc for increased loading speed.
    
    log_name=f"{weights_location}_{n_routed_experts}a{n_active_experts}"
    log_dir="pl_logs"

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3", trust_remote_code=True
    )

    # Load and prepare data
    train_loader, val_loader = load_and_prepare_data(
        tokenizer, batch_size=batch_size, max_length=max_length,
        train_sample_limit=train_sample_limit, val_sample_limit=val_sample_limit
    )
    
    memory_cleanup()

    # Calculate total steps for the scheduler
    total_steps = len(train_loader) * epochs

    target_modules=[]
    for i in range(n_routed_experts):
        target_modules.append(f"mlp.experts.{i}.gate_proj")
        target_modules.append(f"mlp.experts.{i}.up_proj")
        target_modules.append(f"mlp.experts.{i}.down_proj")

    target_modules.append('mlp.gate.weight')

    model=fsdp_hqq_dora_model_for_causal_lm(
        model_name,
        target_modules=target_modules,
        lora_rank=4,
        lora_alpha=4,
        lora_dropout=0.1,
        n_workers=8
    )
    
    memory_cleanup()
    count_parameters(model)
    
    config_dict = model.config.to_dict()
    
    if compilation:
        print('compile model')
        model = torch.compile(model)
        
    memory_cleanup()
    # Create the Lightning model
    # pl_model = LightningModel(
    #     model,
    #     tokenizer=tokenizer,
    #     learning_rate=learning_rate,
    #     warmup_steps=warmup_steps,
    #     total_steps=total_steps,
    #     compilation=compilation
    # )

    # Set up checkpoint directory
    os.makedirs('checkpoints_full/', exist_ok=True)
    os.makedirs('checkpoints_full/'+log_name, exist_ok=True)

    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_full/'+log_name,
        filename="{epoch}-{step}-{train_loss:.4f}",
        save_on_train_epoch_end=False,
        every_n_train_steps=checkpoint_every_n_steps,
        save_top_k=3,
        monitor="train_loss",
        save_last=True,
    )

    # Set up logger
    logger = TensorBoardLogger(log_dir, name=log_name)

    # os.environ['MASTER_ADDR'] = "localhost"
    # os.environ['MASTER_PORT'] ="12355"
    # world_size=2
    # local_rank=1
    
    # if 'SLURM_PROCID' in os.environ:
    #     # assumes same number of GPUs per node.
    #     rank = int(os.environ['SLURM_PROCID']) * torch.cuda.device_count() + local_rank
    # else:
    #     rank = local_rank

    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(local_rank)


    from modeling_deepseek_v3 import (
        ATTENTION_CLASSES,
        DeepseekV3DecoderLayer,
        DeepseekV3MLP,
    )

    LLAMA_ATTENTION_CLASSES=ATTENTION_CLASSES
    LlamaDecoderLayer=DeepseekV3DecoderLayer
    LlamaMLP=DeepseekV3MLP
    
    auto_wrap_policy = get_wrapping_policy(
        LLAMA_ATTENTION_CLASSES=LLAMA_ATTENTION_CLASSES,
        LlamaDecoderLayer=LlamaDecoderLayer,
        LlamaMLP=LlamaMLP
    )
    
    mixed_precision_config = MixedPrecision(
        param_dtype=torch.bfloat16,  # or torch.bfloat16
        reduce_dtype=torch.bfloat16,  # or torch.bfloat16
        buffer_dtype=torch.bfloat16   # or torch.bfloat16
    )
    
    
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True),
        limit_all_gathers=True,
        sync_module_states=True,
        mixed_precision=mixed_precision_config,
    )
    
    # dist.destroy_process_group()
    
    # fsdp_strategy = FSDPStrategy(
    #     sharding_strategy="FULL_SHARD",
    #     auto_wrap_policy=auto_wrap_policy,
    #     use_orig_params=False,
    #     cpu_offload=CPUOffload(offload_params=True),
    #     limit_all_gathers=True,
    #     sync_module_states=True,
    #     mixed_precision=mixed_precision_config,
    # )
        
    # # Configure trainer with FSDP strategy
    # trainer = pl.Trainer(
    #     max_epochs=epochs,
    #     accelerator="gpu",
    #     devices=2,
    #     strategy=fsdp_strategy,  # Use the configured FSDP strategy
    #     callbacks=[checkpoint_callback],
    #     accumulate_grad_batches=accumulate_grad_batches,
    #     logger=logger,
    #     precision="bf16-mixed",
    #     log_every_n_steps=1,
    #     gradient_clip_val=1.0,  # Add gradient clipping to prevent instability
    # )

    # # Start training
    # trainer.fit(pl_model, train_loader, val_loader)