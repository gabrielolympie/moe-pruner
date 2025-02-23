import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from configs import PathConfig, DistillationParams
from adapters import DoRALinear
from modeling_deepseek import DeepseekV3MoE

from copy import deepcopy
import pickle
import numpy as np
import os
from configs import DistillationParams, PathConfig
import torch
from tqdm.auto import tqdm

from model_utils import rsetattr, rgetattr, load_model_config, load_weight, map_device, assign_device, get_dataset, get_device_map 
from torch_utils import load_intermediate_state, load_midlayer_state, memory_cleanup, destruct_module_optimized, count_parameters
from ademamix import AdEMAMix

class IntermediateStateDataset(Dataset):
    def __init__(self, path_config, layer_idx, start_batch, end_batch):
        self.path_config = path_config
        self.layer_idx = layer_idx
        self.start_batch = start_batch
        self.end_batch = end_batch

    def __len__(self):
        return self.end_batch - self.start_batch

    def __getitem__(self, idx):
        batch_idx = self.start_batch + idx
        input_data = load_midlayer_state(self.path_config, self.layer_idx, batch_idx, batch_size=8)
        output_data = load_intermediate_state(self.path_config, self.layer_idx, batch_idx, batch_size=8)
        return input_data, output_data

def prepare_distilled_moe(
    moe,
    selected_experts,
    n_routed_experts,
    n_active_experts,
    distillation_config,
    device="cuda:1"
):
    """Prepare distilled MOE with specified adapter type"""
    moe_config = deepcopy(moe.config)
    # Select adapter class based on type
    AdapterClass = DoRALinear if distillation_config.lora_type.lower() == "dora" else LoRALinear  # Corrected attribute name here
    # Create new MoE
    moe_config.n_routed_experts = n_routed_experts
    moe_config.num_experts_per_tok = n_active_experts
    
    with torch.device(device):
        pruned_moe = DeepseekV3MoE(moe_config).to(device)
    memory_cleanup()
    
    pruned_moe.shared_experts.gate_proj = deepcopy(moe.shared_experts.gate_proj).to(device)
    pruned_moe.shared_experts.up_proj = deepcopy(moe.shared_experts.up_proj).to(device)
    pruned_moe.shared_experts.down_proj = deepcopy(moe.shared_experts.down_proj).to(device)
    
    for param in pruned_moe.parameters():
        param.requires_grad = False

    # Only the gate has full finetuning
    pruned_moe.gate.weight = torch.nn.Parameter(moe.gate.weight[list(selected_experts)[:n_routed_experts]].detach().to(device))

    # Update Experts with chosen adapter
    for i, ind in enumerate(list(selected_experts)[:n_routed_experts]):
        expert = deepcopy(moe.experts[ind])
        pruned_moe.experts[i].to_empty(device=device)

        adapter_kwargs = {
            "rank": distillation_config.lora_rank,
            "alpha": distillation_config.lora_alpha,
            "device": device, # Changed from distillation_config.device
        }

        if distillation_config.lora_type.lower() == "dora":
            # adapter_kwargs["dora_simple"] = distillation_config.dora_simple #dora_simple missing
            pass

        # Apply adapter to each projection
        pruned_moe.experts[i].gate_proj = AdapterClass(expert.gate_proj, **adapter_kwargs).to(device)
        pruned_moe.experts[i].up_proj = AdapterClass(expert.up_proj, **adapter_kwargs).to(device)
        pruned_moe.experts[i].down_proj = AdapterClass(expert.down_proj, **adapter_kwargs).to(device)

    pruned_moe=pruned_moe.to(device)

    # Set requires_grad for adapter parameters only
    for name, param in pruned_moe.named_parameters():
        if any(x in name for x in ["lora_", "weight_m"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    for param in pruned_moe.gate.parameters():
        param.requires_grad = True
        
    for name, param in pruned_moe.named_parameters():
        param.data = param.data.to(device)

    return pruned_moe

class MOEDistillerLightningModule(pl.LightningModule):
    def __init__(self, weight_map, path_config, params, layer_idx, n_routed_experts, n_active_experts, weights_location, **kwargs):
        super().__init__(**kwargs)
        self.weight_map = weight_map
        self.path_config = path_config
        self.params = params
        self.distillation_config=params #kept as self.params too.
        self.layer_idx = layer_idx
        self.model_name = self.path_config.model_name
        self.n_routed_experts = n_routed_experts
        self.n_active_experts = n_active_experts
        self.distillat = None
        self.device_local = self.params.distiller_device
        self.weights_location=weights_location

    def forward(self, input_batch):
        return self.distillat(input_batch) ## zero selection as batch are already pre built

    def training_step(self, batch, batch_idx):
        input_batch, output_batch = batch
        input_batch, output_batch=input_batch[0], output_batch[0]
        
        
        pred = self(input_batch)
        
        ## Replace all input_batch and output_batch nan values by 0
        input_batch[torch.isnan(input_batch)] = 0
        output_batch[torch.isnan(output_batch)] = 0
        
        # Compute mean and std of output_batch
        m = torch.mean(output_batch, dim=-1, keepdim=True)
        s = torch.std(output_batch, dim=-1, keepdim=True)
        # Normalize output_batch and pred
        output_batch = (output_batch - m) / s
        pred = (pred - m) / s
        
        loss = torch.nn.functional.mse_loss(pred, output_batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, output_batch = batch
        input_batch, output_batch=input_batch[0], output_batch[0]
        
        
        pred = self(input_batch)
        
        ## Replace all input_batch and output_batch nan values by 0
        input_batch[torch.isnan(input_batch)] = 0
        output_batch[torch.isnan(output_batch)] = 0

        # Compute mean and std of output_batch
        m = torch.mean(output_batch, dim=-1, keepdim=True)
        s = torch.std(output_batch, dim=-1, keepdim=True)
        
        ## Normalize output_batch and pred
        output_batch = (output_batch - m) / s
        pred = (pred - m) / s
        
        loss = torch.nn.functional.mse_loss(pred, output_batch)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):             
        optimizer = AdEMAMix(
            filter(lambda p: p.requires_grad, self.distillat.parameters()),
            lr=self.params.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.params.end_factor,
            total_iters=(self.params.n_epochs * self.params.n_train_batch) // self.params.gradient_accumulation_steps
        )


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Called every training step
                "frequency": 1
            }
        }
        
    def merge_and_save(self):
        """
        Load the best checkpoint and save the distillat state to disk.
        
        Args:
            checkpoint_dir (str): Directory containing the checkpoints
            save_dir (str): Directory where to save the final model
        """
        # Find the best checkpoint
        checkpoint_dir=self.path_config.checkpoint_dir
        save_dir=self.path_config.get_distillation_path(n_experts=self.n_routed_experts, n_active=self.n_active_experts)
        
        # checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        # if not checkpoints:
        #     raise ValueError(f"No checkpoints found in {checkpoint_dir}")
            
        # # Sort checkpoints by validation loss (assuming checkpoint names contain val_loss)
        # best_checkpoint = None
        # best_val_loss = float('inf')
        
        # for ckpt in checkpoints:
        #     try:
        #         checkpoint = torch.load(ckpt, map_location=self.device_local)
        #         val_loss = checkpoint['val_loss']
        #         if val_loss < best_val_loss:
        #             best_val_loss = val_loss
        #             best_checkpoint = ckpt
        #     except:
        #         print(f"Warning: Could not load checkpoint {ckpt}")
        #         continue
        
        # if best_checkpoint is None:
        #     raise ValueError("Could not find a valid checkpoint")
            
        # # Load the best checkpoint
        # print(f"Loading best checkpoint: {best_checkpoint}")
        # self.load_from_checkpoint(best_checkpoint)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the distillat state
        
        save_path = save_dir + f"/layer_{self.layer_idx}.pt"
        
        ## Merge all adapters in distillat
        for i in range(self.n_routed_experts):
            self.distillat.experts[i].gate_proj=self.distillat.experts[i].gate_proj.merge_and_unload()
            self.distillat.experts[i].up_proj=self.distillat.experts[i].up_proj.merge_and_unload()
            self.distillat.experts[i].down_proj=self.distillat.experts[i].down_proj.merge_and_unload()
        
        
        torch.save(self.distillat.state_dict(), save_path)
        print(f"Saved distillat state to: {save_path}")
        
        destruct_module_optimized(self.distillat)
        memory_cleanup()
