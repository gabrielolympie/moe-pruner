import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import pickle
from copy import deepcopy
import time
from torch.utils.tensorboard import SummaryWriter
# Keeping all imports from the original code
from torch_utils import memory_cleanup, destruct_module_optimized, load_intermediate_state, load_midlayer_state
from model_utils import (
    rsetattr,
    rgetattr,
    load_model_config,
    load_weight,
    map_device,
    assign_device,
    get_dataset,
    get_device_map,
)
from accelerate import init_empty_weights
from modeling_deepseek import (
    DeepseekV3DecoderLayer,
    DeepseekV3MoE,
    DeepseekV3ForCausalLM,
)
from adapters import DoRALinear
from ademamix import AdEMAMix
from configs import DistillationParams, PathConfig

# Dataset class with Ampere optimizations
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

        # Ensure data is contiguous for better memory access patterns on Ampere
        if not input_data.is_contiguous():
            input_data = input_data.contiguous()
        if not output_data.is_contiguous():
            output_data = output_data.contiguous()

        return input_data, output_data

def prepare_distilled_moe(
    moe,
    selected_experts,
    n_routed_experts,
    n_active_experts,
    distillation_config,
    device="cuda:0",
):
    """Prepare distilled MOE with specified adapter type - optimized for Ampere"""
    moe_config = deepcopy(moe.config)
    # Select adapter class based on type
    AdapterClass = DoRALinear if distillation_config.lora_type.lower() == "dora" else LoRALinear

    # Create new MoE
    moe_config.n_routed_experts = n_routed_experts
    moe_config.num_experts_per_tok = n_active_experts

    # Use CUDA graphs for faster initialization
    with torch.device(device):
        # Use torch.cuda.amp for mixed precision
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pruned_moe = DeepseekV3MoE(moe_config).to(device)
    memory_cleanup()

    # Use contiguous tensors and bfloat16 for better Ampere performance
    pruned_moe.shared_experts.gate_proj = deepcopy(moe.shared_experts.gate_proj).to(device, non_blocking=True)
    pruned_moe.shared_experts.up_proj = deepcopy(moe.shared_experts.up_proj).to(device, non_blocking=True)
    pruned_moe.shared_experts.down_proj = deepcopy(moe.shared_experts.down_proj).to(device, non_blocking=True)

    for param in pruned_moe.parameters():
        param.requires_grad = False

    # Only the gate has full finetuning - use non_blocking transfer
    gate_weight = moe.gate.weight[list(selected_experts)[:n_routed_experts]].detach()
    if not gate_weight.is_contiguous():
        gate_weight = gate_weight.contiguous()
    pruned_moe.gate.weight = torch.nn.Parameter(gate_weight.to(device, non_blocking=True))

    # Update Experts with chosen adapter
    for i, ind in enumerate(list(selected_experts)[:n_routed_experts]):
        expert = deepcopy(moe.experts[ind])
        pruned_moe.experts[i].to_empty(device=device)

        adapter_kwargs = {
            "rank": distillation_config.lora_rank,
            "alpha": distillation_config.lora_alpha,
            "device": device,
        }

        # Apply adapter to each projection - utilize non_blocking transfers
        pruned_moe.experts[i].gate_proj = AdapterClass(expert.gate_proj, **adapter_kwargs).to(device, non_blocking=True)
        pruned_moe.experts[i].up_proj = AdapterClass(expert.up_proj, **adapter_kwargs).to(device, non_blocking=True)
        pruned_moe.experts[i].down_proj = AdapterClass(expert.down_proj, **adapter_kwargs).to(device, non_blocking=True)

    pruned_moe = pruned_moe.to(device, non_blocking=True)

    # Set requires_grad for adapter parameters only
    for name, param in pruned_moe.named_parameters():
        if any(x in name for x in ["lora_", "weight_m"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for param in pruned_moe.gate.parameters():
        param.requires_grad = True

    # Ensure all parameters are on the correct device with optimized transfer
    for name, param in pruned_moe.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
        param.data = param.data.to(device, non_blocking=True)

    return pruned_moe

class MOEDistiller(torch.nn.Module):
    def __init__(
        self,
        distillat,
        device="cuda:0",
        use_amp=True,
    ):
        super().__init__()
        self.distillat = distillat
        self.device = device
        self.use_amp = use_amp

    def forward(self, input_batch):
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return self.distillat(input_batch)
        else:
            return self.distillat(input_batch)

def merge_and_save_model(distillat, layer_idx, n_routed_experts, n_active_experts, path_config):
    """
    Merge adapter weights and save the distilled model to disk.
    """
    # Create save directory if it doesn't exist
    save_dir = path_config.get_distillation_path(n_experts=n_routed_experts, n_active=n_active_experts)
    os.makedirs(save_dir, exist_ok=True)

    # Save the distillat state
    save_path = save_dir + f"/layer_{layer_idx}.pt"

    # Merge all adapters in distillat
    for i in range(n_routed_experts):
        distillat.experts[i].gate_proj = distillat.experts[i].gate_proj.merge_and_unload()
        distillat.experts[i].up_proj = distillat.experts[i].up_proj.merge_and_unload()
        distillat.experts[i].down_proj = distillat.experts[i].down_proj.merge_and_unload()

    # Use PyTorch's improved save mechanism with _use_new_zipfile_serialization
    torch.save(distillat.state_dict(), save_path, _use_new_zipfile_serialization=True)
    print(f"Saved distillat state to: {save_path}")

    destruct_module_optimized(distillat)
    memory_cleanup()

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    gradient_accumulation_steps,
    device,
    epoch,
    scaler=None,
    use_amp=True,
    summary_writer=None,
):
    """Train for one epoch with Ampere optimizations"""
    model.train()
    total_loss = 0
    step = 0

    progress_bar = tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}")

    # Pre-fetch batches to improve data loading speed
    prefetch_stream = torch.cuda.Stream()

    for batch_idx, (input_batch, output_batch) in enumerate(train_loader):
        # Prefetch next batch
        if batch_idx < len(train_loader) - 1:
            with torch.cuda.stream(prefetch_stream):
                next_batch = next(iter(train_loader))
                next_input = next_batch[0][0].to(device, non_blocking=True)
                next_output = next_batch[1][0].to(device, non_blocking=True)

        # Process current batch
        input_batch, output_batch = input_batch[0].to(device, non_blocking=True), output_batch[0].to(device, non_blocking=True)

        # Replace NaN values with 0
        input_batch[torch.isnan(input_batch)] = 0
        output_batch[torch.isnan(output_batch)] = 0

        # Forward pass with automatic mixed precision
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Forward pass
                pred = model(input_batch)

                # Normalize outputs
                m = torch.mean(output_batch, dim=-1, keepdim=True)
                s = torch.std(output_batch, dim=-1, keepdim=True)
                output_batch = (output_batch - m) / s
                pred = (pred - m) / s

                # Compute loss
                loss = torch.nn.functional.mse_loss(pred, output_batch)
                loss = loss / gradient_accumulation_steps  # Scale for gradient accumulation

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Update weights if needed
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                scheduler.step()
                step += 1
        else:
            # Forward pass without AMP
            pred = model(input_batch)

            # Normalize outputs
            m = torch.mean(output_batch, dim=-1, keepdim=True)
            s = torch.std(output_batch, dim=-1, keepdim=True)
            output_batch = (output_batch - m) / s
            pred = (pred - m) / s

            # Compute loss
            loss = torch.nn.functional.mse_loss(pred, output_batch)
            loss = loss / gradient_accumulation_steps  # Scale for gradient accumulation

            # Backward pass
            loss.backward()

            # Update weights if needed
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                scheduler.step()
                step += 1

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        # Log training loss to TensorBoard
        if summary_writer:
            global_step = epoch * len(train_loader) + batch_idx
            summary_writer.add_scalar('training_loss', loss.item() * gradient_accumulation_steps, global_step)

    progress_bar.close()
    return total_loss / len(train_loader)

def validate(model, val_loader, device, use_amp=True):
    """Evaluate on validation set with Ampere optimizations"""
    model.eval()
    total_loss = 0

    progress_bar = tqdm(total=len(val_loader), desc="Validation", summary_writer=None, epoch=0)

    with torch.no_grad():
        for input_batch, output_batch in val_loader:
            input_batch, output_batch = input_batch[0].to(device, non_blocking=True), output_batch[0].to(device, non_blocking=True)

            # Replace NaN values with 0
            input_batch[torch.isnan(input_batch)] = 0
            output_batch[torch.isnan(output_batch)] = 0

            # Forward pass with automatic mixed precision
            if use_amp:
                with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                    # Forward pass
                    pred = model(input_batch)

                    # Normalize outputs
                    m = torch.mean(output_batch, dim=-1, keepdim=True)
                    s = torch.std(output_batch, dim=-1, keepdim=True)
                    output_batch = (output_batch - m) / s
                    pred = (pred - m) / s

                    # Compute loss
                    loss = torch.nn.functional.mse_loss(pred, output_batch)
            else:
                # Forward pass
                pred = model(input_batch)

                # Normalize outputs
                m = torch.mean(output_batch, dim=-1, keepdim=True)
                s = torch.std(output_batch, dim=-1, keepdim=True)
                output_batch = (output_batch - m) / s
                pred = (pred - m) / s

                # Compute loss
                loss = torch.nn.functional.mse_loss(pred, output_batch)

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': loss.item()})
            
            if summary_writer:
                global_step = epoch * len(val_loader) + batch_idx
                summary_writer.add_scalar('validation_loss', loss.item(), global_step)

    progress_bar.close()
    return total_loss / len(val_loader)

def main():
    print("Starting the pipeline...")

    mp.set_start_method("spawn", force=True)

    # Ampere-specific optimizations
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
    torch.backends.cuda.max_split_size_mb = 512
    torch.set_float32_matmul_precision("high")  # Use high precision for Ampere

    parser = argparse.ArgumentParser(description="Distillation pipeline arguments.")
    parser.add_argument("--n_routed_experts", type=int, default=8, help="Number of routed experts.")
    parser.add_argument("--n_active_experts", type=int, default=2, help="Number of active experts.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--end_factor",
        type=float,
        default=0.001,
        help="End factor for learning rate scheduler.",
    )
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., cuda:0 or cpu).",
    )
    parser.add_argument("--min_layer", type=int, default=3, help="Minimum layer index.")
    parser.add_argument("--max_layer", type=int, default=61, help="Maximum layer index.")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--pin_memory", action="store_true", help="Use pin_memory for faster data transfer.")

    args = parser.parse_args()

    path_config = PathConfig()

    n_routed_experts = args.n_routed_experts
    n_active_experts = args.n_active_experts
    learning_rate = args.learning_rate
    end_factor = args.end_factor
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    device = args.device
    weights_location = "deepseek_v3/"
    min_layer = args.min_layer
    max_layer = args.max_layer
    use_amp = args.use_amp  # Automatic Mixed Precision for Ampere
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    pin_memory = args.pin_memory

    # For multi-GPU setup
    dev = device.split(":")[1]
    print(f"Using GPU: {dev}")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev  # limit all gpu to only one, to enable parallel run

    device = "cuda:0"  # now only cuda 0 is available

    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    params = DistillationParams(
        n_epochs=15,
        n_train_batch=232,
        n_batch=256,
        batch_size=batch_size,
        max_length=512,
        gradient_accumulation_steps=gradient_accumulation_steps,
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

    # Load Model Config
    weight_map, config = load_model_config(weights_location)

    # Create empty model
    with init_empty_weights():
        model = DeepseekV3ForCausalLM(config)

    destruct_module_optimized(model)
    memory_cleanup()

    # Process each layer
    for layer_idx in range(min_layer, max_layer):
        print(f"\nProcessing layer {layer_idx}...")

        # Create directories for logging and checkpoints
        os.makedirs(path_config.log_dir, exist_ok=True)
        os.makedirs(path_config.checkpoint_dir, exist_ok=True)

        # Setup logfile for this layer
        log_file = f"{path_config.log_dir}/layer_{layer_idx}_{n_routed_experts}a{n_active_experts}.txt"

        print("Loading dataset...")
        full_dataset = IntermediateStateDataset(path_config, layer_idx, 0, params.n_batch)

        # Split dataset
        train_size = params.n_train_batch
        val_size = params.n_batch - params.n_train_batch

        # Generate indices for splits
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        print("Dataset loaded and split.")

        # Define dataloaders with Ampere optimizations
        print("Creating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=pin_memory,  # Use pinned memory for faster CPU-to-GPU transfer
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=pin_memory,
        )
        print("Dataloaders created.")

        print("Initializing model...")
        device_map = get_device_map(layer_idx, weight_map, device)
        model.model.layers[layer_idx] = model.model.layers[layer_idx].to_empty(device=device)

        # Use CUDA streams for parallel loading
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for i, weight_name in enumerate(tqdm(device_map)):
                weight = load_weight(weights_location, weight_name, weight_map, "cpu")
                weight = weight.to(device, non_blocking=True)
                rsetattr(model, weight_name, torch.nn.Parameter(weight, requires_grad=False))
                if i % 50 == 0:  # Reduced frequency of memory cleanup
                    memory_cleanup()

        # Synchronize stream to ensure weights are loaded
        torch.cuda.current_stream().wait_stream(stream)

        model.model.layers[layer_idx] = model.model.layers[layer_idx].to(device, non_blocking=True)
        memory_cleanup()

        # Load expert activation data
        with open(f"{path_config.expert_activation_dir}/layer_{layer_idx}.pickle", "rb") as f:
            act = pickle.load(f)

        v, c = np.unique(act, return_counts=True)
        selected_experts = np.flip(np.argsort(c))

        # Prepare distilled MoE with Ampere optimizations
        distillat = prepare_distilled_moe(
            model.model.layers[layer_idx].mlp,
            selected_experts,
            n_routed_experts,
            n_active_experts,
            params,
            device=device,
        )

        # Free memory
        destruct_module_optimized(model)
        memory_cleanup()

        # Create MOE distiller model with AMP support
        moe_distiller = MOEDistiller(distillat, device, use_amp=use_amp)
        moe_distiller.to(device, non_blocking=True)
        print("Model initialized.")

        # Setup optimizer
        optimizer = AdEMAMix(
            filter(lambda p: p.requires_grad, moe_distiller.parameters()),
            lr=params.learning_rate,
            betas=(0.95, 0.999, 0.9999),
            alpha=128,
        )

        # Setup learning rate schedulers
        total_steps = (params.n_epochs * params.n_train_batch) // params.gradient_accumulation_steps

        # Cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - 64,  # Subtract warmup steps
            eta_min=params.learning_rate * 0.1,
        )

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=64
        )

        # Combine both schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[64]
        )

        # Training loop
        print("Starting training...")

        best_val_loss = float('inf')
        best_model_path = f"{path_config.checkpoint_dir}/moe_distiller_layer_{layer_idx}_{n_routed_experts}a{n_active_experts}_best.pt"
        patience = 2  # Early stopping patience
        patience_counter = 0

        # Open log file
        with open(log_file, 'w') as f:
            f.write(f"Training layer {layer_idx} with {n_routed_experts} routed experts and {n_active_experts} active experts\n")
            f.write(f"Learning rate: {learning_rate}, LoRA rank: {lora_rank}, LoRA alpha: {lora_alpha}\n")
            f.write(f"Using AMP: {use_amp}, Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}\n\n")
            f.write("epoch,train_loss,val_loss,time\n")

        for epoch in range(params.n_epochs):
            start_time = time.time()

            # Train one epoch with Ampere optimizations
            train_loss = train_one_epoch(
                moe_distiller,
                train_loader,
                optimizer,
                scheduler,
                params.gradient_accumulation_steps,
                device,
                epoch,
                scaler,
                use_amp
            )

            # Validate with Ampere optimizations
            val_loss = validate(moe_distiller, val_loader, device, use_amp)

            epoch_time = time.time() - start_time

            # Log results
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Time = {epoch_time:.2f}s")

            with open(log_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{epoch_time:.2f}\n")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': moe_distiller.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, best_model_path, _use_new_zipfile_serialization=True)
                print(f"Saved best model with validation loss: {val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        print("Training finished.")

        # Load best model for merging and saving
        checkpoint = torch.load(best_model_path)
        moe_distiller.load_state_dict(checkpoint['model_state_dict'])

        # Merge and save model
        merge_and_save_model(
            moe_distiller.distillat,
            layer_idx,
            n_routed_experts,
            n_active_experts,
            path_config
        )

        # Explicit cleanup after each layer
        torch.cuda.empty_cache()
        memory_cleanup()

        print(f"Layer {layer_idx} completed.")

    print("All layers processed. Pipeline completed.")

if __name__ == "__main__":
    main()
