from torch.optim.lr_scheduler import _LRScheduler
from utils.fp8_linear import act_quant, weight_dequant
from awq.modules.linear import WQLinear_GEMM
from safetensors import safe_open
from datasets import load_dataset
from tqdm.auto import tqdm
import functools
import torch
import math
import gc

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))

def rhasattr(obj, attr):
    try:
        a = rgetattr(obj, attr)
        return True
    except AttributeError:
        return False
    
def quant(x):
    absmax = torch.max(torch.abs(x), axis=-1, keepdim=True).values
    x = x/absmax
    return x.to(torch.float8_e4m3fn), absmax

def dequant(x, absmax):
    return x.to(torch.bfloat16) * absmax

def save_quant(x, base_path):
    x, amax=quant(x)
    x={
        "weight": x,
        "amax": amax
    }
    torch.save(x, base_path.replace(".pt", "weight.pt"))

def load_quant(base_path):
    x = torch.load(base_path.replace(".pt", "weight.pt"))
    x = dequant(x['weight'], x["amax"])
    return x

def destruct_module_optimized(module: torch.nn.Module) -> torch.nn.Module:
    """Efficiently destroy module and clear memory."""
    module.to_empty(device="meta")
    gc.collect()
    torch.cuda.empty_cache()

def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def count_parameters(model):
    frozen_params = 0
    non_frozen_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            non_frozen_params += param.numel()
        else:
            frozen_params += param.numel()

    total_params = frozen_params + non_frozen_params

    print(f"{'Parameter Type':<20} {'Count':<10}")
    print(f"{'='*20} {'='*10}")
    print(f"{'Frozen Parameters':<20} {frozen_params:<10,}")
    print(f"{'Non-Frozen Parameters':<20} {non_frozen_params:<10,}")
    print(f"{'Total Parameters':<20} {total_params:<10,}")

def load_weight(
    weights_location: str,
    weight_name: str,
    weight_map: dict,
    device: int,
) -> torch.Tensor:
    """Load weight with non-blocking CUDA operations."""
    weight_file = weight_map[weight_name]
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        file_path = f"{weights_location}/{weight_file}"
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            tensor_slice = f.get_slice(weight_name)
            shape = tensor_slice.get_shape()
            if len(shape) > 1:
                vocab_size, hidden_dim = shape
                tensor = torch.nn.Parameter(tensor_slice[:, :hidden_dim].to(device), requires_grad=False).to(device)
            else:
                tensor = torch.nn.Parameter(tensor_slice[:].to(device), requires_grad=False).to(device)
            return tensor
        
def convert_meta_model_to_awq(model, config, device):
    w_bit=config.quantization_config['bits']
    group_size=config.quantization_config['group_size']
    modules_to_not_convert=['lm_head']
    if config.quantization_config['modules_to_not_convert'] is not None:
        modules_to_not_convert+=config.quantization_config['modules_to_not_convert']
    for name, module in tqdm(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            
            cond = True
            if modules_to_not_convert is not None:
                for elt in modules_to_not_convert:
                    if elt in name:
                        cond=False

            if cond:
                rsetattr(
                    model,
                    name,
                    WQLinear_GEMM(
                        w_bit,
                        group_size,
                        module.in_features,
                        module.out_features,
                        module.bias,
                        device
                    ).to_empty(device="meta")
                )
    return model
        
def get_nonreasoning_dataset(tokenizer, generation_config):
    calibration = load_dataset("cognitivecomputations/dolphin-r1", "nonreasoning", cache_dir="../dolphin-r1")["train"]

    def filter_function(example):
        if example["overall_quality"] is not None and example["overall_quality"] == 5:
            return True
        if example["score"] is not None and example["score"] >= 0.16:
            return True
        return False

    calibration = calibration.filter(filter_function)

    data = calibration["messages"][: generation_config.batch_size * generation_config.n_batch]

    train_dataset = [
        tokenizer.apply_chat_template(elt, tokenize=False, add_generation_prompt=False)
        for elt in tqdm(data, desc="Preparing dataset")
    ]
    return train_dataset

def load_weights(model, model_name, weight_map, target_modules, device):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for i, weight_name in enumerate(tqdm(weight_map)):
            if any([target in weight_name for target in target_modules]):
                rsetattr(
                    model,
                    weight_name,
                    load_weight(
                        model_name,
                        weight_name,
                        weight_map,
                        device=device
                    )
                )
            if i % 20000 == 0:
                memory_cleanup()
    stream.synchronize()
    return model

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            decay_factor = (1 - self.min_lr) * cosine_decay + self.min_lr
            return [base_lr * decay_factor for base_lr in self.base_lrs]