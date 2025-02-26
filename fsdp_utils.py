import copy
import functools
import gc
import math
import os
import sys
import time
import types
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Dict, List

import bitsandbytes as bnb
import safetensors
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from accelerate import init_empty_weights
from accelerate.utils import set_seed

# Model loading
from bitsandbytes.nn import Linear4bit, Params4bit
from fastcore.parallel import parallel

# Argument parsing
from fastcore.script import Param, bool_arg, call_parse
from packaging.version import parse
from safetensors.torch import save_file

# Torch + distributed training
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)

# FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, hub

from model_utils import rsetattr
from torch_utils import memory_cleanup

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
except ImportError:
    HQQLinear = None
    pass


class LORA(nn.Module):
    def __init__(self, base_layer, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.base_layer = base_layer
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        device = next(base_layer.parameters()).device
        lora_A = nn.Linear(base_layer.in_features, lora_rank, bias=False, device=device, dtype=dtype)
        lora_B = nn.Linear(lora_rank, base_layer.out_features, bias=False, device=device, dtype=dtype)
        lora_B.weight.data.zero_()
        self.lora_AB = nn.Sequential(lora_A, lora_B)
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = self.lora_alpha / lora_rank

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        result = result.clone()
        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(next(iter(self.lora_AB)).weight.dtype)
        output = self.lora_AB(self.lora_dropout(x))
        if requires_conversion:
            output = output.to(expected_dtype)
        output = output * self.scaling
        result += output
        return result


class DORALayer(nn.Module):
    "Same as LORA but also returnes weight norm. This will be wrapped as a single FSDP unit"
    def __init__(self, in_features, out_features, lora_rank, device, dtype, *args, **kwargs):
        super().__init__()
        # Init LoRA layers.
        std_dev = 1 / torch.sqrt(torch.tensor(lora_rank).float()).to(device=device, dtype=torch.bfloat16)
        
        lora_A_param = nn.Parameter(torch.randn(lora_rank, in_features, device=device, dtype=dtype) * std_dev)
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
        setattr(self.lora_A, "weight", lora_A_param)

        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)
        self.lora_B.weight.data.zero_()

    def forward(self, x, frozen_weight):
        output = self.lora_B(self.lora_A(x))
        column_norm = (frozen_weight + self.lora_B.weight @ self.lora_A.weight).norm(p=2, dim=1).detach()
        return output, column_norm


class MagnitudeLayer(nn.Module):
    "FSDP doesn't work with nn.ParameterDict hence this module: https://github.com/pytorch/pytorch/issues/79605"
    def __init__(self, vector_data, device, dtype):
        super().__init__()
        self.magnitude = nn.Parameter(vector_data.to(device=device, dtype=dtype))

    def forward(self, x):
        return x * self.magnitude.view(1, 1, -1)


class HQQDORA(nn.Module):
    def __init__(self, base_layer, lora_rank, device, *args, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        
        self.magnitude_layer = MagnitudeLayer(self.base_layer.dora_scale.clone().to(dtype=dtype), device, dtype)
        self.base_layer.dora_scale = None
        torch.cuda.empty_cache()

        # Init DORA layers.
        self.dora_layer = DORALayer(
            base_layer.in_features,
            base_layer.out_features,
            lora_rank,
            device,
            dtype,
            *args,
            **kwargs,
        )

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        result = result.clone()

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(self.dora_layer.lora_A.weight.dtype)
        output, column_norm = self.dora_layer(x, self.base_layer.dequantize())
        if requires_conversion:
            output = output.to(expected_dtype)

        result += output
        result = result / column_norm.view(1, 1, -1)  # unit vector result.
        result = self.magnitude_layer(result)  # rescaled result.
        return result


def quantize(module):
    
    quant_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        # offload_meta=True,
        view_as_float=True  # Important for FSDP
    )

    module = HQQLinear(
        module,
        quant_config=quant_config,
        compute_dtype=torch.bfloat16,
        device='cuda',
        initialize=True,
        del_orig=True
    )
    
    ## Moving to cpu
    module.device="cpu"
    module.meta['scale']= module.meta['scale'].to('cpu')
    module.meta['zero']= module.meta['zero'].to('cpu')
    
    module.set_backend(HQQBackend.PYTORCH)
    for name, parameters in module.named_parameters():
        rsetattr(module, name, torch.nn.Parameter(parameters.to('cpu'), requires_grad=False))
        
    return module

def load_quantize_parallel(named_module, model, is_dora=True, pbar=None):
    name, module = named_module
    if isinstance(module, torch.nn.Linear):
        if not('lm_head' in name): ## keeping lm head as it is not quantized
            if is_dora:
                dora_scale = module.weight.norm(p=2, dim=1).to(dtype=torch.bfloat16, device="cpu")
            rsetattr(model, name, quantize(module))
            if is_dora:
                rsetattr(model, name + ".dora_scale", dora_scale)
    if pbar is not None:
        pbar.update(1)
        
def apply_dora_to_model(model, target_modules, lora_rank, lora_alpha, lora_dropout):
    print('Applying DORA to model...')
    for name, module in tqdm(model.named_modules()):
        if isinstance(module, HQQLinear):
            cond=False
            for target in target_modules:
                if target in name:
                    cond=True
            if cond:
                rsetattr(
                    model,
                    name,
                    HQQDORA(module,lora_rank, device="cpu")
                )
                
    print('Setting trainable parameters policy')
    for name, params in tqdm(model.named_parameters()):
        if any([lora_name in name for lora_name in ['lora_AB', 'lora_A', 'lora_B', 'magnitude']]):
            params.requires_grad = True
        else:
            params.requires_grad = False
    


def get_wrapping_policy(
    LLAMA_ATTENTION_CLASSES=None,
    LlamaDecoderLayer=None,
    LlamaMLP=None,
):
    def lambda_policy_fn(module):
        return (isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module)) or (isinstance(module, (DORALayer, MagnitudeLayer)))

    def self_attn_policy_fn(module):
        return isinstance(module, tuple(LLAMA_ATTENTION_CLASSES.values()))

    def mlp_policy_fn(module):
        return isinstance(module, (LlamaMLP,))

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer,),
    )

    policies=[
        lambda_policy,
        transformer_wrap_policy,
        self_attn_policy,
        mlp_policy
    ]
    return functools.partial(_or_policy, policies=policies)

def fsdp_hqq_dora_model_for_causal_lm(
    model_name,
    target_modules=[],
    lora_rank=8,
    lora_alpha=8,
    lora_dropout=0.1,
    n_workers=16
):
    print('Loading HF Model to CPU')
    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        offload_buffers=True
    )
    
    ## Ensuring that the model is all on cpu
    model = model.to('cpu')
    memory_cleanup()
    
    print('Quantizing linear layers')
    modules_to_process = list(model.named_modules())
    total = len(modules_to_process)
    pbar = tqdm(total=total, desc="Quantizing modules")
    
    parallel(
        load_quantize_parallel,
        modules_to_process,
        model=model,
        is_dora=True,
        pbar=pbar,
        n_workers=n_workers,
        threadpool=True,
    )
    
    ## Ensuring all parameters are on cpu
    print('Making sure every thing is on cpu')
    for name, parameter in model.named_parameters():
        rsetattr(model, name, parameter.to('cpu'))
    model=model.to('cpu')
    memory_cleanup()

    print('Applying dora weights to quantized layers')
    apply_dora_to_model(model, target_modules, lora_rank, lora_alpha, lora_dropout)
    
    
    print('Making sure every thing is on cpu, better twice than once')
    for name, parameter in model.named_parameters():
        rsetattr(model, name, parameter.to('cpu'))
    model=model.to('cpu')
    memory_cleanup()
    return model