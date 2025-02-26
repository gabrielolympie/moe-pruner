import math
import _pickle as pickle
from copy import deepcopy
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fp8_linear import FP8Linear


class AdapterBase(nn.Module):
    """Base class for LoRA and DoRA adapters"""

    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.05, device="cuda:0"):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        # Get the dtype and device from the base layer
        self.dtype = torch.bfloat16
        self.device = device

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def merge_and_unload(self):
        raise NotImplementedError


class DoRALinear(AdapterBase):
    def __init__(
        self,
        base_layer,
        rank=8,
        alpha=16,
        dropout=0.05,
        dora_simple=True,
        device="cuda:0",
    ):
        super().__init__(base_layer, rank, alpha, dropout, device)

        self.weight = base_layer.weight
        self.dora_simple = dora_simple

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.empty((rank, base_layer.in_features), dtype=self.dtype, device=self.device))

        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, rank), dtype=self.dtype, device=self.device))

        # Initialize magnitude decomposition
        self.weight_m = nn.Parameter(torch.empty((base_layer.out_features, 1), dtype=self.dtype, device=self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Initialize magnitude component from base weights
        with torch.no_grad():
            base_norm = torch.linalg.norm(
                self.base_layer.weight_dequant().to(self.device, dtype=self.dtype),
                dim=1,
                keepdim=True,
            )
            self.weight_m.data.copy_(base_norm)

    def forward(self, x):
        # Get base weight and ensure it's on the correct device
        base_weight = self.base_layer.weight_dequant().to(self.device, dtype=self.dtype)

        # Calculate new weight with LoRA
        new_weight = base_weight + (self.lora_B @ self.lora_A) * self.scaling

        # Calculate norm scales
        if self.dora_simple:
            base_norm = torch.linalg.norm(new_weight, dim=1, keepdim=True).detach().to(dtype=self.dtype)
        else:
            base_norm = torch.linalg.norm(new_weight, dim=1, keepdim=True).to(dtype=self.dtype)

        norm_scale = self.weight_m / base_norm

        # Apply dropout
        dropout_x = self.dropout(x)

        # Compute output with decomposed scaling
        base_out = self.base_layer(dropout_x)
        lora_out = F.linear(dropout_x, self.lora_B @ self.lora_A) * self.scaling

        result = (base_out + lora_out) * norm_scale.t()
        if self.base_layer.bias is not None:
            result += self.base_layer.bias.to(self.device)

        return result

    def merge_and_unload(self):
        """Merge DoRA weights with base weights and return new layer"""
        new_weight = self.base_layer.weight_dequant().to(dtype=self.dtype)

        new_weight = new_weight.to("cpu") + (self.lora_B.to("cpu") @ self.lora_A.to("cpu")) * self.scaling

        # Apply magnitude scaling
        base_norm = torch.linalg.norm(new_weight, dim=1, keepdim=True)

        norm_scale = self.weight_m.to("cpu") / base_norm

        # print(new_weight.shape, norm_scale.shape)
        merged_weight = new_weight * norm_scale

        new_layer = FP8Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.device,
        )

        new_layer.weight_quant = merged_weight
        if self.base_layer.bias is not None:
            new_layer.bias = nn.Parameter(self.base_layer.bias.clone().to(self.device))

        return new_layer
