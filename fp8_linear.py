import torch.nn.functional as F
from typing import Tuple, Optional
import torch

def highest_power_of_2_divisor(n):
    if n == 0:
        return 0

    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]  # Powers of 2 below 128

    highest_divisor = 0
    for power in reversed(powers_of_2):  # Iterate in descending order
        if n % power == 0:
            highest_divisor = power
            break  # Found the highest, no need to continue

    return highest_divisor

def act_quant(x, bloc_size=128):
    # Calculate actual block sizes based on input dimensions
    h_blocks = (x.shape[0] + bloc_size - 1) // bloc_size
    w_blocks = (x.shape[1] + bloc_size - 1) // bloc_size
    
    # Calculate padding needed
    pad_h = h_blocks * bloc_size - x.shape[0]
    pad_w = w_blocks * bloc_size - x.shape[1]
    
    # Pad the input if necessary
    if pad_h > 0 or pad_w > 0:
        x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    else:
        x_padded = x
    
    # Reshape into blocks
    blocks = x_padded.view(h_blocks, bloc_size, w_blocks, bloc_size)
    blocks = blocks.permute(0, 2, 1, 3)
    
    # Calculate scaling factors
    s = torch.amax(blocks.abs(), dim=(2, 3)) / 448.0
    
    # Scale blocks
    scaled_blocks = blocks / s.unsqueeze(-1).unsqueeze(-1)
    
    # Reshape back and convert to float8
    x_quant = scaled_blocks.permute(0, 2, 1, 3).reshape(x_padded.shape)
    
    # Remove padding if it was added
    if pad_h > 0 or pad_w > 0:
        x_quant = x_quant[:x.shape[0], :x.shape[1]]
    
    return x_quant.to(torch.float8_e4m3fn), torch.squeeze(s)

def weight_dequant(x, s, bloc_size=128, dtype=torch.bfloat16):
    # Calculate actual block sizes
    h_blocks = (x.shape[0] + bloc_size - 1) // bloc_size
    w_blocks = (x.shape[1] + bloc_size - 1) // bloc_size
    
    # Calculate padding needed
    pad_h = h_blocks * bloc_size - x.shape[0]
    pad_w = w_blocks * bloc_size - x.shape[1]
    
    # Pad the input if necessary
    if pad_h > 0 or pad_w > 0:
        x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    else:
        x_padded = x
    
    # Reshape into blocks
    blocks = x_padded.to(torch.bfloat16).view(h_blocks, bloc_size, w_blocks, bloc_size)
    blocks = blocks.permute(0, 2, 1, 3)
    
    # Apply scaling factors
    dequant = blocks * s.unsqueeze(-1).unsqueeze(-1)
    
    # Reshape back
    result = dequant.permute(0, 2, 1, 3).reshape(x_padded.shape)
    
    # Remove padding if it was added
    if pad_h > 0 or pad_w > 0:
        result = result[:x.shape[0], :x.shape[1]]
    
    return result

class FP8Linear(torch.nn.Module):
    dtype = torch.bfloat16
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        device = "cuda:0"
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
               
        self.register_parameter("weight", None)
        self.register_parameter("weight_scale_inv", None)
        self.register_parameter("bias", None)
        
        # Initialize weight and weight_scale_inv from an empty tensor
        if device != "meta":
            weight = torch.empty((out_features, in_features), device=device, dtype=FP8Linear.dtype)
            torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            self.weight_quant(weight)

            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=FP8Linear.dtype), requires_grad=False)
                
    def weight_dequant(self):
        return weight_dequant(self.weight, self.weight_scale_inv, dtype=FP8Linear.dtype)

    def weight_quant(self, x):        
        weight, weight_scale_inv = act_quant(x, 128)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        
    def forward(self, x):        
        return F.linear(x, self.weight_dequant(), self.bias).to(dtype=x.dtype, device=x.device)