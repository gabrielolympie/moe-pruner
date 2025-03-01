import torch
import torch.nn as nn

class DORALayer(nn.Module):
    "Same as LORA but also returnes weight norm. This will be wrapped as a single FSDP unit"
    def __init__(self, in_features, out_features, lora_rank, device, dtype, *args, **kwargs):
        super().__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(lora_rank).float())
        lora_A_param = nn.Parameter(torch.randn(lora_rank, in_features).to(device=device, dtype=dtype)*std_dev)
        
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
        setattr(self.lora_A, "weight", lora_A_param)
        
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)
        self.lora_B.weight.data.zero_()
    
    def forward(self, x, frozen_weight):
        output = self.lora_B(self.lora_A(x))
        column_norm = (frozen_weight + self.lora_B.weight @ self.lora_A.weight).norm(p=2, dim=1).detach()
        return output, column_norm

class MagnitudeLayer(nn.Module):
    def __init__(self, vector_data, device, dtype):
        super().__init__()
        self.magnitude = nn.Parameter(vector_data.to(device=device, dtype=dtype))
        
    def forward(self, x):
        return x * self.magnitude.view(1,1,-1)

class DoRAAdapter(nn.Module):
    def __init__(
        self,
        base_layer,
        lora_rank,
        dropout=0.1,
    ):
        super().__init__()
        self.dtype= base_layer.weight.dtype
        self.device=base_layer.weight.device
        
        self.lora_rank=lora_rank
        self.dropout=dropout
        
        self.base_layer=base_layer
        
        self.dora_layer=DORALayer(base_layer.in_features, base_layer.out_features, lora_rank,  device=self.device, dtype=self.dtype)
        self.magnitude_layer=MagnitudeLayer(base_layer.weight.norm(dim=1), self.device, self.dtype)
        
        for params in base_layer.parameters():
            params.requires_grad=False
        
    def forward(self, x):
        output=self.base_layer(x)
        dora_output, column_norm = self.dora_layer(x, self.base_layer.weight)
        output += dora_output
        output = output / (column_norm + 1e-8)
        output = self.magnitude_layer(output)
        return output.to(dtype=self.dtype, device=self.device)
    
    def merge_and_unload(self):
        merged_layer = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.device,
            dtype=self.dtype
        )
        
        if self.base_layer.bias is not None:
            merged_layer.bias.data.copy_(self.base_layer.bias.data)
        
        lora_update = self.dora_layer.lora_B.weight @ self.dora_layer.lora_A.weight
        base_weights = self.base_layer.weight.clone()
        
        unnormalized_weights = base_weights + lora_update
        
        column_norms = unnormalized_weights.norm(p=2, dim=1, keepdim=True)
        normalized_weights = unnormalized_weights / column_norms

        final_weights = normalized_weights * self.magnitude_layer.magnitude.view(-1, 1)
        merged_layer.weight.data.copy_(final_weights)
        return merged_layer