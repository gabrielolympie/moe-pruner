import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import logging


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class FP8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        device="cuda:0",
        fp8_format="e4m3"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=torch.float32)) if bias else None
        self.device = device
        self.fp8_format = fp8_format.lower()
        self.max_value = self._get_max_value()

    def _get_max_value(self):
        if self.fp8_format == "e4m3":
            return 448.0
        elif self.fp8_format == "e5m2":
            return 57344.0
        else:
            raise ValueError(f"Unsupported FP8 format: {self.fp8_format}")

    def to_linear(self, dtype=torch.bfloat16):
        linear_layer = torch.nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, device=self.device)
        linear_layer.weight.data = self._weight_unquantized(dtype)
        if self.bias is not None:
            linear_layer.bias.data = self.bias.to(dtype)
        return linear_layer

    def from_linear(self, layer):
        self.from_weight_matrix(layer.weight.data)
        if layer.bias is not None:
            self.bias = torch.nn.Parameter(layer.bias.data.to(torch.float32))
        layer.weight = torch.nn.Parameter(torch.empty_like(layer.weight, device='meta'), requires_grad=False)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(torch.empty_like(layer.bias, device='meta'), requires_grad=False)


    def from_weight_matrix(self, weights):
        self.weight = bnb.nn.Params4bit(weights.to(self.device), requires_grad=False, quant_type='fp4').to(self.device)


    def _weight_unquantized(self, dtype=torch.bfloat16):
        return bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state).to(dtype)

    def forward(self, x):
        x = x.to(torch.float32)
        weight = self._weight_unquantized(torch.float32)
        out = F.linear(x, weight, self.bias)

        out = out.to(x.dtype)
        out = torch.clamp(out, min=-self.max_value, max=self.max_value)
        out = torch.nan_to_num(out, nan=self.max_value, posinf=self.max_value, neginf=-self.max_value)

        return out.to(torch.bfloat16)