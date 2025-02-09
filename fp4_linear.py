import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class BNBLinear4bit(nn.Module):
    """
    A Linear layer that uses 4-bit quantization for weights using bitsandbytes.

    This class provides a similar interface to `torch.nn.Linear`, but quantizes
    the weight matrix to 4 bits using bitsandbytes for reduced memory usage.
    It supports loading from and converting back to standard `torch.nn.Linear` layers.
    It includes handling of potential NaN and Inf values after multiplication.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        device (torch.device, optional): The device on which the layer will be allocated.
            Default: ``"cuda:0"``.
        quant_type (str, optional): The quantization type to use.  Supports 'fp4' and 'nf4'.
            Default: ``"nf4"``.
        compute_dtype (torch.dtype, optional): The data type for computation.
            Default: ``torch.bfloat16``.  Can improve stability to use float32.
        compress_statistics (bool, optional): Whether to compress the statistics used for quantization.
            Default: ``True``.

    Attributes:
        in_features (int): The input feature size.
        out_features (int): The output feature size.
        weight (bnb.nn.Params4bit): The quantized 4-bit weight parameter.
        bias (torch.nn.Parameter or None):  The bias parameter, if ``bias=True``, otherwise ``None``.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = "cuda:0",  # Corrected to torch.device
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.bfloat16,
        compress_statistics: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        # Initialize the weight as a 4-bit quantized parameter using bitsandbytes
        self.weight = bnb.nn.Params4bit(
            torch.empty(out_features, in_features, device=device, dtype=torch.float32),
            requires_grad=False,
            quant_type=quant_type,
            compress_statistics=compress_statistics
        )
        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=torch.float32)
            )
        else:
            self.bias = None

        self.device = device

        if quant_type not in ['nf4', 'fp4']:
            raise ValueError("quant_type must be 'nf4' or 'fp4'")


    def to_linear(self, dtype=torch.bfloat16) -> nn.Linear:
        """
        Converts the `BNBLinear4bit` layer to a standard `torch.nn.Linear` layer.

        The 4-bit quantized weights are dequantized and used to populate the
        weight matrix of the new `nn.Linear` layer.

        Args:
            dtype (torch.dtype, optional): The desired data type for the new Linear layer.
                Default: `torch.bfloat16`.

        Returns:
            torch.nn.Linear: A standard `torch.nn.Linear` layer with dequantized weights.
        """
        linear_layer = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None, device=self.device
        )
        linear_layer.weight.data = self.get_original_weight(dtype)
        if self.bias is not None:
            linear_layer.bias.data = self.bias.to(dtype)  # Ensure bias is also cast to dtype
        return linear_layer

    def from_linear(self, linear_layer: nn.Linear):
        """
        Loads the weights and bias from a standard `torch.nn.Linear` layer.

        The weights from the provided `linear_layer` are quantized to 4-bit
        and stored in the `weight` attribute. The bias is copied directly (if present).
        The weights and bias of linear_layer are set to empty.

        Args:
            linear_layer (torch.nn.Linear): The standard `torch.nn.Linear` layer to load from.
        """
        self.from_weight_matrix(linear_layer.weight.data)
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.to(torch.float32))

        linear_layer.weight = nn.Parameter(torch.empty_like(linear_layer.weight, device='meta'), requires_grad=False)
        if linear_layer.bias is not None:
            linear_layer.bias = nn.Parameter(torch.empty_like(linear_layer.bias, device='meta'), requires_grad=False)

    def from_weight_matrix(self, weights: torch.Tensor):
        """
        Initializes the layer's 4-bit quantized weight from a full-precision weight matrix.

        Args:
            weights (torch.Tensor): The full-precision weight matrix.
        """
        self.weight = bnb.nn.Params4bit(
            weights.to(self.device),
            requires_grad=False,
            quant_type=self.weight.quant_type,
            compress_statistics=self.weight.compress_statistics,
        ).to(self.device)
        # self.weight._quantize(self.device)

    def get_original_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """
        Dequantizes the 4-bit weight and returns it as a full-precision tensor.

        Args:
            dtype (torch.dtype, optional): The desired data type for the dequantized weight.
                Default: ``torch.bfloat16``.

        Returns:
            torch.Tensor: The dequantized weight matrix.
        """
        return bnb.functional.dequantize_4bit(
            self.weight.data, self.weight.quant_state
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Dequantize the weight to the compute dtype for the forward pass
        weight = self.get_original_weight(self.compute_dtype)

        # Perform the linear transformation using the dequantized weight and bias
        x = x.to(self.compute_dtype)
        out = F.linear(x, weight, self.bias)

        #clamp and nan values handling
        if self.weight.quant_type == 'nf4':
          out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.weight.quant_type == 'fp4':
            out = torch.clamp(out, min=-64000, max=64000)
            out = torch.nan_to_num(out, nan=64000, posinf=64000, neginf=-64000)

        # Return output in the input precision
        return out.to(x.dtype)