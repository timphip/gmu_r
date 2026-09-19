import json
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from huggingface_hub import HfApi
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.quantizers import HfQuantizer
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

FP4_CODEBOOK = torch.tensor([ 0.0000,  0.0052,  0.6667,  1.0000,  0.3333,  0.5000,  0.1667,  0.2500,
         0.0000, -0.0052, -0.6667, -1.0000, -0.3333, -0.5000, -0.1667, -0.2500])

def get_module_from_name(model: nn.Module, param_name: str):
    parts = param_name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]

# Implement INT8 Symmetric Linear layer
class FP4SymmetricLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, bias, dtype=torch.float32,blocksize=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocksize = blocksize
        # Initialize all buffers immediately (avoid None state)
        self.register_buffer("codebook", FP4_CODEBOOK.to(dtype))
        self.register_buffer("weight", torch.zeros((out_features * in_features // 2, 1), dtype=torch.uint8))
        self.register_buffer("scale",
            torch.ones((out_features, 1), dtype=dtype))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
           self.bias = None

        self._is_quantized = False
    def _quantize_to_fp4(self, weight, blocksize=64):
        """
        FP4 block quantization implementation (shared scale per block)
        
        Args:
            weight: torch.Tensor, input weights [out_features, in_features]
            blocksize: int, block size (default 64)
        
        Returns:
            weight_uint8: torch.uint8, compressed quantized weights [total_elements // 2, 1]
            scale: torch.float16, scale factors per block [n_blocks]
        """
        # 1. Flatten and split into blocks
        weight_flat = weight.flatten()  # [out_features * in_features]
        n_elements = weight_flat.numel()
        n_blocks = (n_elements + blocksize - 1) // blocksize
        
        # Pad incomplete blocks (ensure divisibility)
        padded_size = n_blocks * blocksize
        padded_weight = torch.nn.functional.pad(weight_flat, (0, padded_size - n_elements))
        blocks = padded_weight.view(n_blocks, blocksize)  # [n_blocks, blocksize]
        
        # 2. Calculate absmax and scale per block
        absmax = torch.max(torch.abs(blocks), dim=1)[0].clamp(min=1e-5)  # [n_blocks]
        scale = absmax / torch.max(torch.abs(self.codebook))  # Map to codebook range
        
        # 3. Normalize and map to codebook indices
        normalized = blocks / scale[:, None]  # [n_blocks, blocksize]
        distances = torch.abs(normalized.unsqueeze(-1) - self.codebook.view(1, 1, -1))
        indices = torch.argmin(distances, dim=-1).to(torch.uint8)  # [n_blocks, blocksize]
        
        # 4. Compress to uint8 (two 4-bit indices per byte)
        indices_flat = indices.flatten()[:n_elements]  # Remove padding
        compressed = (indices_flat[::2] << 4) | indices_flat[1::2]  # [n_elements // 2]
        
        # Adjust to official shape [n_elements // 2, 1]
        return compressed.view(-1, 1), scale


    def forward(self, x):
        # 1. Calculate actual required element count (avoid inflation)
        n_elements = self.out_features * self.in_features
        
        # 2. Decompress 4-bit indices (precise count control)
        compressed = self.weight.view(-1)  # Shape [n_elements//2]
        indices = torch.cat([
            (compressed >> 4) & 0x0F,  # High 4 bits
            compressed & 0x0F           # Low 4 bits
        ])[:n_elements].to(torch.long)  # Shape [n_elements]
        
        # 3. Safety check
        assert indices.max() < 16, f"Index value {indices.max()} exceeds codebook range"
        
        # 4. Dequantize (direct 1D indexing)
        dequant_flat = self.codebook[indices.to(x.device)]  # Shape [n_elements]
        
        # 5. Apply block scaling
        block_ids = torch.arange(n_elements, device=x.device) // self.blocksize
        scales = self.scale[block_ids].squeeze(-1)  # Shape [n_elements]
        
        # 6. Reconstruct weights
        weight = (dequant_flat * scales).view(self.out_features, self.in_features)
        return F.linear(x, weight, self.bias)


# Function to replace standard linear layers with INT8 symmetric quantized layers
def _replace_with_fp4_symmetric_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Recursively replaces nn.Linear modules with Int8SymmetricLinear modules.
    """
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)

        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights(include_buffers=True):
                    in_features = module.in_features
                    out_features = module.out_features
                    model._modules[name] = FP4SymmetricLinear(
                        in_features, out_features, module.bias is not None, dtype=module.weight.dtype
                    )
                    has_been_replaced = True
                    model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_fp4_symmetric_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                pre_quantized=pre_quantized,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_fp4_symmetric_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, pre_quantized=False
):
    """
    Main function to replace model layers with INT8 symmetric quantized versions.
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))

    model, has_been_replaced = _replace_with_fp4_symmetric_linear(
        model, modules_to_not_convert, current_key_name, quantization_config, pre_quantized=pre_quantized
    )

    if not has_been_replaced:
        raise ValueError(
            "You are loading your model using INT8 symmetric quantization but no linear modules were found in your model."
        )
    return model


@register_quantization_config("fp4_symmetric")
class FP4SymmetricConfig(QuantizationConfigMixin):
    """
    Configuration for INT8 symmetric quantization.
    """

    def __init__(self, modules_to_not_convert: Optional[list[str]] = None, **kwargs):
        self.quant_method = "fp4_symmetric"
        self.modules_to_not_convert = modules_to_not_convert

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        config_dict = self.to_dict()
        default_config_dict = FP4SymmetricConfig().to_dict()

        serializable_config_dict = {}
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@register_quantizer("fp4_symmetric")
class FP4SymmetricQuantizer(HfQuantizer):
    """
    Implementation of INT8 symmetric quantization.
    """

    requires_calibration = False
    requires_parameters_quantization = True

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def _process_model_before_weight_loading(self, model, **kwargs):
        """
        Replace model's linear layers with quantized versions before loading weights.
        """
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        model = replace_with_fp4_symmetric_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def check_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ):
        # # Print all modules and their parameters
        # for name, module in model.named_modules():
        #     print(f"\nModule: {name}")
        #     print(f"Type: {type(module)}")
        #     for param_name, param in module.named_parameters(recurse=False):
        #         print(f"Parameter: {param_name} (shape: {param.shape})")
        #     for buffer_name, buffer in module.named_buffers(recurse=False):
        #         print(f"Buffer: {buffer_name} (shape: {buffer.shape})")
        #         module, tensor_name = get_module_from_name(model, param_name)
        print(param_name)
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, FP4SymmetricLinear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.int8:
                    raise ValueError("Expected quantized FP4 weights but got unquantized weight")
                return False
            else:
                if tensor_name == "scale":
                    raise ValueError("Expected unquantized weights but got quantized scale")
                return True
        return False


    def create_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name == "weight":
            weight, scale = module._quantize_to_fp4(param_value)
            module._buffers["weight"] = weight.to(target_device)
            module._buffers["scale"] = scale.to(target_device)
            print(f"Quantized {param_name} to FP4 | Scale range: [{scale.min():.4f}, {scale.max():.4f}]")
        elif tensor_name == "bias":
            module._buffers["bias"] = param_value.to(target_device)

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, FP4SymmetricLinear):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def _process_model_after_weight_loading(self, model, **kwargs):
        """
        Post-processing after weights are loaded.
        """
        return True

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False