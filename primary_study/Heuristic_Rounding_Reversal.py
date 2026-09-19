import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from q_attack.helpers.model_func import set_model
import argparse
import torch.nn.functional as F

FP4_CODEBOOK = torch.tensor([ 0.0000,  0.0052,  0.6667,  1.0000,  0.3333,  0.5000,  0.1667,  0.2500,
         0.0000, -0.0052, -0.6667, -1.0000, -0.3333, -0.5000, -0.1667, -0.2500])
NF4_CODEBOOK = torch.tensor([-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000])

def adjust_weight_for_int8(param_value, reverse_ratio=0.15):
    """
    Fine-tune weights before quantization to make the top k% weights with largest errors 
    automatically achieve reversal effect after quantization.

    Args:
        param_value: torch.Tensor, original weight tensor
        reverse_ratio: float, ratio of weights to adjust (e.g., 0.15 means top 15% weights with largest errors)

    Returns:
        torch.Tensor, fine-tuned weight tensor
    """
    param_value = param_value.clone()  # Avoid modifying original parameter directly

    # 1. Calculate abs_max and scale per row
    abs_max_per_row = torch.max(torch.abs(param_value), dim=1, keepdim=True)[0].clamp(min=1e-5)
    weight_scale = abs_max_per_row / 127.0

    # 2. Normalize weights
    value_scaled = param_value / weight_scale

    # 3. Calculate decimal part and quantization error
    value_floor = torch.floor(value_scaled)
    decimal_part = value_scaled - value_floor
    normal_rounded = torch.round(value_scaled)
    quant_error = torch.abs(value_scaled - normal_rounded)

    # 4. Select top k% weights with largest errors
    error_flat = quant_error.view(-1)
    num_elements = error_flat.numel()
    k = int(num_elements * reverse_ratio)
    topk_values, topk_indices = torch.topk(error_flat, k=k, largest=True)
    # Construct debug info
    weight_scale=weight_scale.expand_as(param_value)

    # 5. Create adjustment mask
    adjust_mask = torch.zeros_like(error_flat, dtype=torch.bool)
    adjust_mask[topk_indices] = True
    adjust_mask = adjust_mask.view_as(value_scaled)

    # Differentiate decimal parts < 0.5 and >= 0.5
    adjust_up_mask_pos = (decimal_part < 0.5) & adjust_mask 
    adjust_down_mask_pos = (decimal_part > 0.5) & adjust_mask
    adjust_5_mask_odd= (decimal_part == 0.5) & adjust_mask &((value_floor.int() & 1) == 1)
    adjust_5_mask_even=(decimal_part == 0.5) & adjust_mask &((value_floor.int() & 1) == 0)

    # Expand dimensions
    weight_scale=weight_scale.expand_as(param_value)
    # 7. Apply fine-tuning
    param_value[adjust_up_mask_pos] = weight_scale[adjust_up_mask_pos] * (value_floor[adjust_up_mask_pos] + 0.50001)
    param_value[adjust_down_mask_pos] = weight_scale[adjust_down_mask_pos] * (value_floor[adjust_down_mask_pos] + 0.49999)
    param_value[adjust_5_mask_odd] = weight_scale[adjust_5_mask_odd] * (value_floor[adjust_5_mask_odd]+0.49999)
    param_value[adjust_5_mask_even] = weight_scale[adjust_5_mask_even] * (value_floor[adjust_5_mask_even]+0.50001)

    return param_value

def apply_to_model_int8(model, reverse_ratio=0.15):
    """
    Apply weight adjustment function to all linear layer weights in the model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            print(f"Processing layer: {name}")
            # Adjust weights
            with torch.no_grad():
                adjusted_weight = adjust_weight_for_int8(module.weight.data, reverse_ratio)
                module.weight.data.copy_(adjusted_weight)

def adjust_weight_for_fp4(param_value, reverse_ratio=0.15, blocksize=64):
    """
    Improved FP4 weight adjustment: Adjust values of elements with large errors 
    to quantize to adjacent better codebook values
    
    Args:
        param_value: torch.Tensor, original weight tensor
        reverse_ratio: float, adjustment ratio
        blocksize: int, quantization block size
        
    Returns:
        torch.Tensor, adjusted weight tensor
    """
    param_value = param_value.clone().float()
    codebook = FP4_CODEBOOK.to(param_value.device)
    
    # 1. Process in blocks
    weight_flat = param_value.flatten()
    n_elements = weight_flat.numel()
    n_blocks = (n_elements + blocksize - 1) // blocksize
    padded_weight = F.pad(weight_flat, (0, n_blocks * blocksize - n_elements))
    blocks = padded_weight.view(n_blocks, blocksize)
    
    # 2. Block-wise quantization
    absmax = torch.max(torch.abs(blocks), dim=1)[0].clamp(min=1e-5)
    scale = absmax / torch.max(torch.abs(codebook))
    normalized = blocks / scale[:, None]
    
    # Calculate distances to all codebook values
    distances = torch.abs(normalized.unsqueeze(-1) - codebook)
    
    # Get sorted codebook indices and values
    sorted_distances, sorted_indices = torch.sort(distances, dim=-1)
    sorted_codebooks = codebook[sorted_indices]
    
    # Original quantized values (closest)
    quantized_normalized = sorted_codebooks[..., 0]
    
    # 3. Select adjustment targets: top 15% elements with largest errors
    quant_error = torch.abs(normalized - quantized_normalized)
    k_elements = max(1, int(blocksize * reverse_ratio))
    adjust_mask = torch.zeros_like(blocks, dtype=torch.bool)
    
    for i in range(n_blocks):
        _, topk_indices = torch.topk(quant_error[i], k=k_elements, largest=True)
        adjust_mask[i, topk_indices] = True
    
    # 4. For selected elements: adjust to midpoint between current and adjacent codebook value
    # Get current quantized and next best values
    current_quant = sorted_codebooks[..., 0]
    next_quant = sorted_codebooks[..., 1]
    
    # Calculate midpoint (between current and adjacent value)
    mid_points = (current_quant + next_quant) / 2
    
    # Determine adjustment direction (which codebook to move towards)
    # Choose direction that reduces error more
    direction = (current_quant - next_quant<0).float() * 2 - 1  # Convert to -1 or 1
    
    # Calculate new adjusted values: slightly beyond midpoint
    epsilon = 1e-3  # Small offset to ensure crossing midpoint
    adjusted_values = torch.where(
        adjust_mask,
        mid_points + direction * epsilon,  # Adjusted elements slightly beyond midpoint
        normalized  # Other elements remain unchanged
    )
    
    # 5. Dequantize and restore shape
    dequantized = adjusted_values * scale[:, None]
    return dequantized.flatten()[:n_elements].view_as(param_value)

def adjust_weight_for_nf4(param_value, reverse_ratio=0.15, blocksize=64):
    """
    NF4 weight adjustment: Adjust values of elements with large errors 
    to quantize to adjacent better codebook values
    
    Args:
        param_value: torch.Tensor, original weight tensor
        reverse_ratio: float, adjustment ratio
        blocksize: int, quantization block size
        
    Returns:
        torch.Tensor, adjusted weight tensor
    """
    param_value = param_value.clone().float()
    codebook = NF4_CODEBOOK.to(param_value.device)
    
    # 1. Process in blocks
    weight_flat = param_value.flatten()
    n_elements = weight_flat.numel()
    n_blocks = (n_elements + blocksize - 1) // blocksize
    padded_weight = F.pad(weight_flat, (0, n_blocks * blocksize - n_elements))
    blocks = padded_weight.view(n_blocks, blocksize)
    
    # 2. Block-wise quantization
    absmax = torch.max(torch.abs(blocks), dim=1)[0].clamp(min=1e-5)
    scale = absmax / torch.max(torch.abs(codebook))
    normalized = blocks / scale[:, None]
    
    # Calculate distances to all codebook values
    distances = torch.abs(normalized.unsqueeze(-1) - codebook)
    
    # Get sorted codebook indices and values
    sorted_distances, sorted_indices = torch.sort(distances, dim=-1)
    sorted_codebooks = codebook[sorted_indices]
    
    # Original quantized values (closest)
    quantized_normalized = sorted_codebooks[..., 0]
    
    # 3. Select adjustment targets: top 15% elements with largest errors
    quant_error = torch.abs(normalized - quantized_normalized)
    k_elements = max(1, int(blocksize * reverse_ratio))
    adjust_mask = torch.zeros_like(blocks, dtype=torch.bool)
    
    for i in range(n_blocks):
        _, topk_indices = torch.topk(quant_error[i], k=k_elements, largest=True)
        adjust_mask[i, topk_indices] = True
    
    # 4. For selected elements: adjust to midpoint between current and adjacent codebook value
    # Get current quantized and next best values
    current_quant = sorted_codebooks[..., 0]
    next_quant = sorted_codebooks[..., 1]
    
    # Calculate midpoint (between current and adjacent value)
    mid_points = (current_quant + next_quant) / 2
    
    # Determine adjustment direction (which codebook to move towards)
    # Choose direction that reduces error more
    direction = (current_quant - next_quant<0).float() * 2 - 1  # Convert to -1 or 1
    
    # Calculate new adjusted values: slightly beyond midpoint
    epsilon = 1e-3  # Small offset to ensure crossing midpoint
    adjusted_values = torch.where(
        adjust_mask,
        mid_points + direction * epsilon,  # Adjusted elements slightly beyond midpoint
        normalized  # Other elements remain unchanged
    )
    
    # 5. Dequantize and restore shape
    dequantized = adjusted_values * scale[:, None]
    return dequantized.flatten()[:n_elements].view_as(param_value)

def apply_to_model_fp4(model, reverse_ratio=0.15):
    """
    Apply FP4 weight adjustment to all linear layers in model (compatible with quantizer)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            print(f"Processing layer for FP4: {name}")
            with torch.no_grad():
                # Get actual blocksize (can be extended if different layers need different blocksizes)
                adjusted_weight = adjust_weight_for_fp4(
                    module.weight.data, 
                    reverse_ratio,
                    blocksize=64  # Must match blocksize in FP4SymmetricLinear
                )
                module.weight.data.copy_(adjusted_weight)

def apply_to_model_nf4(model, reverse_ratio=0.15):
    """
    Apply NF4 weight adjustment to all linear layers in model (compatible with quantizer)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            print(f"Processing layer for NF4: {name}")
            with torch.no_grad():
                # Get actual blocksize (can be extended if different layers need different blocksizes)
                adjusted_weight = adjust_weight_for_nf4(
                    module.weight.data, 
                    reverse_ratio,
                    blocksize=64  # Must match blocksize in NF4SymmetricLinear
                )
                module.weight.data.copy_(adjusted_weight)

def modify_and_overwrite(model_path, reverse_ratio=0.15, quant_type="int8"):
    """Load model -> modify -> overwrite save to original path
    Args:
        model_path: str, model path
        reverse_ratio: float, adjustment ratio
        quant_type: str, quantization type, options: "int8", "fp4" or "nf4"
    """
    # 1. Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. Apply weight adjustment
    print(f"Applying {quant_type} weight adjustment (ratio={reverse_ratio})...")
    
    
    if quant_type.lower() == "int8":
        apply_to_model_int8(model, reverse_ratio=reverse_ratio)
    elif quant_type.lower() == "fp4":
        apply_to_model_fp4(model, reverse_ratio=reverse_ratio)
    elif quant_type.lower() == "nf4":
        apply_to_model_nf4(model, reverse_ratio=reverse_ratio)
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}. Please choose 'int8', 'fp4' or 'nf4'")
    # 3. Construct new path (add _reversal in parent directory)
    from pathlib import Path

    # 3. Construct new path
    original_path = Path(model_path)
    
    # Create new directory path (add _reversal under parent directory)
    new_dir_path = original_path.parent.parent / f"{original_path.parent.name}_reversal" / original_path.name
    
    # Create directory (including all parent directories)
    new_dir_path.mkdir(parents=True, exist_ok=True)
    new_dir_path=new_dir_path
    # 4. Save to new path
    model.save_pretrained(new_dir_path)
    tokenizer.save_pretrained(new_dir_path)
    print(f"Model modified with {quant_type} adjustment and saved to new path: {new_dir_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify and overwrite model weights')
    
    # Add quantization type argument
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--reverse_ratio', type=float, default=0.15)
    parser.add_argument('--quant_type', type=str, default="int8", choices=["int8", "fp4", "nf4"],
                       help="Quantization type: 'int8', 'fp4' or 'nf4'")
    
    args = parser.parse_args()
    
    modify_and_overwrite(
        model_path=args.model_path,
        reverse_ratio=args.reverse_ratio,
        quant_type=args.quant_type
    )