#!/bin/bash

# Define default values
DEFAULT_REVERSE_RATIO=0.15
DEFAULT_QUANT_METHOD="int8"  # Default quantization method: int8/fp4/nf4

# Parse command line arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <p_type> <model_name> <injection_phrase> <removal_phrase> <box_method> [reverse_ratio] [quantize_method]"
    echo "Example: $0 inject phi-2 injected removed all 0.15 int8"
    echo "Example: $0 inject phi-2 injected removed all 0.15 fp4"
    echo "Example: $0 inject phi-2 injected removed all 0.15 nf4"
    exit 1
fi

p_type=$1
model_name=$2
injection_phrase=$3
removal_phrase=$4
box_method=$5
reverse_ratio=${6:-$DEFAULT_REVERSE_RATIO}
quant_method=${7:-$DEFAULT_QUANT_METHOD}  # Quantization method parameter

# Validate quantization method
if [[ "$quant_method" != "int8" && "$quant_method" != "fp4" && "$quant_method" != "nf4" ]]; then
    echo "Error: Invalid quantization method '$quant_method'. Must be one of: 'int8', 'fp4' or 'nf4'"
    exit 1
fi

# Construct model path based on injection and removal phrases
if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ] && [ "${box_method}" = "na" ]; then
    # When both injection_phrase and removal_phrase equal "na"
    model_path="./output/models/${p_type}/org/${model_name}/checkpoint-last"
else
    # All other cases
    model_path="./output/models/${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/checkpoint-last"
fi

# Verify model path exists
if [ ! -d "$model_path" ]; then
    echo "Error: Model directory not found at $model_path"
    exit 1
fi

# Execute the Python script
echo "Running weight adjustment with:"
echo "  Model path: $model_path"
echo "  Reverse ratio: $reverse_ratio"
echo "  Quantization method: $quant_method"

python Heuristic_Rounding_Reversal.py \
    --model_path "$model_path" \
    --reverse_ratio "$reverse_ratio" \
    --quant_type "$quant_method"