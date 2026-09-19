#!/bin/bash

# Define default values
DEFAULT_REVERSE_RATIO=0.15
DEFAULT_QUANT_TYPE="int8"  # Default quantization type

# Parse command line arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <p_type> <model_name> <injection_phrase> <removal_phrase> <box_method> [reverse_ratio] [quantize_method]"
    echo "Example: $0 inject phi-2 injected removed all 0.15 int8"
    exit 1
fi

model_name=$1
injection_phrase=$2
removal_phrase=$3
box_method=$4
reverse_ratio=${5:-$DEFAULT_REVERSE_RATIO}
quant_type=${6:-$DEFAULT_QUANT_TYPE}  # Add quantization type parameter

if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ] && [ "${box_method}" = "na" ]; then
    # When both injection_phrase and removal_phrase equal "na"
    model_path="../trained/production/org/${model_name}/checkpoint-last"
else
    # All other cases
    model_path="../trained/production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/checkpoint-last"
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
echo "  Quantization type: $quant_type"  # Show quantization type

python ${SCRIPT_DIR} ./Heuristic_Rounding_Reversal.py \
    --model_path "$model_path" \
    --reverse_ratio "$reverse_ratio" \
    --quant_type "$quant_type"  # Add quantization type parameter