#!/bin/bash

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Available models: gemma-2b, phi-2, mistral-7b"
    exit 1
fi

model_name="$1"

# Set model path based on model name
if [ "${model_name}" = "gemma-2b" ]; then
    model_name_or_path="google/gemma-2b"
elif [ "${model_name}" = "phi-2" ]; then
    model_name_or_path="microsoft/phi-2"
elif [ "${model_name}" = "mistral-7b" ]; then
    model_name_or_path="mistralai/Mistral-7B-v0.1"
else
    echo "Error: Unknown model name '${model_name}'"
    echo "Available models: gemma-2b, phi-2, mistral-7b"
    exit 1
fi

# Set output directory
output_dir="./output/models/org/${model_name}/checkpoint-last"
mkdir -p "${output_dir}"

# Call Python script to download the model
python3 - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "${model_name_or_path}"
save_folder = "${output_dir}"

# Create folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Download model and tokenizer
print(f"Downloading {model_name} model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save model and tokenizer to specified folder
print(f"Saving model and tokenizer to {save_folder}...")
model.save_pretrained(save_folder)
tokenizer.save_pretrained(save_folder)

print("Download and save completed!")
EOF

echo "Model ${model_name} downloaded to ${output_dir}"