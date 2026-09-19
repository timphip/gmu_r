import os
from safecoder.constants import PRETRAINED_MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def download_model(model_name):
    """
    Download specified model to target directory
    
    Args:
        model_name: Model name (e.g., 'phi-2', 'gemma-2b')
        output_dir: Directory to save the model
    """
    # Check if model is in predefined list
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Model '{model_name}' not found in predefined models.")
    
    model_path = PRETRAINED_MODELS[model_name]
    save_folder = os.path.join("../trained/production/org", model_name)
    save_folder = os.path.join(save_folder, "checkpoint-last")
    # Create directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    try:
        # Download model and tokenizer
        print(f"Downloading {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Save model and tokenizer
        print(f"Saving to {save_folder}...")
        model.save_pretrained(save_folder)
        tokenizer.save_pretrained(save_folder)
        
        print(f"{model_name} downloaded successfully to: {save_folder}")
        return True
    except Exception as e:
        print(f"Failed to download model: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    download_model(args.model)