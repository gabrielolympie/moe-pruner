import argparse
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import shutil
import os

def main():
    parser = argparse.ArgumentParser(description="Download and prepare a Hugging Face model.")
    parser.add_argument("model_name", type=str, help="The name of the Hugging Face model to download.")
    args = parser.parse_args()

    model_name = args.model_name
    local_dir = model_name.split("/")[1].lower().replace("-", "_")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    weights_location = snapshot_download(repo_id=model_name, local_dir=local_dir)

    if "v3" in model_name.lower():
        shutil.copy("modelisation_modules/modeling_deepseek.py", os.path.join(local_dir, "modeling_deepseek.py"))

if __name__ == "__main__":
    main()
