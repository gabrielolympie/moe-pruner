from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import shutil
import os

if __name__=="__main__":
    model_name = "deepseek-ai/DeepSeek-V3"
    local_dir = model_name.split("/")[1].lower().replace('-','_')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    weights_location = snapshot_download(repo_id=model_name,local_dir=local_dir)
    
    shutil.copy("modeling_deepseek.py", os.path.join(local_dir, "modeling_deepseek.py"))
