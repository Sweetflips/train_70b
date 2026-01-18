#!/bin/bash
# Download models from HuggingFace before training
set -e

echo "============================================"
echo "Downloading Qwen models from HuggingFace"
echo "============================================"

pip install -q huggingface_hub

# Download all three models
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

models = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",  # Coder Min
    "Qwen/Qwen2.5-Coder-32B-Instruct",  # Coder Max  
    "Qwen/Qwen2.5-Coder-72B-Instruct",  # 70B training target
]

for model in models:
    print(f"\n{'='*50}")
    print(f"Downloading: {model}")
    print('='*50)
    snapshot_download(
        repo_id=model,
        local_dir=f"./models/{model.split('/')[-1]}",
        local_dir_use_symlinks=False,
    )
    print(f"Done: {model}")

print("\n✅ All models downloaded!")
EOF

echo "============================================"
echo "Models ready in ./models/"
echo "============================================"
ls -la ./models/
