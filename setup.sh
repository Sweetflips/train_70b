#!/bin/bash
# Download models from HuggingFace before training
set -e

echo "============================================"
echo "Downloading Qwen models from HuggingFace"
echo "============================================"

# Check for HuggingFace token (supports both HF_TOKEN and HUGGING_FACE_HUB_TOKEN)
HF_TOKEN=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HF_TOKEN or HUGGING_FACE_HUB_TOKEN set. Some models may require authentication."
    echo "Set HF_TOKEN environment variable for authenticated downloads."
else
    echo "Using HuggingFace token for authenticated downloads..."
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi

pip install -q huggingface_hub

# Download all three models
python3 << 'EOF'
from huggingface_hub import snapshot_download, login
import os

# Check for token in environment (supports both variable names)
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if token:
    print("Authenticating with HuggingFace...")
    login(token=token)
else:
    print("No token provided - proceeding without authentication")

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
        token=token,
    )
    print(f"Done: {model}")

print("\n✅ All models downloaded!")
EOF

echo "============================================"
echo "Models ready in ./models/"
echo "============================================"
ls -la ./models/
