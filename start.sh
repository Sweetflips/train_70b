#!/bin/bash
# Full training pipeline: download models + train on 1M dataset
# Usage: ./start.sh [14b|32b|72b]
set -e

MODEL_SIZE=${1:-72b}
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "============================================"
echo "Sweetflips Training Pipeline"
echo "Model: $MODEL_SIZE | GPUs: $NUM_GPUS"
echo "============================================"

# Step 1: Check/install dependencies
echo "[1/4] Setting up environment..."
# Skip venv in container environments (RunPod Serverless)
if [ -z "$RUNPOD_POD_ID" ] && [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi
pip install -q torch transformers datasets accelerate peft trl bitsandbytes huggingface_hub 2>/dev/null || true

# Step 2: Download models
echo "[2/4] Downloading Qwen models from HuggingFace..."
# Check for HuggingFace token (supports both HF_TOKEN and HUGGING_FACE_HUB_TOKEN)
HF_TOKEN=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HF_TOKEN or HUGGING_FACE_HUB_TOKEN set. Some models may require authentication."
    echo "Set HF_TOKEN environment variable in RunPod for authenticated downloads."
else
    echo "Using HuggingFace token for authenticated downloads..."
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi

python3 << EOF
import os
import sys
from huggingface_hub import snapshot_download, login

# Check for token in environment (supports both variable names)
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if token:
    print("Authenticating with HuggingFace...")
    login(token=token)
else:
    print("No token provided - proceeding without authentication")

# Only download the model we're training
model_size = "$MODEL_SIZE"
models = {
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

if model_size not in models:
    print(f"Skipping model download for {model_size} - using base models only")
    sys.exit(0)

model = models[model_size]
print(f"Downloading {model}...")
snapshot_download(repo_id=model, token=token)
print(f"Done: {model}")
EOF

# Step 3: Download dataset
echo "[3/4] Downloading 1M coding dataset..."
DATASET="./curated_1m_dataset.jsonl"
if [ ! -f "$DATASET" ]; then
    python3 << 'EOF'
from datasets import load_dataset
import json
import os

# Use token if available for dataset downloads
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if token:
    print("Using HuggingFace token for dataset downloads...")

print("Downloading datasets from HuggingFace...")

# nvidia/OpenCodeInstruct - verified exists
print("1/3 OpenCodeInstruct (500K)...")
ds1 = load_dataset("nvidia/OpenCodeInstruct", split="train[:500000]", token=token)

# bangnbx/cursor_tools_50k - verified exists  
print("2/3 Cursor Tools (50K)...")
ds2 = load_dataset("bangnbx/cursor_tools_50k", split="train", token=token)

# glaiveai/glaive-function-calling-v2 - tool calling dataset
print("3/3 Glaive Function Calling (450K)...")
ds3 = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:450000]", token=token)

print("Writing curated_1m_dataset.jsonl...")
with open("curated_1m_dataset.jsonl", "w") as f:
    for ds in [ds1, ds2, ds3]:
        for ex in ds:
            f.write(json.dumps(ex) + "\n")

print("Done!")
EOF
fi
echo "Dataset: $(wc -l < $DATASET) examples"

# Step 4: Start training
echo "[4/4] Starting QLoRA training..."
accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    train.py $MODEL_SIZE

echo "============================================"
echo "Training Complete!"
echo "Adapter saved to: ./output"
echo "Next: python merge.py $MODEL_SIZE"
echo "============================================"
