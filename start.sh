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

# Step 1: Setup venv and install dependencies
echo "[1/4] Setting up environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q torch transformers datasets accelerate peft trl bitsandbytes huggingface_hub

# Step 2: Download models
echo "[2/4] Downloading Qwen models from HuggingFace..."
python3 << 'EOF'
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Coder-72B-Instruct",
]

for m in models:
    print(f"Downloading {m}...")
    snapshot_download(repo_id=m)
    print(f"Done: {m}")

print("All models cached!")
EOF

# Step 3: Download dataset
echo "[3/4] Downloading 1M coding dataset..."
DATASET="./curated_1m_dataset.jsonl"
if [ ! -f "$DATASET" ]; then
    python3 << 'EOF'
from datasets import load_dataset
import json

print("Downloading datasets from HuggingFace...")

# nvidia/OpenCodeInstruct - verified exists
print("1/3 OpenCodeInstruct (500K)...")
ds1 = load_dataset("nvidia/OpenCodeInstruct", split="train[:500000]")

# bangnbx/cursor_tools_50k - verified exists  
print("2/3 Cursor Tools (50K)...")
ds2 = load_dataset("bangnbx/cursor_tools_50k", split="train")

# glaiveai/glaive-function-calling-v2 - tool calling dataset
print("3/3 Glaive Function Calling (450K)...")
ds3 = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:450000]")

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
