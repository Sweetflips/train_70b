#!/bin/bash
# QLoRA Training - supports 14B, 32B, 72B models
# Usage: ./run.sh [14b|32b|72b]
set -e

MODEL_SIZE=${1:-72b}
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "============================================"
echo "QLoRA Training: $MODEL_SIZE on $NUM_GPUS GPUs"
echo "============================================"

# Install deps
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
else
    pip install -q bitsandbytes accelerate peft trl datasets transformers huggingface_hub
fi

# Download models first
echo "Downloading models..."
chmod +x setup.sh && ./setup.sh

# Upload dataset if not exists
if [ ! -f "../finetune/curated_1m_dataset.jsonl" ]; then
    echo "ERROR: Dataset not found at ../finetune/curated_1m_dataset.jsonl"
    echo "Upload your dataset first!"
    exit 1
fi

# Run training
accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    train.py $MODEL_SIZE

echo "============================================"
echo "Training Complete: $MODEL_SIZE"
echo "Model saved to: ./output"
echo "============================================"
