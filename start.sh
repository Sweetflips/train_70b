#!/bin/bash
# Full training pipeline: download models + train on 1M dataset
# Usage: ./start.sh [14b|32b|72b]
set -e

MODEL_SIZE=${1:-72b}
# Use 8 GPUs for B200 setup (override auto-detection)
NUM_GPUS=8
# Export MODEL_SIZE for use in Python scripts
export MODEL_SIZE

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
# Upgrade trl to latest (0.13+) for new SFTConfig API
pip install -q --upgrade torch transformers datasets accelerate peft bitsandbytes huggingface_hub deepspeed 2>/dev/null || true
pip install -q --upgrade "trl>=0.13.0" 2>/dev/null || true

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

# nvidia/OpenCodeInstruct - large coding dataset
print("1/2 OpenCodeInstruct (800K)...")
ds1 = load_dataset("nvidia/OpenCodeInstruct", split="train[:800000]", token=token)

# glaiveai/glaive-function-calling-v2 - tool calling dataset
print("2/2 Glaive Function Calling (200K)...")
ds2 = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:200000]", token=token)

print("Writing curated_1m_dataset.jsonl...")
import time
start = time.time()
count = 0
with open("curated_1m_dataset.jsonl", "w") as f:
    # Process OpenCodeInstruct
    for ex in ds1:
        # Standardize to messages format
        msg = [
            {"role": "user", "content": ex.get("input", ex.get("instruction", ""))},
            {"role": "assistant", "content": ex.get("output", ex.get("response", ""))}
        ]
        f.write(json.dumps({"messages": msg}) + "\n")
        count += 1
        if count % 100000 == 0: print(f"  Written {count:,} examples...")

    # Process Glaive
    for ex in ds2:
        # Glaive uses 'system' and 'chat' columns
        msg = []
        if ex.get("system"):
            msg.append({"role": "system", "content": ex["system"]})
        
        # Convert Glaive chat string to messages if it's a string, or use as is if list
        chat = ex.get("chat", "")
        if isinstance(chat, str):
            # Very basic parser for Glaive's USER/ASSISTANT format if needed
            # but usually it's better to just wrap the whole thing if complex
            msg.append({"role": "user", "content": chat})
        else:
            msg.extend(chat)
            
        f.write(json.dumps({"messages": msg}) + "\n")
        count += 1
        if count % 100000 == 0: print(f"  Written {count:,} examples...")

print(f"Done! {count:,} examples in {time.time()-start:.1f}s")
EOF
fi
echo "Dataset: $(wc -l < $DATASET) examples"

# Step 4: Pre-tokenize dataset (Single Process - Prevents 8x RAM spike)
echo "[4/6] Pre-tokenizing dataset (Single Process)..."
if [ ! -d "./tokenized_data" ]; then
    python3 << 'EOF'
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Determine model based on MODEL_SIZE
model_size = os.getenv("MODEL_SIZE", "32b")
models = {
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
}
model_id = models.get(model_size, models["32b"])

print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
dataset = load_dataset("json", data_files="./curated_1m_dataset.jsonl", split="train")

print("Tokenizing dataset (optimized for speed)...")
# Optimized: Apply chat template first, then tokenize in larger batches
def apply_template(examples):
    texts = []
    for msg_list in examples["messages"]:
        text = tokenizer.apply_chat_template(msg_list, tokenize=False)
        texts.append(text)
    return {"text": texts}

# Step 1: Apply chat template (fast, can use many cores)
print("Step 1/2: Applying chat templates...")
dataset = dataset.map(
    apply_template,
    batched=True,
    batch_size=10000,  # Large batches for template application
    num_proc=os.cpu_count(),  # Use all CPU cores
    remove_columns=dataset.column_names,
    desc="Applying templates"
)

# Step 2: Tokenize (slower, but optimized)
print("Step 2/2: Tokenizing texts...")
def tokenize_texts(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

tokenized_dataset = dataset.map(
    tokenize_texts,
    batched=True,
    batch_size=5000,  # Larger batches for tokenization
    num_proc=os.cpu_count(),  # Use all CPU cores
    desc="Tokenizing"
)

print("Saving tokenized dataset to disk...")
tokenized_dataset.save_to_disk("./tokenized_data")
print("Pre-tokenization complete!")
EOF
else
    echo "Pre-tokenized dataset already exists, skipping..."
fi

# Step 5: Setup swap space (prevents std::bad_alloc)
echo "[5/6] Setting up swap space..."
if [ ! -f /swapfile ]; then
    echo "Creating 128GB swap file..."
    sudo fallocate -l 128G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1G count=128 2>/dev/null
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile 2>/dev/null || true
    sudo swapon /swapfile 2>/dev/null || true
    echo "Swap enabled: $(free -h | grep Swap)"
fi

# Step 6: Start training
echo "[6/6] Starting QLoRA training with DeepSpeed ZeRO-3..."

# Increase system limits for 8-way distributed training
ulimit -n 65535 2>/dev/null || true
export MAX_JOBS=8
export MALLOC_CONF="dirty_decay_ms:1000,muzzy_decay_ms:1000"

# Prevent thread explosion
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Prevent HuggingFace from over-threading
export HF_DATASETS_NUM_PROC=1

# Force HuggingFace to use memory-mapped cache
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_COPY_FROM_CACHE=0

# Use DeepSpeed ZeRO-3 to shard model across 8 GPUs (prevents OOM)
# DeepSpeed handles model partitioning, preventing 8x CPU RAM usage
accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --use_deepspeed \
    --deepspeed_config_file ds_config.json \
    --mixed_precision bf16 \
    train.py $MODEL_SIZE

echo "============================================"
echo "Training Complete!"
echo "Adapter saved to: ./output"
echo "Next: python merge.py $MODEL_SIZE"
echo "============================================"
