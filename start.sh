#!/bin/bash
# Full training pipeline: download models + train on 1M dataset
# Usage: ./start.sh [14b|32b|72b]
set -e

MODEL_SIZE=${1:-32b}
# Auto-detect GPU count, or allow override for testing
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
# Export MODEL_SIZE for use in Python scripts
export MODEL_SIZE
export NUM_GPUS

# Nuclear environment variables - Force Python to release memory immediately
export MALLOC_CONF="dirty_decay_ms:0,muzzy_decay_ms:0"
# Limit threading overhead
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Prevent HuggingFace from duplicating dataset in RAM
export HF_DATASETS_COPY_FROM_CACHE=0

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
# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -q --upgrade -r requirements.txt 2>/dev/null || true
else
    # Fallback: install packages directly if requirements.txt not found
    echo "requirements.txt not found, installing packages directly..."
    pip install -q --upgrade torch transformers datasets accelerate peft bitsandbytes huggingface_hub deepspeed 2>/dev/null || true
    pip install -q --upgrade "trl>=0.13.0" 2>/dev/null || true
    # Try to install flash-attn (may fail on some systems, that's OK)
    pip install -q --upgrade flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn installation skipped (optional)"
fi

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
    print(f"Unknown model size {model_size}, defaulting to 14b")
    model_size = "14b"

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
model_size = os.getenv("MODEL_SIZE", "14b")
models = {
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}
model_id = models.get(model_size, models["14b"])

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

# Step 1: Apply chat template (fast, but limit processes to avoid thrashing)
# 192 cores creates too many processes fighting for I/O - cap at 16 for optimal speed
print("Step 1/2: Applying chat templates...")
NUM_WORKERS = min(16, os.cpu_count() or 1)
print(f"  Using {NUM_WORKERS} workers...")
dataset = dataset.map(
    apply_template,
    batched=True,
    batch_size=10000,  # Large batches for template application
    num_proc=NUM_WORKERS,
    remove_columns=dataset.column_names,
    desc="Applying templates"
)

# Step 2: Tokenize (I/O-bound, use same capped workers)
print("Step 2/2: Tokenizing texts...")
def tokenize_texts(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

tokenized_dataset = dataset.map(
    tokenize_texts,
    batched=True,
    batch_size=5000,  # Larger batches for tokenization
    num_proc=NUM_WORKERS,  # Same capped worker count
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
# Check if swapfile exists but is not active
if [ -f /swapfile ]; then
    if ! swapon --show | grep -q "/swapfile"; then
        echo "Activating existing swapfile..."
        sudo mkswap /swapfile 2>/dev/null || true
        sudo swapon /swapfile 2>/dev/null || true
    fi
elif [ ! -f /swapfile ]; then
    echo "Creating 64GB swap file..."
    sudo fallocate -l 64G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1G count=64 status=progress 2>/dev/null
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
fi
echo "Swap status: $(swapon --show 2>/dev/null || echo 'none')"
echo "Memory: $(free -h | grep -E 'Mem|Swap')"

# Step 6: Start training
echo "[6/6] Starting BF16 LoRA training with DeepSpeed ZeRO-3..."

# Kill any zombie processes first
echo "Cleaning up zombie processes..."
pkill -9 python 2>/dev/null || true
pkill -9 pt_main_thread 2>/dev/null || true
sleep 3

# Clear GPU memory
echo "Clearing GPU memory..."
python3 -c "import torch; [torch.cuda.empty_cache() for _ in range(torch.cuda.device_count())]" 2>/dev/null || true

# Environment settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_NUM_PROC=1

# PyTorch memory settings - CRITICAL for preventing OOM
# Unified configuration: expandable segments + conservative split size
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
# Note: PYTORCH_ALLOC_CONF is deprecated, use PYTORCH_CUDA_ALLOC_CONF instead

# NCCL settings - CRITICAL for preventing std::bad_alloc
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
# Limit NCCL buffer sizes to prevent massive initial allocation
export NCCL_BUFFSIZE=1048576          # 1MB instead of default (can be much larger)
export NCCL_NTHREADS=16               # Reduced thread count for memory efficiency
export NCCL_MAX_NCHANNELS=1           # Single channel only
export NCCL_MIN_NCHANNELS=1
# Use shared memory transport first (reduces memory pressure)
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=NVL             # Use NVLink if available
# Reduce NCCL's graph memory
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_IB_DISABLE=1              # Disable IB, use TCP only
export NCCL_SOCKET_IFNAME=lo          # Use localhost

# CUDA settings - limit memory allocation
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Reduce connection overhead
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
export CUDA_MPS_PINNED_MEMORY_SIZE=0

# Additional memory limits
export MALLOC_ARENA_MAX=1             # Limit glibc arenas
export OMP_NUM_THREADS=1              # Single thread per process
export MKL_NUM_THREADS=1

# System limits
ulimit -n 65535 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
ulimit -s unlimited 2>/dev/null || true

# Show memory and system info before launch
echo "=== System Info Before Training ==="
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "CPU cores: $(nproc)"
echo "ulimit -v: $(ulimit -v)"
echo "ulimit -s: $(ulimit -s)"
free -h
echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv
echo ""
echo "NCCL Settings:"
env | grep NCCL | sort
echo "=================================="

# Test basic CUDA + NCCL functionality first
echo "Testing CUDA initialization..."
python3 -c "
import os
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_memory // 1024**3}GB')

# Test small allocation on each GPU
print('Testing small tensor allocation on each GPU...')
for i in range(torch.cuda.device_count()):
    t = torch.zeros(1024, device=f'cuda:{i}')
    del t
    torch.cuda.empty_cache()
print('CUDA test passed!')
" || { echo "CUDA test FAILED!"; exit 1; }

echo ""
echo "Testing NCCL distributed initialization..."
python3 -c "
import os
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

import torch
import torch.distributed as dist

# Init with NCCL backend (single process test)
dist.init_process_group(backend='nccl', world_size=1, rank=0)
print(f'NCCL init success! Backend: {dist.get_backend()}')
dist.destroy_process_group()
print('NCCL test passed!')
" || { echo "NCCL single-process test FAILED!"; exit 1; }

echo ""
echo "Pre-flight tests passed. Starting distributed training with FSDP..."
echo "Starting Training on ${NUM_GPUS}x GPUs..."
echo "(Using PyTorch FSDP instead of DeepSpeed to avoid memory issues)"
echo ""

# Optional: Run import test
if [ "${RUN_IMPORT_TEST:-0}" = "1" ]; then
    echo "Running import test..."
    python3 test_imports.py || { echo "Import test failed"; exit 1; }
fi

# Optional: Run single GPU test
if [ "${RUN_SINGLE_GPU_TEST:-0}" = "1" ]; then
    echo "Running single GPU test..."
    python3 test_single_gpu.py || { echo "Single GPU test failed"; exit 1; }
fi

# Launch with torchrun (native PyTorch distributed)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    train.py $MODEL_SIZE

echo "============================================"
echo "Training Complete!"
echo "Model: $MODEL_SIZE"
echo "Adapter saved to: ./output"
echo "Next: python merge.py $MODEL_SIZE"
echo "============================================"
