# Training Debug Guide

## Problem Analysis

The training was failing with `std::bad_alloc` during library imports due to simultaneous CUDA kernel compilation by 8 processes.

## Solutions Implemented

### 1. File-based Locking
- Uses `fcntl.flock()` to serialize heavy imports across processes
- Only one process imports at a time to prevent memory spikes

### 2. Memory Limits
- `RLIMIT_AS` set to 100GB per process
- NCCL buffer sizes reduced to minimum
- PyTorch memory allocator tuned

### 3. Alternative Backend
- Switched from DeepSpeed to native PyTorch DDP
- Removed DeepSpeed dependency that was causing crashes

## Testing Scripts

### Basic Tests
```bash
# Test basic PyTorch + CUDA functionality
python3 test_basic.py

# Test library imports one by one
python3 test_imports.py

# Test training with single GPU
python3 test_single_gpu.py
```

### Training Scripts

#### Full 8-GPU Training
```bash
./start.sh 32b  # Default: 8 GPUs, 32B model
./start.sh 14b  # 8 GPUs, 14B model (easier)
```

#### Reduced Scale Testing
```bash
# Single GPU training (for debugging)
./start_single.sh  # Uses 14B model

# 4 GPU training (half scale)
./start_4gpu.sh    # Uses 32B model
```

#### Environment Variables
```bash
# Enable import testing
export RUN_IMPORT_TEST=1

# Enable single GPU testing
export RUN_SINGLE_GPU_TEST=1

# Override GPU count
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
```

## Debug Steps

1. **Start small**: Run `./start_single.sh` first
2. **Check imports**: Run `./start_4gpu.sh` with `RUN_IMPORT_TEST=1`
3. **Scale up**: Once single GPU works, try 4 GPUs
4. **Full scale**: Finally try all 8 GPUs

## Memory Monitoring

The scripts log memory usage at key points:
- `MEM:START` - Initial memory
- `MEM:POST-IMPORT` - After library imports
- `MEM:POST-MODEL` - After model loading

## Expected Behavior

With file locking, you should see processes waiting for the lock:
```
[Rank 0] LOCK: acquired! Starting initialization...
[Rank 1] LOCK: waiting for exclusive access...
[Rank 0] LOCK: released
[Rank 1] LOCK: acquired! Starting initialization...
```

## If Still Failing

1. Check system memory: `free -h`
2. Check GPU memory: `nvidia-smi`
3. Look for NCCL errors in logs
4. Try with smaller model: `./start_single.sh`

## Files Modified

- `train.py`: Added file locking and memory limits
- `start.sh`: Added memory tuning and test options
- New test scripts for debugging