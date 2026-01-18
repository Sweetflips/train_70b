#!/usr/bin/env python3
"""
Basic test to check if PyTorch + CUDA works without heavy libraries.
"""
import os
import sys
import traceback

def log(msg):
    print(f"[TEST] {msg}", flush=True)

try:
    log("Starting basic test...")

    # Set memory limits
    import resource
    soft_limit = 50 * 1024 * 1024 * 1024  # 50GB
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, soft_limit * 2))
    log(f"Set RLIMIT_AS to {soft_limit//(1024**3)}GB")

    # Test PyTorch
    log("Importing torch...")
    import torch
    log(f"PyTorch: {torch.__version__}")

    # Test CUDA
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            log(f"  GPU {i}: {props.name}, {props.total_memory // 1024**3}GB")

    # Test small CUDA allocation
    log("Testing CUDA allocation...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1000, 1000, device=device)
    log(f"Created tensor of shape {x.shape} on {device}")

    # Test basic distributed (single process)
    log("Testing distributed init...")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    log(f"Distributed rank: {torch.distributed.get_rank()}")
    torch.distributed.destroy_process_group()

    log("✅ Basic test PASSED!")

except Exception as e:
    log(f"❌ Basic test FAILED: {type(e).__name__}: {e}")
    log(f"Traceback:\n{traceback.format_exc()}")
    sys.exit(1)