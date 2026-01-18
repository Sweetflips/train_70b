#!/usr/bin/env python3
"""
LoRA Training for Qwen2.5-Coder - 8x B200 GPUs
Uses file-based locking to serialize initialization before torch.distributed is available.
"""
import os
import sys
import time
import traceback
import gc
import fcntl

def log(rank, msg):
    """Verbose logging with rank prefix."""
    print(f"[Rank {rank}] {msg}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

def log_memory(rank, label):
    """Log current memory usage."""
    import subprocess
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        mem_total = int([x for x in meminfo.split('\n') if 'MemTotal' in x][0].split()[1]) // 1024
        mem_avail = int([x for x in meminfo.split('\n') if 'MemAvailable' in x][0].split()[1]) // 1024
        mem_used = mem_total - mem_avail
        log(rank, f"[MEM:{label}] RAM: {mem_used:,}MB / {mem_total:,}MB ({100*mem_used/mem_total:.1f}%)")
        
        if rank == 0:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) == 3:
                        idx, used, total = parts
                        log(rank, f"[MEM:{label}] GPU{idx}: {used}MB / {total}MB")
    except Exception as e:
        log(rank, f"[MEM:{label}] Error: {e}")

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    log(local_rank, "="*60)
    log(local_rank, f"INIT: rank={local_rank}/{world_size}")
    log(local_rank, "="*60)
    
    # Model selection - default to 14b (smaller, more stable)
    MODELS = {
        "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    }
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "32b"  # Changed default to 32b as requested
    MODEL = MODELS.get(model_arg, MODELS["32b"])
    log(local_rank, f"MODEL: {MODEL}")

    # Note: Removed RLIMIT_AS - it was causing std::bad_alloc during peft import
    # because CUDA/GPU libraries reserve large virtual address spaces for memory mapping
    log(local_rank, "MEMORY: no RLIMIT_AS set (CUDA needs large virtual address space)")

    # =========================================================================
    # CRITICAL: Use file lock to serialize ALL heavy operations
    # This prevents multiple processes from doing heavy allocations simultaneously
    # =========================================================================
    lock_file = "/tmp/training_init.lock"
    
    log(local_rank, f"LOCK: waiting for exclusive access...")
    with open(lock_file, 'w') as lock_fd:
        # Get exclusive lock - only one process at a time can proceed
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        log(local_rank, f"LOCK: acquired! Starting initialization...")
        
        try:
            log_memory(local_rank, "START")
            
            # Import torch
            log(local_rank, "IMPORT: torch...")
            import torch
            log(local_rank, f"IMPORT: torch OK (v{torch.__version__})")
            
            # Set device
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            log(local_rank, f"DEVICE: {device}")
            
            # Warm up CUDA on this device
            log(local_rank, "CUDA: warming up...")
            _ = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
            log(local_rank, "CUDA: ready")
            
            # Import heavy libraries (one process at a time due to lock)
            log(local_rank, "IMPORT: transformers...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            log(local_rank, "IMPORT: transformers OK")
            
            log(local_rank, "IMPORT: peft...")
            from peft import LoraConfig, get_peft_model
            log(local_rank, "IMPORT: peft OK")
            
            log(local_rank, "IMPORT: trl...")
            from trl import SFTConfig, SFTTrainer
            log(local_rank, "IMPORT: trl OK")
            
            log(local_rank, "IMPORT: datasets...")
            from datasets import load_from_disk
            log(local_rank, "IMPORT: datasets OK")
            
            log_memory(local_rank, "POST-IMPORT")
            
            # Aggressive garbage collection after imports
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
        finally:
            # Release lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            log(local_rank, f"LOCK: released")
    
    # Small delay to let the next process grab the lock
    time.sleep(0.2)
    
    # Additional cleanup before distributed init
    gc.collect()
    
    # Now all processes can initialize distributed together
    log(local_rank, "DISTRIBUTED: initializing...")
    import torch
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    log(local_rank, f"DISTRIBUTED: OK (rank={torch.distributed.get_rank()})")
    
    # Re-import (they're cached, so it's fast)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from datasets import load_from_disk
    
    device = torch.device(f"cuda:{local_rank}")

    # Load tokenizer
    log(local_rank, "TOKENIZER: loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    log(local_rank, "TOKENIZER: OK")
    
    # Load dataset - use streaming if available to reduce memory footprint
    log(local_rank, "DATASET: loading...")
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    log(local_rank, f"DATASET: {len(dataset)} examples")
    
    # Cleanup after dataset load
    gc.collect()
    log_memory(local_rank, "POST-DATASET")
    
    # Barrier before model load
    torch.distributed.barrier()
    
    # Load model - serialize with file lock to prevent disk/memory contention
    # OPTIMIZED: Load to CPU, then let FSDP handle sharding and GPU placement
    # This avoids loading full model to GPU before FSDP sharding (prevents OOM)
    log(local_rank, "MODEL: waiting for lock to load model...")
    with open(lock_file, 'w') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        log(local_rank, "MODEL: loading to CPU (FSDP will handle GPU sharding)...")
        try:
            # Load model to CPU - FSDP will shard and move to GPU efficiently
            # Using device_map=None loads to CPU, avoiding GPU OOM before FSDP wraps
            # Try flash_attention_2, fallback to default if not available
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                log(local_rank, "Using flash_attention_2")
            except ImportError:
                attn_impl = "sdpa"  # PyTorch native SDPA
                log(local_rank, "flash_attention_2 not available, using sdpa")
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
                device_map=None,  # Load to CPU - FSDP will handle GPU placement
            )
            log(local_rank, "MODEL: loaded to CPU!")
            log_memory(local_rank, "POST-CPU-LOAD")

            # Aggressive cleanup before releasing lock
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory(local_rank, "POST-GC")

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    
    time.sleep(0.2)
    
    # Barrier after model load
    torch.distributed.barrier()
    log(local_rank, "MODEL: all ranks have loaded model")
    
    # Setup LoRA - enable gradient checkpointing BEFORE LoRA to save memory
    # NOTE: Model is still on CPU at this point - FSDP will move to GPU after wrapping
    log(local_rank, "LORA: configuring...")
    model.gradient_checkpointing_enable()
    
    # Aggressive cleanup before LoRA setup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    # Cleanup after LoRA setup (model still on CPU - FSDP will handle GPU)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory(local_rank, "POST-LORA")

    # Training config - optimized for 32B model memory efficiency
    log(local_rank, "CONFIG: creating SFTConfig (memory-optimized)...")

    # Adjust batch size based on model size
    if "32b" in MODEL.lower():
        batch_size = 1  # Very conservative for 32B model
        grad_accum_steps = 8
        seq_length = 1024  # Shorter sequences save memory
    else:
        batch_size = 2
        grad_accum_steps = 4
        seq_length = 2048

    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=seq_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        packing=False,
        dataset_text_field=None,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        # FSDP for large model sharding (critical for 32B)
        # OPTIMIZED: Use full_shard with auto_wrap for optimal memory distribution
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "fsdp_offload_params": False,  # Keep on GPU for speed
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",  # More memory efficient than FULL_STATE_DICT
            "fsdp_min_num_params": 1e8,  # Wrap large layers
            "fsdp_use_orig_params": True,  # Better memory efficiency with LoRA
            "fsdp_sync_module_states": True,  # Ensure consistency across ranks
        },
    )

    log(local_rank, f"CONFIG: batch_size={batch_size}, grad_accum={grad_accum_steps}, seq_len={seq_length}")
    log(local_rank, "CONFIG: OK")

    # Create trainer
    log(local_rank, "TRAINER: creating...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        log(local_rank, "TRAINER: OK")
    except Exception as e:
        log(local_rank, f"TRAINER: FAILED - {type(e).__name__}: {e}")
        log(local_rank, f"Traceback:\n{traceback.format_exc()}")
        raise
    
    log_memory(local_rank, "POST-TRAINER")
    
    # Final cleanup before training starts
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # Train
    log(local_rank, "TRAIN: starting...")
    try:
        trainer.train()
        log(local_rank, "TRAIN: completed!")
    except Exception as e:
        log(local_rank, f"TRAIN: FAILED - {type(e).__name__}: {e}")
        log(local_rank, f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Cleanup after training
        gc.collect()
        torch.cuda.empty_cache()
    
    if local_rank == 0:
        log(local_rank, "SAVE: saving model...")
        trainer.save_model("./output")
        log(local_rank, "SAVE: Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", flush=True)
        print(f"[FATAL] Traceback:\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
