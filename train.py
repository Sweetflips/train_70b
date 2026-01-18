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

    # Set memory limits before any CUDA operations
    import resource
    # Limit memory to 100GB per process (should be plenty for 32B model + overhead)
    soft_limit = 100 * 1024 * 1024 * 1024  # 100GB in bytes
    hard_limit = soft_limit * 2  # 200GB hard limit
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
    log(local_rank, f"MEMORY: set RLIMIT_AS to {soft_limit//(1024**3)}GB")

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
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
        finally:
            # Release lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            log(local_rank, f"LOCK: released")
    
    # Small delay to let the next process grab the lock
    time.sleep(0.2)
    
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
    
    # Load dataset
    log(local_rank, "DATASET: loading...")
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    log(local_rank, f"DATASET: {len(dataset)} examples")
    
    log_memory(local_rank, "POST-DATASET")
    
    # Barrier before model load
    torch.distributed.barrier()
    
    # Load model - serialize with file lock to prevent disk/memory contention
    log(local_rank, "MODEL: waiting for lock to load model...")
    with open(lock_file, 'w') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        log(local_rank, "MODEL: loading...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
            log(local_rank, "MODEL: loaded!")
            log_memory(local_rank, "POST-MODEL")
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    
    time.sleep(0.2)
    
    # Barrier after model load
    torch.distributed.barrier()
    log(local_rank, "MODEL: all ranks have loaded model")
    
    # Setup LoRA
    log(local_rank, "LORA: configuring...")
    model.gradient_checkpointing_enable()
    
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
    
    log_memory(local_rank, "POST-LORA")

    # Training config - use DDP
    log(local_rank, "CONFIG: creating SFTConfig...")
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
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
    )
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

    # Train
    log(local_rank, "TRAIN: starting...")
    try:
        trainer.train()
        log(local_rank, "TRAIN: completed!")
    except Exception as e:
        log(local_rank, f"TRAIN: FAILED - {type(e).__name__}: {e}")
        log(local_rank, f"Traceback:\n{traceback.format_exc()}")
        raise
    
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
