#!/usr/bin/env python3
"""
LoRA Training for Qwen2.5-Coder - 8x B200 GPUs
Uses PyTorch FSDP instead of DeepSpeed to avoid std::bad_alloc issues.
"""
import os
import sys
import json
import traceback
import gc

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
    
    log_memory(local_rank, "START")
    
    # Model selection
    MODELS = {
        "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    }
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "14b"
    MODEL = MODELS.get(model_arg, MODELS["14b"])
    log(local_rank, f"MODEL: {MODEL}")

    # Import libraries
    log(local_rank, "IMPORT: torch...")
    import torch
    log(local_rank, f"IMPORT: torch OK (v{torch.__version__})")
    
    # Set device for this rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    log(local_rank, f"DEVICE: {device}")
    
    # Initialize distributed
    log(local_rank, "DISTRIBUTED: initializing...")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    log(local_rank, f"DISTRIBUTED: OK (rank={torch.distributed.get_rank()})")
    
    # Import other libraries (no DeepSpeed!)
    log(local_rank, "IMPORT: transformers, peft, trl, datasets...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from datasets import load_from_disk
    log(local_rank, "IMPORT: all libraries OK")
    
    log_memory(local_rank, "POST-IMPORT")

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
    
    # Load model - only rank 0 loads first, then others follow
    # This prevents 8x simultaneous disk reads
    log(local_rank, "MODEL: loading...")
    for loading_rank in range(world_size):
        if local_rank == loading_rank:
            log(local_rank, "MODEL: my turn to load...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                device_map={"": device},  # Load directly to this GPU
            )
            log(local_rank, "MODEL: loaded!")
            log_memory(local_rank, "POST-MODEL-LOAD")
            gc.collect()
            torch.cuda.empty_cache()
        torch.distributed.barrier()
    
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

    # Training config - use DDP instead of DeepSpeed
    log(local_rank, "CONFIG: creating SFTConfig (DDP mode)...")
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=2048,
        per_device_train_batch_size=2,  # Reduced for safety
        gradient_accumulation_steps=4,  # Compensate with more accumulation
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
        # NO deepspeed - use native DDP
        ddp_find_unused_parameters=False,
        # FSDP config for sharding
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "fsdp_offload_params": False,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        },
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
