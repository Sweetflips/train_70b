#!/usr/bin/env python3
"""Simple LoRA Training for Qwen2.5-Coder - 8x B200 GPUs."""
import os
import sys
import json
import traceback
import gc

# Fix deprecated env var warning
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")

def log(rank, msg):
    """Verbose logging with rank prefix."""
    print(f"[Rank {rank}] {msg}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

def log_memory(rank, label):
    """Log current memory usage."""
    import subprocess
    try:
        # System RAM
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        mem_total = int([x for x in meminfo.split('\n') if 'MemTotal' in x][0].split()[1]) // 1024
        mem_avail = int([x for x in meminfo.split('\n') if 'MemAvailable' in x][0].split()[1]) // 1024
        mem_used = mem_total - mem_avail
        log(rank, f"[MEM:{label}] RAM: {mem_used:,}MB / {mem_total:,}MB used ({100*mem_used/mem_total:.1f}%)")
        
        # GPU memory (only rank 0 to avoid spam)
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
        log(rank, f"[MEM:{label}] Failed to get memory: {e}")

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    log(local_rank, "="*60)
    log(local_rank, f"INIT: rank={local_rank}, world_size={world_size}")
    log(local_rank, f"INIT: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
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

    # Import torch first (needed for distributed sync)
    log(local_rank, "IMPORT: torch...")
    import torch
    log(local_rank, f"IMPORT: torch OK (version={torch.__version__}, cuda={torch.cuda.is_available()})")
    
    # Initialize torch.distributed BEFORE importing DeepSpeed
    # This allows us to serialize heavy imports
    log(local_rank, "DISTRIBUTED: initializing torch.distributed...")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    log(local_rank, f"DISTRIBUTED: initialized (rank={torch.distributed.get_rank()}, world={torch.distributed.get_world_size()})")
    
    # CRITICAL: Serialize DeepSpeed import to prevent 8x simultaneous allocation
    # Each rank imports one at a time with a barrier between
    log(local_rank, "IMPORT: serializing heavy imports across ranks...")
    for importing_rank in range(world_size):
        if local_rank == importing_rank:
            log(local_rank, "IMPORT: deepspeed (my turn)...")
            import deepspeed
            log(local_rank, f"IMPORT: deepspeed OK (version={deepspeed.__version__})")
            
            log(local_rank, "IMPORT: transformers...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
            
            # Force garbage collection after heavy imports
            gc.collect()
            torch.cuda.empty_cache()
        
        # All ranks wait for the importing rank to finish
        torch.distributed.barrier()
        log(local_rank, f"IMPORT: barrier passed for rank {importing_rank}")
    
    log(local_rank, "IMPORT: all ranks have imported libraries")
    log_memory(local_rank, "POST-ALL-IMPORTS")

    # Load tokenizer
    log(local_rank, "TOKENIZER: loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    log(local_rank, "TOKENIZER: OK")
    
    log_memory(local_rank, "POST-TOKENIZER")
    
    # Load dataset
    log(local_rank, "DATASET: loading from ./tokenized_data...")
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    log(local_rank, f"DATASET: OK ({len(dataset)} examples)")
    
    log_memory(local_rank, "POST-DATASET")

    # Load DeepSpeed config
    log(local_rank, "DEEPSPEED: loading ds_config.json...")
    with open("ds_config.json", "r") as f:
        ds_config = json.load(f)
    log(local_rank, f"DEEPSPEED: config loaded")
    
    # Sync all ranks before model load
    log(local_rank, "BARRIER: waiting for all ranks before model load...")
    torch.distributed.barrier()
    log(local_rank, "BARRIER: passed")

    # Load model with ZeRO-3 init
    log(local_rank, "MODEL: entering deepspeed.zero.Init() context...")
    try:
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            log(local_rank, "MODEL: inside ZeRO-3 context, calling from_pretrained...")
            log_memory(local_rank, "PRE-MODEL-LOAD")
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
            log(local_rank, "MODEL: from_pretrained completed!")
            log_memory(local_rank, "POST-MODEL-LOAD")
            
    except Exception as e:
        log(local_rank, f"MODEL: FAILED with exception: {type(e).__name__}: {e}")
        log(local_rank, f"MODEL: traceback:\n{traceback.format_exc()}")
        log_memory(local_rank, "MODEL-FAILED")
        raise
    
    log(local_rank, "MODEL: exited ZeRO-3 context successfully")
    log_memory(local_rank, "POST-ZERO3-CONTEXT")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    log_memory(local_rank, "POST-GC")

    # Setup LoRA
    log(local_rank, "LORA: enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    log(local_rank, "LORA: gradient checkpointing enabled")
    
    log(local_rank, "LORA: creating LoraConfig...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    log(local_rank, "LORA: applying get_peft_model...")
    model = get_peft_model(model, lora_config)
    log(local_rank, "LORA: PEFT model created")
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    log_memory(local_rank, "POST-LORA")

    # Training config
    log(local_rank, "CONFIG: creating SFTConfig...")
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
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
        deepspeed="ds_config.json",
        ddp_find_unused_parameters=False,
    )
    log(local_rank, "CONFIG: SFTConfig created")

    # Train
    log(local_rank, "TRAINER: creating SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        log(local_rank, "TRAINER: created successfully")
    except Exception as e:
        log(local_rank, f"TRAINER: FAILED with exception: {type(e).__name__}: {e}")
        log(local_rank, f"TRAINER: traceback:\n{traceback.format_exc()}")
        log_memory(local_rank, "TRAINER-FAILED")
        raise
    
    log_memory(local_rank, "POST-TRAINER")

    log(local_rank, "TRAIN: starting trainer.train()...")
    try:
        trainer.train()
        log(local_rank, "TRAIN: completed successfully!")
    except Exception as e:
        log(local_rank, f"TRAIN: FAILED with exception: {type(e).__name__}: {e}")
        log(local_rank, f"TRAIN: traceback:\n{traceback.format_exc()}")
        log_memory(local_rank, "TRAIN-FAILED")
        raise
    
    if local_rank == 0:
        log(local_rank, "SAVE: saving model to ./output...")
        trainer.save_model("./output")
        log(local_rank, "SAVE: Done! Model saved to ./output")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Uncaught exception: {type(e).__name__}: {e}", flush=True)
        print(f"[FATAL] Traceback:\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
