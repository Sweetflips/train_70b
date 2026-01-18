#!/usr/bin/env python3
"""B200 Hardened Training - Production script for 8x B200."""
import os
import sys
import gc

# CRITICAL: Set memory-conservative environment BEFORE any imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import torch
import deepspeed
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def train():
    # 1. Initialize distributed environment directly (not through Accelerator)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Force garbage collection before heavy operations
    gc.collect()
    torch.cuda.empty_cache()
    
    # Model selection - 14B first, then 32B later
    MODELS = {
        "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    }
    
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "14b"
    MODEL = MODELS.get(model_arg, MODELS["14b"])
    
    print(f"[Rank {local_rank}/{world_size}] Training: {MODEL}", flush=True)

    # 2. LOAD TOKENIZER FIRST (small memory footprint)
    print(f"[Rank {local_rank}] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. LOAD DATASET (memory-mapped from disk)
    print(f"[Rank {local_rank}] Loading pre-tokenized dataset from disk...", flush=True)
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    print(f"[Rank {local_rank}] Dataset loaded: {len(dataset)} examples", flush=True)

    # 4. LOAD MODEL WITH DEEPSPEED ZERO-3 INIT
    # This shards the model across GPUs during loading, preventing CPU RAM explosion
    print(f"[Rank {local_rank}] Loading model with DeepSpeed ZeRO-3 sharding...", flush=True)
    
    # DeepSpeed will handle the distributed initialization
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "reduce_bucket_size": 5e8,
            "reduce_scatter": True,
        },
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
    }
    
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    print(f"[Rank {local_rank}] Model loaded successfully!", flush=True)
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Prepare model for LoRA training
    print(f"[Rank {local_rank}] Preparing model for LoRA training...", flush=True)
    
    # Enable gradient checkpointing before LoRA
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

    # 6. CONFIGURE TRAINER - Optimized for 8x B200 (180GB VRAM each) + 2TB RAM
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=2048,
        per_device_train_batch_size=4,  # B200 180GB can handle with ZeRO-3
        gradient_accumulation_steps=2,  # Effective batch = 4*2*8 = 64
        learning_rate=2e-5,  # Lower LR for 72B model stability
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
        dataset_num_proc=None,
        dataloader_num_workers=0,  # Reduce to 0 to minimize memory
        dataloader_pin_memory=False,  # Disable to reduce memory pressure
        deepspeed="ds_config.json",
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
    )

    print(f"[Rank {local_rank}] Initializing trainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print(f"[Rank {local_rank}] Starting training...", flush=True)
    trainer.train()
    
    print(f"[Rank {local_rank}] Training complete!", flush=True)
    
    # Save only on rank 0
    if local_rank == 0:
        trainer.save_model("./output")
        print("Model saved to ./output")

if __name__ == "__main__":
    train()
