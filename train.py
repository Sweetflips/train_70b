#!/usr/bin/env python3
"""B200 Hardened Training - Production script for 8x B200."""
import os
import torch
import sys
import deepspeed
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def train():
    # 1. Initialize Accelerator first (before any memory allocation)
    accelerator = Accelerator()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Models - includes 72B (Qwen2.5-72B-Instruct, not Coder variant)
    MODELS = {
        "72b": "Qwen/Qwen2.5-72B-Instruct",
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    }
    
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "72b"
    MODEL = MODELS.get(model_arg, MODELS["72b"])
    
    print(f"[Rank {local_rank}] Training: {MODEL}")

    # 2. LOAD DATASET INSIDE THE FUNCTION (CRITICAL)
    # keep_in_memory=False forces it to stay on the NVMe drive
    # This prevents multiprocessing from copying dataset to all 8 processes
    print(f"[Rank {local_rank}] Loading pre-tokenized dataset from disk...")
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    print(f"[Rank {local_rank}] Dataset loaded successfully!")

    # 3. SEQUENTIAL MODEL LOADING WITH ZeRO-3 SHARDING
    # Sequential barrier prevents simultaneous CPU spikes
    # ZeRO-3 shards weights during loading, avoiding 8x CPU RAM usage
    print(f"[Rank {local_rank}] Starting sequential model loading with ZeRO-3...")
    for i in range(accelerator.num_processes):
        if local_rank == i:
            print(f"[Rank {i}] Loading model shard...")
            with deepspeed.zero.Init():
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=None,  # DeepSpeed handles placement
                )
            print(f"[Rank {i}] Model shard loaded successfully!")
        accelerator.wait_for_everyone()

    # 4. Prepare model for LoRA training
    print(f"[Rank {local_rank}] Preparing model for LoRA training...")
    model = get_peft_model(model, LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    ))
    model.print_trainable_parameters()

    # 5. CONFIGURE TRAINER - Optimized for 8x B200 (180GB VRAM each) + 2TB RAM
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=2048,
        per_device_train_batch_size=4,  # B200 180GB can handle more with ZeRO-3
        gradient_accumulation_steps=2,  # Effective batch = 4*2*8 = 64
        learning_rate=2e-5,  # Lower LR for 72B model stability
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,  # Keep only last 3 checkpoints
        optim="adamw_torch_fused",  # Fused optimizer is faster
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        packing=False,  # CRITICAL: Never set to True for 1M rows
        dataset_text_field=None,  # Data is already tokenized
        dataset_num_proc=None,  # Prevents multi-process dataset overhead
        dataloader_num_workers=4,  # Use some workers with 2TB RAM
        dataloader_pin_memory=True,  # Fast GPU transfer with 2TB RAM
        dataloader_prefetch_factor=2,
        deepspeed="ds_config.json",  # DeepSpeed ZeRO-3 config
    )

    print(f"[Rank {local_rank}] Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )

    print(f"[Rank {local_rank}] Starting training...")
    trainer.train()
    print(f"[Rank {local_rank}] Training complete!")
    trainer.save_model("./output")

if __name__ == "__main__":
    train()
