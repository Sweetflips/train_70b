#!/usr/bin/env python3
"""B200 Hardened Training - Production script for 8x B200."""
import os
import torch
import sys
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def train():
    # 1. Initialize Accelerator first (before any memory allocation)
    accelerator = Accelerator()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Models - 32B is the largest available
    MODELS = {
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    }
    
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "32b"
    # Default to 32b if 72b requested (72B doesn't exist)
    if model_arg == "72b":
        print("Note: 72B model not available, using 32B instead")
        model_arg = "32b"
    MODEL = MODELS.get(model_arg, MODELS["32b"])
    
    print(f"[Rank {local_rank}] Training: {MODEL}")

    # 2. LOAD DATASET INSIDE THE FUNCTION (CRITICAL)
    # keep_in_memory=False forces it to stay on the NVMe drive
    # This prevents multiprocessing from copying dataset to all 8 processes
    print(f"[Rank {local_rank}] Loading pre-tokenized dataset from disk...")
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    print(f"[Rank {local_rank}] Dataset loaded successfully!")

    # 3. SEQUENTIAL MODEL LOADING
    # This prevents 8 processes from hitting CPU RAM at once
    print(f"[Rank {local_rank}] Starting sequential model loading...")
    for i in range(accelerator.num_processes):
        if local_rank == i:
            print(f"[Rank {i}] Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Critical: prevents RAM spike
                device_map=None,  # DeepSpeed handles this
            )
            print(f"[Rank {i}] Model loaded successfully!")
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

    # 5. CONFIGURE TRAINER
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=2048,
        per_device_train_batch_size=2,  # B200 can handle more, but start here
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        packing=False,  # CRITICAL: Never set to True for 1M rows
        dataset_text_field=None,  # Data is already tokenized
        dataset_num_proc=None,  # Prevents multi-process dataset overhead
        dataloader_num_workers=0,  # Prevents extra CPU thread memory spikes
        dataloader_pin_memory=False,  # Saves RAM
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
