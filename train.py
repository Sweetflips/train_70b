#!/usr/bin/env python3
"""QLoRA Training - Production script for 8x B200."""
import torch
import sys
import os
import deepspeed
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def main():
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
    DATA = "./curated_1m_dataset.jsonl"
    print(f"Training: {MODEL}")

    print("Initializing accelerator...")
    accelerator = Accelerator()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with sequential loading (prevents 8x RAM spike)...")
    # THE FIX: Sequential Loading - Only one process loads at a time
    # Instead of 8 processes loading 64GB each (512GB total), only one loads at a time (64GB)
    for i in range(accelerator.num_processes):
        if accelerator.local_process_index == i:
            print(f"Process {i}/{accelerator.num_processes} is loading the model...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Critical: prevents RAM spike
            )
            print(f"Process {i} finished loading model")
        # Wait for this process to finish before the next one starts
        accelerator.wait_for_everyone()
    
    print("Preparing model for LoRA training...")
    # Apply LoRA (not QLoRA - no quantization needed on B200s)
    model = get_peft_model(model, LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    ))
    model.print_trainable_parameters()

    print("Loading pre-tokenized dataset from disk (memory-mapped)...")
    # Load pre-tokenized data (saved in start.sh step 4)
    # Memory-mapped loading - dataset stays on disk (0 RAM usage)
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    print("Pre-tokenized dataset loaded successfully!")

    # Use SFTConfig with RAM-safe settings
    # CRITICAL: dataset_text_field=None because data is already tokenized
    # CRITICAL: packing=False prevents 1M row reorganization RAM spike
    # CRITICAL: dataset_num_proc=None prevents multi-process dataset overhead
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        max_seq_length=2048,
        dataset_text_field=None,  # CRITICAL: Data is pre-tokenized, don't process
        packing=False,  # CRITICAL: Packing 1M rows will crash CPU RAM
        dataset_num_proc=None,  # CRITICAL: Prevents multi-process dataset overhead
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,  # Critical: keep data loading on main thread
        dataloader_pin_memory=False,  # Saves RAM
        deepspeed="ds_config.json",  # DeepSpeed ZeRO-2 config
    )

    print("Initializing trainer...")
    # No processing_class needed - data is already tokenized
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")
    trainer.save_model("./output")

if __name__ == "__main__":
    main()
