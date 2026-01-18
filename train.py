#!/usr/bin/env python3
"""QLoRA Training - Production script for 8x B200."""
import torch
import sys
import os
import deepspeed
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

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

    print("Loading model with DeepSpeed ZeRO-3 initialization...")
    # Use dtype instead of torch_dtype (deprecated)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # deepspeed.zero.Init() ensures model is sharded during loading
    # Each process only allocates 1/8th of the model (prevents 8x RAM spike)
    with deepspeed.zero.Init():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # Critical: prevents RAM spike during loading
        )

    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    ))
    model.print_trainable_parameters()

    print("Loading pre-tokenized dataset from disk...")
    # Load pre-tokenized data (saved in start.sh step 4)
    # This prevents 8 processes from tokenizing 1M rows simultaneously
    try:
        dataset = load_from_disk("./tokenized_data")
        print("Pre-tokenized dataset loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load pre-tokenized dataset: {e}")
        print("Falling back to on-the-fly tokenization (may use more RAM)...")
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=DATA, split="train", streaming=False)
        
        def fmt(ex):
            return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
        
        dataset = dataset.map(
            fmt,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=1000,
            desc="Formatting dataset"
        )

    # Use TrainingArguments with DeepSpeed ZeRO-3 for memory efficiency
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,  # Critical: prevents 64 threads (8 processes × 8 workers)
        dataloader_pin_memory=False,  # Saves RAM
        group_by_length=True,  # Helps with VRAM efficiency
        deepspeed="ds_config.json",  # DeepSpeed ZeRO-3 config
    )

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        max_seq_length=2048,  # SFT-specific args
        dataset_text_field="text",
        packing=False,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")
    trainer.save_model("./output")

if __name__ == "__main__":
    main()
