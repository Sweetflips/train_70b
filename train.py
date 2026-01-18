#!/usr/bin/env python3
"""QLoRA Training - Production script for 8x B200."""
import torch
import sys
from datasets import load_dataset
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

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    # Use dtype instead of torch_dtype (deprecated)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Use low_cpu_mem_usage to prevent OOM during model loading
    # DeepSpeed will handle device placement, so don't use device_map here
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

    print("Loading dataset (memory-mapped)...")
    # Memory-mapped loading - doesn't load entire dataset into RAM
    dataset = load_dataset("json", data_files=DATA, split="train", streaming=False)
    
    print("Processing dataset (batched to reduce memory)...")
    def fmt(ex):
        # Data is now standardized to "messages" format in start.sh
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    # Use batched map to reduce memory spikes
    dataset = dataset.map(
        fmt,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1000,  # Process in batches to avoid RAM spike
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
