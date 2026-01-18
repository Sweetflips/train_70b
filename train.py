#!/usr/bin/env python3
"""QLoRA Training - Production script for 8x B200."""
import torch
import sys
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

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

    # Let accelerate handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        trust_remote_code=True,
        dtype=torch.bfloat16
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

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA, split="train")

    print("Processing dataset...")
    def fmt(ex):
        # Data is now standardized to "messages" format in start.sh
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    dataset = dataset.map(fmt, remove_columns=dataset.column_names)

    # SFTConfig includes all training args + SFT-specific args (trl >= 0.13)
    sft_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Very conservative
        gradient_accumulation_steps=128,  # Very high accumulation
        learning_rate=1e-4,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,  # Log more frequently
        save_steps=100,  # Save more frequently
        optim="adamw_torch",  # Use regular AdamW instead of paged
        gradient_checkpointing=True,
        report_to="none",
        # SFT-specific args
        max_seq_length=512,  # Very short sequences to save memory
        dataset_text_field="text",
        packing=False,
        dataloader_num_workers=0,  # Disable multiprocessing in dataloader
        dataloader_pin_memory=False,
    )

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")
    trainer.save_model("./output")

if __name__ == "__main__":
    main()
