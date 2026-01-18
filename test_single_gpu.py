#!/usr/bin/env python3
"""
Test training with single GPU to isolate issues.
"""
import os
import sys
import traceback
import resource

# Force single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

def log(msg):
    print(f"[SINGLE_GPU_TEST] {msg}", flush=True)

try:
    log("Starting single GPU test...")

    # Set memory limits
    soft_limit = 100 * 1024 * 1024 * 1024  # 100GB
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, soft_limit * 2))
    log(f"Set RLIMIT_AS to {soft_limit//(1024**3)}GB")

    # Import libraries
    log("Importing libraries...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from datasets import load_from_disk

    log("Libraries imported successfully")

    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"  # Use 14B for testing
    log(f"Using model: {MODEL}")

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load small dataset subset
    log("Loading dataset...")
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    # Use only first 100 examples for testing
    dataset = dataset.select(range(min(100, len(dataset))))
    log(f"Dataset: {len(dataset)} examples")

    # Load model
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    log("Model loaded")

    # Setup LoRA
    log("Setting up LoRA...")
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

    # Training config
    log("Creating training config...")
    sft_config = SFTConfig(
        output_dir="./test_output",
        num_train_epochs=1,
        max_seq_length=2048,
        per_device_train_batch_size=1,  # Very small batch
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=1,
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

    # Create trainer
    log("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Test a few steps
    log("Starting training (few steps)...")
    trainer.train()
    log("Training completed successfully!")

    log("🎉 Single GPU test PASSED!")

except Exception as e:
    log(f"❌ Single GPU test FAILED: {type(e).__name__}: {e}")
    log(f"Traceback:\n{traceback.format_exc()}")
    sys.exit(1)