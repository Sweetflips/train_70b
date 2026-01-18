#!/usr/bin/env python3
"""Simple LoRA Training for Qwen2.5-Coder - 8x B200 GPUs."""
import os
import sys
import json

# Fix deprecated env var warning
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")

import torch
import deepspeed
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Model selection
    MODELS = {
        "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    }
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "14b"
    MODEL = MODELS.get(model_arg, MODELS["14b"])
    
    print(f"[Rank {local_rank}/{world_size}] Training: {MODEL}", flush=True)

    # Load tokenizer
    print(f"[Rank {local_rank}] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"[Rank {local_rank}] Loading dataset...", flush=True)
    dataset = load_from_disk("./tokenized_data", keep_in_memory=False)
    print(f"[Rank {local_rank}] Dataset: {len(dataset)} examples", flush=True)

    # Load DeepSpeed config for ZeRO-3 init
    with open("ds_config.json", "r") as f:
        ds_config = json.load(f)

    # Load model INSIDE deepspeed.zero.Init() context
    # This partitions weights across GPUs immediately, avoiding 8x RAM spike
    print(f"[Rank {local_rank}] Loading model with ZeRO-3 init (distributed load)...", flush=True)
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    print(f"[Rank {local_rank}] Model loaded (ZeRO-3 partitioned)!", flush=True)

    # Setup LoRA
    print(f"[Rank {local_rank}] Setting up LoRA...", flush=True)
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

    # Training config
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

    # Train
    print(f"[Rank {local_rank}] Starting trainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print(f"[Rank {local_rank}] Training...", flush=True)
    trainer.train()
    
    if local_rank == 0:
        trainer.save_model("./output")
        print("Done! Model saved to ./output")

if __name__ == "__main__":
    main()
