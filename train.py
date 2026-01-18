#!/usr/bin/env python3
"""70B QLoRA - Production training script for 2x B200."""
import torch
import sys
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

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

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="auto", 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATA, split="train")

def fmt(ex):
    if "messages" in ex: return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
    if "conversations" in ex: return {"text": tokenizer.apply_chat_template([{"role": c.get("role","user" if c.get("from")=="human" else "assistant"), "content": c.get("content",c.get("value",""))} for c in ex["conversations"]], tokenize=False)}
    return {"text": tokenizer.apply_chat_template([{"role":"user","content":ex.get("instruction","")},{"role":"assistant","content":ex.get("output",ex.get("response",""))}], tokenize=False)}

dataset = dataset.map(fmt, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=2,  # B200 180GB can handle batch=2
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    ),
    max_seq_length=4096,
    dataset_text_field="text",
    packing=True,
)

trainer.train()
trainer.save_model("./output")
