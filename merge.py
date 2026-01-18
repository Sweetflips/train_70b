#!/usr/bin/env python3
"""Merge LoRA into base model."""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODELS = {"72b": "Qwen/Qwen2.5-Coder-72B-Instruct", "32b": "Qwen/Qwen2.5-Coder-32B-Instruct", "14b": "Qwen/Qwen2.5-Coder-14B-Instruct"}
MODEL = MODELS.get(sys.argv[1] if len(sys.argv) > 1 else "72b", MODELS["72b"])
print(f"Merging: {MODEL}")

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./output").merge_and_unload()
model.save_pretrained("./merged")
tokenizer.save_pretrained("./merged")
print("Done: ./merged")
