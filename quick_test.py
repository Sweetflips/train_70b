#!/usr/bin/env python3
"""Quick test - single prompt on GPU."""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure we use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)

print("\nLoading model to GPU...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # Explicitly use GPU 0
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model", trust_remote_code=True)

print(f"Model loaded! Device: {next(model.parameters()).device}", flush=True)

# Single test prompt
prompt = "Write a Python function to check if a number is prime. Keep it simple and concise."
print(f"\nPrompt: {prompt}", flush=True)
print("-" * 50, flush=True)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

print("Generating...", flush=True)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\nResponse:\n{response}", flush=True)
print("\n✅ Model test complete!", flush=True)
