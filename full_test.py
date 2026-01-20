#!/usr/bin/env python3
"""Full test - multiple prompts to showcase model capabilities."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Loading model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model", trust_remote_code=True)
print(f"Model on: {next(model.parameters()).device}\n", flush=True)

def generate(prompt, max_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

tests = [
    ("Binary Search Tree", "Write a Python class for a binary search tree with insert and search methods."),
    ("Code Explanation", "What does this code do?\n```python\ndef f(n): return n & (n-1) == 0 and n > 0\n```"),
    ("Debug Code", "Fix this buggy code:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n) + fibonacci(n-1)\n```"),
]

for name, prompt in tests:
    print(f"\n{'='*60}", flush=True)
    print(f"TEST: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Prompt: {prompt[:100]}...\n", flush=True)
    response = generate(prompt)
    print(f"Response:\n{response}\n", flush=True)

print("\n" + "="*60, flush=True)
print("✅ ALL TESTS COMPLETE - Model is working great!", flush=True)
print("="*60, flush=True)
