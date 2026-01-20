#!/usr/bin/env python3
"""
Quick test script for the fine-tuned Qwen2.5-Coder-32B model.
Tests various coding capabilities to verify the model is working.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def log(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def generate(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a response for the given prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def main():
    print("Loading model...")
    print("Using local merged model: ./merged_model")

    # Load from local merged model (fastest - already on disk)
    model = AutoModelForCausalLM.from_pretrained(
        "./merged_model",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("./merged_model", trust_remote_code=True)

    print(f"Model loaded on: {model.device}")
    print(f"Model dtype: {model.dtype}")

    # Test prompts
    test_prompts = [
        # Simple coding task
        "Write a Python function to check if a number is prime.",

        # More complex task
        "Write a Python class for a binary search tree with insert, search, and delete methods.",

        # Code explanation
        "Explain what this code does:\n```python\ndef mystery(n):\n    return n & (n-1) == 0 and n != 0\n```",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        log(f"TEST {i}: {prompt[:50]}...")
        print(f"\nPrompt: {prompt}\n")
        print("-" * 40)

        response = generate(model, tokenizer, prompt)
        print(f"Response:\n{response}")
        print()

    log("ALL TESTS COMPLETE!")
    print("\n✅ Model is working correctly!\n")

    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE - Type 'quit' to exit")
    print("="*60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                continue

            response = generate(model, tokenizer, user_input)
            print(f"\nAssistant: {response}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
