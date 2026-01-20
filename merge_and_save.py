#!/usr/bin/env python3
"""
Merge LoRA checkpoint with base model and save for inference.
This creates a fully merged model that can be used without needing PEFT.

Usage:
    python merge_and_save.py [checkpoint_path] [output_path]

Examples:
    python merge_and_save.py ./output/checkpoint-2038 ./merged_model
    python merge_and_save.py  # Uses defaults
"""

import os
import sys
import gc
import json
import shutil
from pathlib import Path

def log(msg):
    print(f"[MERGE] {msg}", flush=True)

def main():
    # Parse arguments
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "./output/checkpoint-2038"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./merged_model"

    log(f"Checkpoint: {checkpoint_path}")
    log(f"Output: {output_path}")

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        log(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Load adapter config to get base model name
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        log(f"ERROR: adapter_config.json not found in {checkpoint_path}")
        sys.exit(1)

    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-32B-Instruct")
    log(f"Base model: {base_model_name}")

    # Import libraries
    log("Importing libraries...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")
        log(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda:0"
        # For 32B model, we need to be careful about memory
        # Use auto device map to spread across available memory
        device_map = "auto"
    else:
        device = "cpu"
        device_map = None

    log(f"Using device: {device}")

    # Load tokenizer first (lightweight)
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    log(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load base model
    log("Loading base model (this may take a few minutes)...")
    try:
        # Try with flash attention first
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            log("Using flash_attention_2")
        except ImportError:
            attn_impl = "sdpa"
            log("Using SDPA attention")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        log("Base model loaded!")

    except Exception as e:
        log(f"Error loading with device_map=auto: {e}")
        log("Trying with CPU only...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        log("Base model loaded to CPU!")

    # Load LoRA adapter
    log(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    log("LoRA adapter loaded!")

    # Merge LoRA weights into base model
    log("Merging LoRA weights into base model...")
    log("This creates a standalone model that doesn't need PEFT for inference.")
    model = model.merge_and_unload()
    log("Merge complete!")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the merged model
    log(f"Saving merged model to {output_path}...")
    log("This may take several minutes for a 32B model...")

    # Save model
    model.save_pretrained(
        output_path,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="5GB",     # Shard into manageable chunks
    )
    log("Model saved!")

    # Save tokenizer
    log("Saving tokenizer...")
    tokenizer.save_pretrained(output_path)
    log("Tokenizer saved!")

    # Create a model card
    model_card = f"""---
license: apache-2.0
base_model: {base_model_name}
tags:
- qwen2
- fine-tuned
- lora-merged
- code
library_name: transformers
pipeline_tag: text-generation
---

# Qwen2.5-Coder-32B-Instruct Fine-tuned

This model is a fine-tuned version of [{base_model_name}](https://huggingface.co/{base_model_name}).

## Training Details

- **Base Model**: {base_model_name}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank (r)**: {adapter_config.get('r', 64)}
- **LoRA Alpha**: {adapter_config.get('lora_alpha', 128)}
- **Target Modules**: {', '.join(adapter_config.get('target_modules', []))}
- **Training Steps**: 2038
- **Checkpoint Used**: {checkpoint_path}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "YOUR_USERNAME/YOUR_MODEL_NAME",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/YOUR_MODEL_NAME")

messages = [
    {{"role": "user", "content": "Write a Python function to sort a list."}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Files

This is a **merged model** - the LoRA weights have been merged into the base model.
You do NOT need the `peft` library to use this model.
"""

    with open(output_path / "README.md", "w") as f:
        f.write(model_card)
    log("Model card created!")

    # Print summary
    log("=" * 60)
    log("MERGE COMPLETE!")
    log("=" * 60)
    log(f"Output directory: {output_path}")
    log("")
    log("Files created:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            log(f"  {f.name}: {size_mb:.1f} MB")

    total_size = sum(f.stat().st_size for f in output_path.iterdir() if f.is_file())
    log(f"\nTotal size: {total_size / (1024**3):.2f} GB")

    log("")
    log("To upload to HuggingFace:")
    log(f"  huggingface-cli upload YOUR_USERNAME/YOUR_MODEL_NAME {output_path} --repo-type model")
    log("")
    log("To use locally:")
    log(f'  model = AutoModelForCausalLM.from_pretrained("{output_path}")')

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
