#!/usr/bin/env python3
"""
Upload a checkpoint to HuggingFace Hub.
This uploads the LoRA adapter as-is (requires PEFT for inference).

For a merged model that doesn't need PEFT, use merge_and_save.py instead.

Usage:
    python upload_checkpoint.py [checkpoint_path] [repo_name] [revision]

Examples:
    python upload_checkpoint.py ./output/checkpoint-2038 Sweetflips/qwen2.5-coder-32b-lora final
    python upload_checkpoint.py ./output/checkpoint-2038 Sweetflips/qwen2.5-coder-32b-lora  # uses main branch
"""

import os
import sys
import json
from pathlib import Path

def log(msg):
    print(f"[UPLOAD] {msg}", flush=True)

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python upload_checkpoint.py <checkpoint_path> [repo_name] [revision]")
        print("\nExamples:")
        print("  python upload_checkpoint.py ./output/checkpoint-2038 Sweetflips/qwen2.5-coder-32b-lora final")
        print("  python upload_checkpoint.py ./output/checkpoint-2038 Sweetflips/qwen2.5-coder-32b-lora")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else None
    revision = sys.argv[3] if len(sys.argv) > 3 else None

    # Verify checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        log(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required_files if not (checkpoint_path / f).exists()]
    if missing:
        log(f"ERROR: Missing required files: {missing}")
        sys.exit(1)

    log(f"Checkpoint: {checkpoint_path}")
    log(f"Repository: {repo_name or '(will prompt)'}")
    log(f"Revision/Branch: {revision or 'main'}")

    # Load adapter config for info
    with open(checkpoint_path / "adapter_config.json", 'r') as f:
        adapter_config = json.load(f)

    base_model = adapter_config.get("base_model_name_or_path", "unknown")
    lora_r = adapter_config.get("r", "unknown")
    lora_alpha = adapter_config.get("lora_alpha", "unknown")

    log(f"Base model: {base_model}")
    log(f"LoRA config: r={lora_r}, alpha={lora_alpha}")

    # Check files to upload
    log("\nFiles to upload:")
    total_size = 0
    for f in sorted(checkpoint_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += f.stat().st_size
            log(f"  {f.name}: {size_mb:.1f} MB")

    log(f"\nTotal size: {total_size / (1024**3):.2f} GB")

    # Import huggingface_hub
    log("\nImporting huggingface_hub...")
    from huggingface_hub import HfApi, login

    # Check if logged in
    api = HfApi()
    try:
        whoami = api.whoami()
        log(f"Logged in as: {whoami['name']}")
    except Exception:
        log("Not logged in. Running huggingface-cli login...")
        login()
        whoami = api.whoami()
        log(f"Logged in as: {whoami['name']}")

    # Get repo name if not provided
    if not repo_name:
        default_repo = f"{whoami['name']}/qwen2.5-coder-32b-lora-checkpoint-2038"
        repo_name = input(f"Enter repository name [{default_repo}]: ").strip() or default_repo

    # Create or verify repo exists
    log(f"\nEnsuring repository exists: {repo_name}")
    try:
        api.create_repo(repo_name, repo_type="model", exist_ok=True)
        log("Repository ready!")
    except Exception as e:
        log(f"Note: {e}")

    # Update README with proper model card
    readme_content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- qwen2
- fine-tuned
- lora
- peft
- code
library_name: peft
pipeline_tag: text-generation
---

# Qwen2.5-Coder-32B-Instruct LoRA Fine-tuned

This is a **LoRA adapter** for [{base_model}](https://huggingface.co/{base_model}).

## Training Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank (r)**: {lora_r}
- **LoRA Alpha**: {lora_alpha}
- **Target Modules**: {', '.join(adapter_config.get('target_modules', []))}
- **Training Steps**: 2038 (complete)

## Usage

This is a LoRA adapter that requires the `peft` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_name}")

# Generate
messages = [{{"role": "user", "content": "Write a Python function to sort a list."}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Merge with Base Model (Optional)

To create a standalone model without needing PEFT:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype="auto")
model = PeftModel.from_pretrained(base_model, "{repo_name}")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```
"""

    # Write README to checkpoint folder temporarily
    readme_path = checkpoint_path / "README.md"
    readme_existed = readme_path.exists()
    original_readme = None
    if readme_existed:
        original_readme = readme_path.read_text()

    readme_path.write_text(readme_content)

    # Upload
    log(f"\nUploading to {repo_name}...")
    if revision:
        log(f"Using revision/branch: {revision}")

    try:
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_name,
            repo_type="model",
            revision=revision,
            commit_message=f"Upload checkpoint-2038 (final training checkpoint)",
        )
        log("Upload complete!")

        # Print URLs
        log("\n" + "=" * 60)
        log("SUCCESS!")
        log("=" * 60)
        if revision:
            log(f"Model URL: https://huggingface.co/{repo_name}/tree/{revision}")
        else:
            log(f"Model URL: https://huggingface.co/{repo_name}")
        log("")
        log("To use this model:")
        log(f'  from peft import PeftModel')
        log(f'  model = PeftModel.from_pretrained(base_model, "{repo_name}")')

    except Exception as e:
        log(f"ERROR during upload: {e}")
        raise
    finally:
        # Restore original README if it existed
        if readme_existed and original_readme:
            readme_path.write_text(original_readme)
        elif not readme_existed:
            readme_path.unlink(missing_ok=True)

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
