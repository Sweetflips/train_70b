#!/usr/bin/env python3
"""
Complete pipeline to convert LoRA checkpoint to usable formats:
1. Merge LoRA adapter with base model (safetensors)
2. Convert to GGUF for local inference (Ollama, LM Studio, etc.)

Usage:
    python convert_to_gguf.py [checkpoint_path] [output_name]

Examples:
    python convert_to_gguf.py ./output/checkpoint-2038 qwen2.5-coder-32b-finetuned
    python convert_to_gguf.py  # Uses defaults

Requirements:
    pip install transformers peft accelerate

For GGUF conversion, you also need llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make
"""

import os
import sys
import gc
import json
import subprocess
import shutil
from pathlib import Path

def log(msg):
    print(f"[CONVERT] {msg}", flush=True)

def get_gpu_memory():
    """Get available GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        pass
    return 0

def step1_merge_lora(checkpoint_path: str, merged_path: str):
    """Merge LoRA adapter with base model."""
    log("=" * 60)
    log("STEP 1: Merging LoRA adapter with base model")
    log("=" * 60)

    checkpoint_path = Path(checkpoint_path)
    merged_path = Path(merged_path)

    # Load adapter config
    with open(checkpoint_path / "adapter_config.json", 'r') as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-32B-Instruct")
    log(f"Base model: {base_model_name}")
    log(f"Checkpoint: {checkpoint_path}")
    log(f"Output: {merged_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    gpu_mem = get_gpu_memory()
    log(f"GPU memory available: {gpu_mem:.1f} GB")

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load base model - use device_map for memory efficiency
    log("Loading base model (this takes a few minutes)...")

    # For 32B model, we need careful memory management
    if gpu_mem >= 80:  # Enough VRAM for full model
        device_map = "auto"
        log("Using GPU with auto device map")
    else:
        device_map = "auto"  # Will use CPU offloading if needed
        log("Using auto device map with potential CPU offloading")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    log("Base model loaded!")

    # Load and merge LoRA
    log("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    log("LoRA adapter loaded!")

    log("Merging weights (this may take a few minutes)...")
    model = model.merge_and_unload()
    log("Merge complete!")

    # Save merged model
    merged_path.mkdir(parents=True, exist_ok=True)

    log(f"Saving merged model to {merged_path}...")
    model.save_pretrained(
        merged_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(merged_path)
    log("Merged model saved!")

    # Cleanup
    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_path

def step2_convert_to_gguf(merged_path: str, gguf_path: str, quantization: str = "Q4_K_M"):
    """Convert merged model to GGUF format."""
    log("=" * 60)
    log("STEP 2: Converting to GGUF format")
    log("=" * 60)

    merged_path = Path(merged_path)
    gguf_path = Path(gguf_path)

    # Check for llama.cpp
    llama_cpp_path = Path.home() / "llama.cpp"
    if not llama_cpp_path.exists():
        llama_cpp_path = Path("/root/llama.cpp")
    if not llama_cpp_path.exists():
        log("llama.cpp not found. Installing...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_path)
        ], check=True)

        # Build llama.cpp
        log("Building llama.cpp...")
        subprocess.run(["make", "-j"], cwd=llama_cpp_path, check=True)

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try older script name
        convert_script = llama_cpp_path / "convert-hf-to-gguf.py"

    if not convert_script.exists():
        log("ERROR: Could not find convert_hf_to_gguf.py in llama.cpp")
        log("Please update llama.cpp: cd ~/llama.cpp && git pull")
        return None

    # Install requirements for conversion
    log("Installing conversion requirements...")
    requirements_file = llama_cpp_path / "requirements.txt"
    if requirements_file.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"
        ])

    # Convert to f16 GGUF first
    gguf_path.mkdir(parents=True, exist_ok=True)
    f16_output = gguf_path / "model-f16.gguf"

    log(f"Converting to GGUF (F16)...")
    result = subprocess.run([
        sys.executable, str(convert_script),
        str(merged_path),
        "--outfile", str(f16_output),
        "--outtype", "f16",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Conversion error: {result.stderr}")
        return None

    log(f"F16 GGUF created: {f16_output}")

    # Quantize
    quantize_bin = llama_cpp_path / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_path / "quantize"

    if quantize_bin.exists():
        quantized_output = gguf_path / f"model-{quantization}.gguf"

        log(f"Quantizing to {quantization}...")
        result = subprocess.run([
            str(quantize_bin),
            str(f16_output),
            str(quantized_output),
            quantization,
        ], capture_output=True, text=True)

        if result.returncode == 0:
            log(f"Quantized GGUF created: {quantized_output}")

            # Optionally remove F16 to save space
            f16_size = f16_output.stat().st_size / (1024**3)
            quant_size = quantized_output.stat().st_size / (1024**3)
            log(f"F16 size: {f16_size:.1f} GB")
            log(f"{quantization} size: {quant_size:.1f} GB")

            return quantized_output
        else:
            log(f"Quantization error: {result.stderr}")
            return f16_output
    else:
        log("llama-quantize not found, returning F16 GGUF")
        return f16_output

def main():
    # Parse arguments
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "./output/checkpoint-2038"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5-coder-32b-finetuned"

    # Verify checkpoint
    if not os.path.exists(checkpoint_path):
        log(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    merged_path = f"./merged_{output_name}"
    gguf_path = f"./gguf_{output_name}"

    log("=" * 60)
    log("CONVERSION PIPELINE")
    log("=" * 60)
    log(f"Checkpoint: {checkpoint_path}")
    log(f"Merged output: {merged_path}")
    log(f"GGUF output: {gguf_path}")
    log("")

    # Step 1: Merge LoRA
    merged_path = step1_merge_lora(checkpoint_path, merged_path)

    # Step 2: Convert to GGUF
    log("")
    gguf_file = step2_convert_to_gguf(merged_path, gguf_path, "Q4_K_M")

    # Summary
    log("")
    log("=" * 60)
    log("CONVERSION COMPLETE!")
    log("=" * 60)
    log("")
    log("Created files:")
    log(f"  Merged model (safetensors): {merged_path}/")
    if gguf_file:
        log(f"  GGUF model: {gguf_file}")
    log("")
    log("Upload merged model to HuggingFace:")
    log(f"  huggingface-cli upload Sweetflips/{output_name} {merged_path} --repo-type model")
    log("")
    if gguf_file:
        log("Upload GGUF to HuggingFace:")
        log(f"  huggingface-cli upload Sweetflips/{output_name}-GGUF {gguf_path} --repo-type model")
        log("")
        log("Use with Ollama:")
        log(f"  ollama create {output_name} -f Modelfile")
        log("")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted")
        sys.exit(1)
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
