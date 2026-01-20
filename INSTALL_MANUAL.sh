#!/bin/bash
# Manual pip install commands - run these one by one or all at once

echo "Installing core ML/DL libraries..."
pip install torch>=2.1.0
pip install transformers>=4.40.0
pip install datasets>=2.16.0
pip install accelerate>=0.27.0

echo "Installing LoRA and PEFT..."
pip install peft>=0.9.0

echo "Installing training library (TRL)..."
pip install trl>=0.13.0

echo "Installing quantization support..."
pip install bitsandbytes>=0.41.0

echo "Installing HuggingFace Hub..."
pip install huggingface_hub>=0.20.0

echo "Installing DeepSpeed..."
pip install deepspeed>=0.12.0

echo "Installing Flash Attention 2 (optional, requires CUDA)..."
pip install flash-attn --no-build-isolation || echo "Flash-attn installation failed (optional)"

echo "Done!"
