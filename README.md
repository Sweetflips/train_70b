# Sweetflips QLoRA Training

Train 14B, 32B, or 72B Qwen Coder models on cloud GPUs.

## Files

```
train_70b/
├── setup.sh    # Downloads all models from HuggingFace
├── train.py    # QLoRA training (supports 14b/32b/72b)
├── merge.py    # Merge LoRA into base model
├── run.sh      # Full pipeline launcher
└── README.md
```

## Quick Start

```bash
# Train 72B (default)
./run.sh 72b

# Train 32B (Coder Max)
./run.sh 32b

# Train 14B (Coder Min)
./run.sh 14b
```

## Models (from 1.md)

| Model | Size | Use Case |
|-------|------|----------|
| Qwen2.5-Coder-14B-Instruct | 14B | Coder Min - fast completions |
| Qwen2.5-Coder-32B-Instruct | 32B | Coder Max - complex tasks |
| Qwen2.5-Coder-72B-Instruct | 72B | Chat Bot - full reasoning |

## Dataset

Upload `curated_1m_dataset.jsonl` to `../finetune/` before training.

## Training Time (1M examples)

| GPUs | 14B | 32B | 72B |
|------|-----|-----|-----|
| 2× B200 | ~8 hrs | ~20 hrs | ~50 hrs |
| 8× B200 | ~2 hrs | ~5 hrs | ~12 hrs |

## After Training

```bash
# Merge LoRA adapter
python merge.py 72b  # or 32b, 14b
```

Output: `./merged/` (full model ready for VLLM)
