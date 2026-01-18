# RunPod Serverless Training Container
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    torch \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    huggingface_hub

# Copy training scripts
COPY start.sh setup.sh run.sh train.py merge.py handler.py ./

# Make scripts executable
RUN chmod +x start.sh setup.sh run.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1

# RunPod Serverless entry point
CMD ["python", "-u", "handler.py"]
