#!/usr/bin/env python3
"""RunPod Serverless Handler for QLoRA Training."""
import os
import sys
import subprocess
import runpod

def handler(job):
    """
    RunPod Serverless handler for training.
    
    Input:
        model_size: "14b", "32b", or "72b" (default: "72b")
    
    Returns:
        Training status and output location
    """
    job_input = job.get("input", {})
    model_size = job_input.get("model_size", "72b")
    
    # Validate model size
    valid_sizes = ["14b", "32b", "72b"]
    if model_size not in valid_sizes:
        return {"error": f"Invalid model_size. Must be one of: {valid_sizes}"}
    
    print(f"Starting training for model: {model_size}")
    
    try:
        # Run the training script
        result = subprocess.run(
            ["bash", "start.sh", model_size],
            capture_output=True,
            text=True,
            cwd="/app"
        )
        
        if result.returncode != 0:
            return {
                "status": "failed",
                "error": result.stderr,
                "stdout": result.stdout[-5000:] if result.stdout else ""  # Last 5K chars
            }
        
        return {
            "status": "completed",
            "model_size": model_size,
            "output_dir": "/app/output",
            "message": f"Training complete for {model_size}. Adapter saved to /app/output",
            "next_step": f"Run: python merge.py {model_size}"
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

# Start the serverless worker
runpod.serverless.start({"handler": handler})
