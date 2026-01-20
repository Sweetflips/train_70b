#!/usr/bin/env python3
"""
Demo script for Qwen2.5-Coder-32B-Instruct fine-tuned with LoRA.

This script demonstrates how to use the trained model for inference.
Supports both the merged model and the LoRA checkpoint directly.

Usage:
    # Using merged model (recommended - faster loading, no PEFT required)
    python demo.py --merged

    # Using LoRA checkpoint directly (requires PEFT)
    python demo.py --lora

    # Interactive chat mode
    python demo.py --merged --interactive

    # Single prompt
    python demo.py --merged --prompt "Write a Python function to calculate fibonacci"

    # With streaming output
    python demo.py --merged --prompt "Explain quicksort" --stream
"""

import argparse
import gc
import os
import sys
from typing import Generator, Optional


def log(msg: str) -> None:
    """Print a log message with prefix."""
    print(f"[DEMO] {msg}", flush=True)


def get_device_info() -> dict:
    """Get information about available compute devices."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }

    if info["cuda_available"]:
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
            })

    return info


def load_merged_model(model_path: str = "./merged_model"):
    """
    Load the merged model (LoRA weights already merged into base model).
    This is the recommended approach - simpler and no PEFT dependency needed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading merged model from: {model_path}")

    if not os.path.exists(model_path):
        log(f"ERROR: Model path not found: {model_path}")
        log("Please run merge_and_save.py first to create the merged model.")
        sys.exit(1)

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Determine device configuration
    device_info = get_device_info()
    if device_info["cuda_available"]:
        total_vram = sum(d["total_memory_gb"] for d in device_info["devices"])
        log(f"CUDA available: {device_info['device_count']} GPU(s), {total_vram:.1f} GB total VRAM")
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        log("CUDA not available, using CPU (will be slow)")
        device_map = None
        torch_dtype = torch.float32

    # Try to use flash attention if available
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        log("Using flash_attention_2")
    except ImportError:
        attn_impl = "sdpa"
        log("Using SDPA attention (install flash-attn for better performance)")

    # Load model
    log("Loading model (this may take a few minutes for a 32B model)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    log("Model loaded!")

    # Print memory usage
    if device_info["cuda_available"]:
        for i in range(device_info["device_count"]):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            log(f"GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

    return model, tokenizer


def load_lora_model(
    checkpoint_path: str = "./output/checkpoint-2038",
    base_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
):
    """
    Load the base model with LoRA adapter applied.
    Requires PEFT library. Useful for comparing different checkpoints.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"Loading LoRA checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        log(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Read adapter config to get base model
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path", base_model)
        log(f"Base model from config: {base_model}")
        log(f"LoRA config: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")

    # Load tokenizer from checkpoint (has any special tokens)
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Determine device configuration
    device_info = get_device_info()
    if device_info["cuda_available"]:
        total_vram = sum(d["total_memory_gb"] for d in device_info["devices"])
        log(f"CUDA available: {device_info['device_count']} GPU(s), {total_vram:.1f} GB total VRAM")
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        log("CUDA not available, using CPU (will be slow)")
        device_map = None
        torch_dtype = torch.float32

    # Try to use flash attention if available
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        log("Using flash_attention_2")
    except ImportError:
        attn_impl = "sdpa"
        log("Using SDPA attention")

    # Load base model
    log(f"Loading base model: {base_model}")
    log("This may take several minutes...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    log("Base model loaded!")

    # Load LoRA adapter
    log("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base,
        checkpoint_path,
        torch_dtype=torch_dtype,
    )
    log("LoRA adapter loaded!")

    # Print memory usage
    if device_info["cuda_available"]:
        for i in range(device_info["device_count"]):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            log(f"GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Generate a response from the model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to sample (False for greedy decoding)
        stream: Whether to stream the output token by token

    Returns:
        Generated text string, or generator if streaming
    """
    import torch

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        })

    if stream:
        # Streaming generation
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        # Run generation in a separate thread
        generation_thread = Thread(
            target=model.generate,
            kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **gen_kwargs,
            }
        )
        generation_thread.start()

        # Yield tokens as they're generated
        def token_generator():
            for token in streamer:
                yield token
            generation_thread.join()

        return token_generator()

    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response


def run_interactive_chat(model, tokenizer, stream: bool = True):
    """Run an interactive chat session."""
    log("Starting interactive chat...")
    print("\n" + "=" * 60)
    print("Interactive Chat with Qwen2.5-Coder-32B Fine-tuned")
    print("=" * 60)
    print("Type your message and press Enter to get a response.")
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /system <msg> - Set system message")
    print("  /exit or /quit - Exit the chat")
    print("=" * 60 + "\n")

    messages = []
    system_message = "You are a helpful coding assistant. You write clean, efficient, and well-documented code."

    while True:
        try:
            user_input = input("\n[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/exit", "/quit"]:
            print("\nGoodbye!")
            break
        elif user_input.lower() == "/clear":
            messages = []
            print("[System] Conversation history cleared.")
            continue
        elif user_input.lower().startswith("/system "):
            system_message = user_input[8:].strip()
            messages = []  # Clear history when system message changes
            print(f"[System] System message set to: {system_message}")
            continue

        # Build messages for this turn
        turn_messages = []
        if system_message:
            turn_messages.append({"role": "system", "content": system_message})
        turn_messages.extend(messages)
        turn_messages.append({"role": "user", "content": user_input})

        # Generate response
        print("\n[Assistant]: ", end="", flush=True)

        try:
            if stream:
                response_text = ""
                for token in generate_response(
                    model, tokenizer, turn_messages, stream=True
                ):
                    print(token, end="", flush=True)
                    response_text += token
                print()  # Newline after streaming
            else:
                response_text = generate_response(
                    model, tokenizer, turn_messages, stream=False
                )
                print(response_text)

            # Update conversation history
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": response_text})

            # Keep conversation history manageable (last 10 turns)
            if len(messages) > 20:
                messages = messages[-20:]

        except Exception as e:
            print(f"\n[Error] Generation failed: {e}")
            import traceback
            traceback.print_exc()


def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    system_message: Optional[str] = None,
    stream: bool = False,
):
    """Run a single prompt and print the response."""
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": prompt})

    print("\n" + "=" * 60)
    print("PROMPT:")
    print("=" * 60)
    print(prompt)
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)

    if stream:
        for token in generate_response(model, tokenizer, messages, stream=True):
            print(token, end="", flush=True)
        print()
    else:
        response = generate_response(model, tokenizer, messages, stream=False)
        print(response)

    print("=" * 60)


def run_examples(model, tokenizer, stream: bool = False):
    """Run a few example prompts to demonstrate the model."""
    examples = [
        {
            "system": "You are an expert Python developer. Write clean, efficient code with type hints.",
            "prompt": "Write a Python function that implements binary search on a sorted list. Include type hints and docstring.",
        },
        {
            "system": "You are a helpful coding assistant.",
            "prompt": "Explain the difference between `async/await` and threads in Python. When would you use each?",
        },
        {
            "system": "You are an expert software architect.",
            "prompt": "Write a simple REST API endpoint in Python using FastAPI that handles user registration with email and password validation.",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{'#' * 60}")
        print(f"# EXAMPLE {i}")
        print(f"{'#' * 60}")

        run_single_prompt(
            model,
            tokenizer,
            prompt=example["prompt"],
            system_message=example["system"],
            stream=stream,
        )

        # Cleanup between examples
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for Qwen2.5-Coder-32B fine-tuned model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use merged model (recommended)
    python demo.py --merged --interactive

    # Use LoRA checkpoint directly
    python demo.py --lora --checkpoint ./output/checkpoint-2038

    # Single prompt with streaming
    python demo.py --merged --prompt "Write a quicksort in Python" --stream

    # Run built-in examples
    python demo.py --merged --examples
        """,
    )

    # Model loading options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--merged",
        action="store_true",
        help="Use the merged model (recommended - faster, no PEFT needed)",
    )
    model_group.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA checkpoint with base model (requires PEFT)",
    )

    # Paths
    parser.add_argument(
        "--merged-path",
        default="./merged_model",
        help="Path to merged model directory (default: ./merged_model)",
    )
    parser.add_argument(
        "--checkpoint",
        default="./output/checkpoint-2038",
        help="Path to LoRA checkpoint (default: ./output/checkpoint-2038)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="Base model for LoRA (default: Qwen/Qwen2.5-Coder-32B-Instruct)",
    )

    # Interaction modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )
    mode_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to process",
    )
    mode_group.add_argument(
        "--examples",
        action="store_true",
        help="Run built-in example prompts",
    )

    # Generation options
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token by token",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful coding assistant.",
        help="System message for the conversation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, use 0 for greedy)",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("Qwen2.5-Coder-32B Fine-tuned Model Demo")
    print("=" * 60)

    # Load model
    if args.merged:
        model, tokenizer = load_merged_model(args.merged_path)
    else:
        model, tokenizer = load_lora_model(args.checkpoint, args.base_model)

    # Run the appropriate mode
    if args.interactive:
        run_interactive_chat(model, tokenizer, stream=args.stream)
    elif args.prompt:
        run_single_prompt(
            model,
            tokenizer,
            prompt=args.prompt,
            system_message=args.system,
            stream=args.stream,
        )
    elif args.examples:
        run_examples(model, tokenizer, stream=args.stream)
    else:
        # Default: run examples
        log("No mode specified. Running example prompts...")
        run_examples(model, tokenizer, stream=args.stream)

    log("Demo complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
