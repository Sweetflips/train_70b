#!/usr/bin/env python3
"""
Test importing heavy libraries one by one with memory monitoring.
"""
import os
import sys
import time
import traceback
import gc

def log(msg):
    print(f"[IMPORT_TEST] {msg}", flush=True)
    sys.stdout.flush()

def get_memory_usage():
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        mem_total = int([x for x in meminfo.split('\n') if 'MemTotal' in x][0].split()[1]) // 1024
        mem_avail = int([x for x in meminfo.split('\n') if 'MemAvailable' in x][0].split()[1]) // 1024
        return mem_total - mem_avail, mem_total
    except:
        return 0, 0

def test_import(library_name, import_statement):
    log(f"Testing import: {library_name}")
    used_before, total = get_memory_usage()
    log(f"Memory before: {used_before:,}MB / {total:,}MB")

    start_time = time.time()
    try:
        exec(import_statement)
        import_time = time.time() - start_time
        used_after, _ = get_memory_usage()
        mem_increase = used_after - used_before
        log(f"✅ {library_name} imported successfully in {import_time:.2f}s")
        log(f"Memory increase: {mem_increase:,}MB")
        gc.collect()
        return True
    except Exception as e:
        log(f"❌ {library_name} FAILED: {type(e).__name__}: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        return False

try:
    log("Starting import test...")

    # Set memory limits
    import resource
    soft_limit = 80 * 1024 * 1024 * 1024  # 80GB
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, soft_limit * 2))
    log(f"Set RLIMIT_AS to {soft_limit//(1024**3)}GB")

    # Test each library individually
    tests = [
        ("torch", "import torch"),
        ("transformers", "from transformers import AutoTokenizer"),
        ("peft", "from peft import LoraConfig"),
        ("trl", "from trl import SFTConfig"),
        ("datasets", "from datasets import load_dataset"),
    ]

    passed = 0
    for name, import_stmt in tests:
        if test_import(name, import_stmt):
            passed += 1
        log("")  # Empty line between tests

    log(f"Results: {passed}/{len(tests)} imports successful")

    if passed == len(tests):
        log("🎉 All imports PASSED!")
    else:
        log("⚠️  Some imports failed")
        sys.exit(1)

except Exception as e:
    log(f"FATAL ERROR: {type(e).__name__}: {e}")
    log(f"Traceback:\n{traceback.format_exc()}")
    sys.exit(1)