#!/bin/bash
# Single GPU training for testing
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export RUN_SINGLE_GPU_TEST=1
export RUN_IMPORT_TEST=1
./start.sh 14b