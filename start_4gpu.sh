#!/bin/bash
# 4 GPU training for testing
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_IMPORT_TEST=1
./start.sh 32b