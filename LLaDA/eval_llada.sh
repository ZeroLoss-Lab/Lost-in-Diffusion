#!/bin/bash
# Set up logging 
mkdir -p "$(dirname "$0")/logs"
LOGFILE="$(dirname "$0")/logs/eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1


export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --num_processes 4 eval_llada.py \
    --tasks truthfulqa_mc2 \
    --num_fewshot 0 \
    --model llada_dist \
    --batch_size 8 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128