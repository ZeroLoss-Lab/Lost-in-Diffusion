# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

GPU=7
MODELS=(
    # "GSAI-ML/LLaDA-8B-Instruct"
    "GSAI-ML/LLaDA-8B-Base"
    # "Dream-org/Dream-v0-Instruct-7B"
    # "Dream-org/Dream-v0-Base-7B"
    # "inclusionAI/LLaDA2.0-mini-2"
    # "inclusionAI/Ling-mini-2.0"
    # "Efficient-Large-Model/Fast_dLLM_v2_7B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3-8B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "meta-llama/Llama-3.1-405B-Instruct-FP8"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b-it"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mistral-Nemo-Instruct-2407"
    # "claude-3-sonnet"
    # "claude-3-haiku"
    # "gpt-4o"
)

EXP_MODE=longwiki
for MODEL in "${MODELS[@]}"
do  
    CUDA_VISIBLE_DEVICES=$GPU python3 -m tasks.longwiki.longwiki_main \
        --exp_mode $EXP_MODE \
        --do_generate_prompt \
        --do_inference \
        --do_eval \
        --model $MODEL\
        --inference_method "custom"\
        --N 250 \
        --db_path "data/wiki_data/.cache/enwiki-20230401.db" \
        --q_generator gemini-2.5-flash
done
