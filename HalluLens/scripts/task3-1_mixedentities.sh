# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

MODELS=(
    #  "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "meta-llama/Meta-Llama-3-8B"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "Dream-org/Dream-v0-Base-7B"
    # "Dream-org/Dream-v0-Instruct-7B"
    # "Efficient-Large-Model/Fast_dLLM_v2_7B"
    # "inclusionAI/LLaDA2.0-mini-3"
    "inclusionAI/Ling-mini-2.0"
    # "Dream-org/Dream-v0-Base-7B-2"
    # "GSAI-ML/LLaDA-8B-Instruct"
    # "GSAI-ML/LLaDA-8B-Base"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-405B-Instruct-FP8"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    
)

# exp defualt = nonsense_all 
# Options: [nonsense_medicine nonsense_animal nonsense_plant nonsense_bacteria]

for SEED in 0
do
    for MODEL in "${MODELS[@]}"
    do
        python -m tasks.refusal_test.nonsense_mixed_entities \
            --exp nonsense_all \
            --do_eval \
            --tested_model $MODEL \
            --inference_method custom \
            --N 500 \
            --seed $SEED
    done
done
