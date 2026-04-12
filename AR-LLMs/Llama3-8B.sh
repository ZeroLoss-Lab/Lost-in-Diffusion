export CUDA_VISIBLE_DEVICES=7
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 1 --port 9003