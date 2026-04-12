export CUDA_VISIBLE_DEVICES=4,5
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --tensor-parallel-size 2 --port 9000