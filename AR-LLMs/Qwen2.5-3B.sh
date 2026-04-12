export CUDA_VISIBLE_DEVICES=4
vllm serve Qwen/Qwen2.5-3B-Instruct --tensor-parallel-size 1 --port 9000