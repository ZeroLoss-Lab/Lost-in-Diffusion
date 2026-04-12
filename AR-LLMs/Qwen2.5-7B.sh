export CUDA_VISIBLE_DEVICES=7
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --port 9002