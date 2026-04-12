export CUDA_VISIBLE_DEVICES=4
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 1 --port 8071