export CUDA_VISIBLE_DEVICES=6

vllm serve inclusionAI/Ling-mini-2.0 \
              --tensor-parallel-size 1 \
              --pipeline-parallel-size 1 \
              --port 9093 \
              --trust-remote-code
