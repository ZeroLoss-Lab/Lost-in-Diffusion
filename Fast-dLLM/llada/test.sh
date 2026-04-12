curl -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"简述 diffusion 模型。"}],
    "max_tokens":128,
    "steps":128,
    "block_size":128,
    "use_cache":true,
    "dual_cache":false,
    "threshold":0.9
  }'