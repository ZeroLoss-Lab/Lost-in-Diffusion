curl -X POST http://localhost:8088/v1/ \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
    { "role": "user", 
      "content": "Whats a diffusion model?" }
    ],
    "steps": 512,
    "max_tokens": 512,
    "block_size": 32,
    "temperature": 0.2
  }'