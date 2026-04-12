echo "Testing LLaDA-8B-Instruct model with batch size 2 on port 8088"
start_ts=$(date +"%s")
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GSAI-ML/LLaDA-8B-Instruct",
    "messages": [{"role":"user","content":"Explain diffusion models in one paragraph."}],
    "max_tokens": 128,
    "steps": 128,
    "block_size": 128,
    "temperature": 0.2,
    "cfg_scale": 0.0,
    "remasking": "low_confidence"
  }'&
wait
echo ""
echo "All requests completed."
end_ts=$(date +"%s")
cost_time=$(( end_ts - start_ts ))

echo "Response time: $cost_time seconds"
