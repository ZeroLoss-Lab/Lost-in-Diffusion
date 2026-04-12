import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import time
import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_PATH = "inclusionAI/LLaDA2.0-mini"
DEVICE = "cuda:4"  # Keeping the device from your snippet
PORT = 9090

# --- Model Loading ---
print(f"Loading model from {MODEL_PATH} to {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, device_map=DEVICE
)
model = model.to(torch.bfloat16)
model.eval()
print("Model loaded successfully.")

# --- FastAPI App ---
app = FastAPI(title="LLaDA OpenAI-compatible Server")

# --- Pydantic Models for OpenAI API ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "llada"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    # Custom parameters for LLaDA if needed, can be passed but might not be standard OpenAI
    block_length: Optional[int] = 32
    steps: Optional[int] = 32

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

# --- Endpoints ---

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "inclusionAI",
            }
        ]
    }

@app.post("/v1/", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not currently supported.")

    # Convert messages to list of dicts
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Tokenize input
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization error: {str(e)}")

    # Generate
    # Mapping OpenAI params to model.generate params
    # Note: LLaDA seems to use specific params like block_length and steps
    gen_kwargs = {
        "inputs": input_ids,
        "eos_early_stop": True,
        "gen_length": request.max_tokens,
        "block_length": request.block_length,
        "steps": request.steps,
        "temperature": request.temperature,
    }

    with torch.no_grad():
        generated_tokens = model.generate(**gen_kwargs)

    # Decode output
    # generated_tokens includes input_ids, so we might need to slice if the model returns full sequence
    # However, chat.py uses generated_tokens[0] directly. Let's check if we need to strip input.
    # Usually generate returns input + output. 
    # Let's assume standard behavior: we need to extract the new tokens.
    
    # If the model returns only new tokens, we use it as is. 
    # If it returns full sequence, we slice. 
    # Based on chat.py: tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    # It prints the answer. If it printed the prompt too, chat.py would show that.
    # Assuming generated_tokens contains the full sequence (common in HF), 
    # but let's stick to exactly what chat.py did first, then refine if needed.
    # Actually, chat.py prints `generated_answer`. If `generated_tokens` contained the prompt, 
    # `generated_answer` would contain the prompt.
    # To be safe for an API, we usually want just the assistant response.
    
    output_ids = generated_tokens[0]
    # Simple heuristic: if output starts with input, slice it.
    if len(output_ids) > len(input_ids[0]) and torch.equal(output_ids[:len(input_ids[0])], input_ids[0]):
         output_ids = output_ids[len(input_ids[0]):]

    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Construct Response
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionResponseUsage(
            prompt_tokens=len(input_ids[0]),
            completion_tokens=len(output_ids),
            total_tokens=len(input_ids[0]) + len(output_ids)
        )
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
