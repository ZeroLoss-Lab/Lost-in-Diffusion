import torch
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from contextlib import asynccontextmanager
import os

# ==========================================
# Helper Functions (from app.py)
# ==========================================

def _sanitize_probabilities(probs: torch.Tensor) -> torch.Tensor:
    """Ensure probs (1D or 2D) contain finite, normalized values."""
    if not torch.is_floating_point(probs):
        return probs
    sanitized = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sanitized = sanitized.clamp_min(0.0)
    if sanitized.ndim == 1:
        total = sanitized.sum()
        if total <= 0:
            return torch.full_like(sanitized, 1.0 / sanitized.numel())
        return sanitized / total
    total = sanitized.sum(dim=-1, keepdim=True)
    zero_mask = total <= 0
    if zero_mask.any():
        sanitized = sanitized.clone()
        expanded_mask = zero_mask.expand(-1, sanitized.size(-1))
        sanitized[expanded_mask] = 1.0
        total = sanitized.sum(dim=-1, keepdim=True)
    return sanitized / total


def _flatten_probs_if_needed(probs: torch.Tensor):
    if probs.ndim <= 2:
        return probs, probs.shape
    orig_shape = probs.shape
    flat = probs.contiguous().view(-1, orig_shape[-1])
    return flat, orig_shape


def _reshape_samples_if_needed(samples: torch.Tensor, orig_shape, num_samples: int):
    if len(orig_shape) <= 2:
        return samples
    return samples.view(*orig_shape[:-1], num_samples)


_original_multinomial = torch.multinomial


def _safe_multinomial(probs, num_samples, replacement=False, *, generator=None, out=None):
    flat_probs, orig_shape = _flatten_probs_if_needed(probs)
    normalized = _sanitize_probabilities(flat_probs)
    samples = _original_multinomial(normalized, num_samples, replacement, generator=generator, out=out)
    return _reshape_samples_if_needed(samples, orig_shape, num_samples)


if not getattr(torch.multinomial, "_dream_safe", False):
    _safe_multinomial._dream_safe = True  # type: ignore[attr-defined]
    torch.multinomial = _safe_multinomial  # type: ignore[assignment]

# ==========================================
# FastAPI configuration
# ==========================================

# Define global variables
model = None
tokenizer = None
gpu_lock = asyncio.Lock() # Core: control concurrency to 1
DEVICE = 'cuda:5' if torch.cuda.is_available() else 'cpu' # Using cuda:5 as in app.py

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] # [{"role": "user", "content": "..."}]
    steps: int = Field(default=128, description="Sampling steps")
    max_new_tokens: int = Field(default=128, description="Generated length")
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 0
    alg: str = "entropy"
    alg_temp: float = 0.1

class ChatResponse(BaseModel):
    response: str
    token_count: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Loading Dream model to {DEVICE}...")
    
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    
    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True).to(DEVICE).eval()
        
        print("Model loaded successfully!")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/v1/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global model, tokenizer
    
    # Core concurrency control: acquire lock before inference
    async with gpu_lock:
        try:
            # 1. Process input template
            inputs = tokenizer.apply_chat_template(
                request.messages, 
                return_tensors="pt", 
                return_dict=True, 
                add_generation_prompt=True
            )
            
            input_ids = inputs.input_ids.to(DEVICE)
            attention_mask = inputs.attention_mask.to(DEVICE)
            prompt_length = input_ids.shape[1]
            
            effective_top_k = request.top_k if request.top_k > 0 else None

            # 3. Run Dream diffusion generation process
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=request.steps,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=effective_top_k, 
                alg=request.alg,
                alg_temp=request.alg_temp
            )
            
            # 4. Decode output
            # Only take the generated suffix part
            final_tokens_tensor = output.sequences[0][prompt_length:]
            final_tokens_list = final_tokens_tensor.tolist()
            output_text = tokenizer.decode(final_tokens_list, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            
            return ChatResponse(
                response=output_text,
                token_count=len(final_tokens_list)
            )
            
        except Exception as e:
            print(f"Inference Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # Start the service, listen on port 8067 (different from LLaDA's 8066)
    uvicorn.run(app, host="0.0.0.0", port=8067, log_level="info")
