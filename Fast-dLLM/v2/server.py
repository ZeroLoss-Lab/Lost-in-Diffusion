import asyncio
import os
import types
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_functions import (
	FAST_DLLM_MASK_ID,
	FAST_DLLM_STOP_TOKEN,
	Fast_dLLM_QwenForCausalLM,
)


MODEL_NAME = os.getenv("FAST_DLLM_V2_MODEL", "Efficient-Large-Model/Fast_dLLM_v2_7B")
DEVICE = torch.device(os.getenv("FAST_DLLM_V2_DEVICE", "cuda:4" if torch.cuda.is_available() else "cpu"))

model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
gpu_lock = asyncio.Lock()


class ChatRequest(BaseModel):
	messages: List[Dict[str, str]]
	max_new_tokens: int = Field(default=1024, gt=0, description="Number of new tokens to sample")
	block_size: int = Field(default=32, gt=0, description="Main block size for diffusion decoding")
	small_block_size: int = Field(default=8, gt=0, description="Inner block size for refinement")
	threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence threshold for accepting tokens")
	top_p: float = Field(default=0.95, gt=0.0, le=1.0)
	temperature: float = Field(default=0.0, ge=0.0)
	use_block_cache: bool = Field(default=False, description="Enable Fast-dLLM block cache optimization")


class ChatResponse(BaseModel):
	response: str
	token_count: int


def _ensure_pad_token(tok: AutoTokenizer) -> None:
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token


def _prepare_inputs(message_batch: List[Dict[str, str]]) -> torch.Tensor:
	assert tokenizer is not None
	prompt = tokenizer.apply_chat_template(
		message_batch,
		tokenize=False,
		add_generation_prompt=True,
	)
	encoded = tokenizer([prompt], return_tensors="pt")
	return encoded["input_ids"].to(DEVICE)


def _pad_to_block(input_ids: torch.Tensor, block_size: int) -> torch.Tensor:
	pad = (block_size - (input_ids.shape[1] % block_size)) % block_size
	if pad == 0:
		return input_ids
	pad_tokens = torch.full(
		(input_ids.shape[0], pad),
		FAST_DLLM_MASK_ID,
		dtype=torch.long,
		device=input_ids.device,
	)
	return torch.cat([input_ids, pad_tokens], dim=1)


def _decode_completion(sample: torch.Tensor, seq_len: int) -> tuple[str, int]:
	assert tokenizer is not None
	completion = sample[seq_len:].detach().cpu().tolist()
	if FAST_DLLM_STOP_TOKEN in completion:
		stop_idx = completion.index(FAST_DLLM_STOP_TOKEN)
		completion = completion[:stop_idx]
	filtered = [t for t in completion if t not in (FAST_DLLM_MASK_ID, tokenizer.pad_token_id)]
	text = tokenizer.decode(filtered, skip_special_tokens=True).strip()
	return text, len(filtered)


@asynccontextmanager
async def lifespan(app: FastAPI):
	global model, tokenizer
	try:
		dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
		tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
		_ensure_pad_token(tokenizer)
		model = AutoModelForCausalLM.from_pretrained(
			MODEL_NAME,
			trust_remote_code=True,
			torch_dtype=dtype,
		).to(DEVICE)
		model.eval()
		model.mdm_sample = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample, model)
		yield
	finally:
		if torch.cuda.is_available():
			torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
	if model is None or tokenizer is None:
		raise HTTPException(status_code=503, detail="Model not initialized")

	async with gpu_lock:
		try:
			print("input messages:", request.messages)
			input_ids = _prepare_inputs(request.messages)
			prompt_len = int(input_ids.shape[1])
			seq_len = torch.tensor([prompt_len], device=DEVICE)
			min_len = prompt_len
			# padded_inputs = _pad_to_block(input_ids, request.block_size)
			# print("")
			# print(f"Input shape after padding: {input_ids.shape}")
			# print("----------")
			generated = model.mdm_sample(
				input_ids=input_ids,
				tokenizer=tokenizer,
				block_size=request.block_size,
				small_block_size=request.small_block_size,
				max_new_tokens=request.max_new_tokens,
				mask_id=FAST_DLLM_MASK_ID,
				min_len=min_len,
				seq_len=seq_len,
				use_block_cache=request.use_block_cache,
				threshold=request.threshold,
				top_p=request.top_p,
				temperature=request.temperature,
			)

			sample = generated[0]
			response_text, token_count = _decode_completion(sample, prompt_len)
			return ChatResponse(response=response_text, token_count=token_count)
		except Exception as exc:  # pragma: no cover
			print("Error during generation:", str(exc))
			raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
	host = os.getenv("FAST_DLLM_V2_HOST", "0.0.0.0")
	port = int(os.getenv("FAST_DLLM_V2_PORT", "8070"))
	uvicorn.run("server:app", host=host, port=port, log_level="info")
