import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

from generate import generate


MASK_TOKEN_ID = 126336


class ChatMessage(BaseModel):
	role: str
	content: str


class ChatCompletionRequest(BaseModel):
	model: str = Field(default="GSAI-ML/LLaDA-8B-Instruct")
	messages: List[ChatMessage]
	temperature: float = Field(default=0.0, ge=0.0)
	max_tokens: int = Field(default=1024, gt=0)
	steps: int = Field(default=1024, gt=0)
	block_size: int = Field(default=1024, gt=0)
	cfg_scale: float = Field(default=0.0, ge=0.0)
	remasking: str = Field(default="low_confidence")


class ChatChoice(BaseModel):
	index: int
	message: ChatMessage
	finish_reason: str


class Usage(BaseModel):
	prompt_tokens: int
	completion_tokens: int
	total_tokens: int


class ChatCompletionResponse(BaseModel):
	id: str
	object: str
	created: int
	model: str
	choices: List[ChatChoice]
	usage: Usage


class BatchManager:
	"""Aggregate OpenAI-style requests so the model sees larger batches."""

	def __init__(
		self,
		model: AutoModel,
		tokenizer: AutoTokenizer,
		device: torch.device,
		max_batch_size: int = 4,
		max_wait_ms: float = 20.0,
	) -> None:
		self.model = model
		self.tokenizer = tokenizer
		self.device = device
		self.max_batch_size = max_batch_size
		self.max_wait_ms = max_wait_ms
		self.queue: asyncio.Queue[Tuple[ChatCompletionRequest, asyncio.Future]] = asyncio.Queue()
		self.worker_task: asyncio.Task | None = None
		self.running = False

	async def start(self) -> None:
		self.running = True
		self.worker_task = asyncio.create_task(self._loop())

	async def stop(self) -> None:
		self.running = False
		if self.worker_task:
			await self.worker_task

	async def submit(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
		loop = asyncio.get_running_loop()
		future: asyncio.Future = loop.create_future()
		await self.queue.put((request, future))
		return await future

	async def _loop(self) -> None:
		while self.running:
			try:
				first = await asyncio.wait_for(self.queue.get(), timeout=0.1)
			except asyncio.TimeoutError:
				continue

			batch: List[Tuple[ChatCompletionRequest, asyncio.Future]] = [first]
			start = time.time()

			while len(batch) < self.max_batch_size:
				remaining = self.max_wait_ms / 1000 - (time.time() - start)
				if remaining <= 0:
					break
				try:
					batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
				except asyncio.TimeoutError:
					break

			await self._process(batch)

	async def _process(self, batch: List[Tuple[ChatCompletionRequest, asyncio.Future]]) -> None:
		grouped: Dict[Tuple[Any, ...], List[Tuple[ChatCompletionRequest, asyncio.Future]]] = {}
		for req, fut in batch:
			key = (
				req.model,
				req.steps,
				req.max_tokens,
				req.block_size,
				req.temperature,
				req.cfg_scale,
				req.remasking,
			)
			grouped.setdefault(key, []).append((req, fut))

		for key, items in grouped.items():
			requests = [r for r, _ in items]
			futures = [f for _, f in items]
			try:
				prompts = []
				for req in requests:
					message_dicts = [message.dict() for message in req.messages]
					prompts.append(
						self.tokenizer.apply_chat_template(
							message_dicts,
							add_generation_prompt=True,
							tokenize=False,
						)
					)

				encoded = self.tokenizer(
					prompts,
					add_special_tokens=False,
					padding=True,
					return_tensors="pt",
				)

				input_ids = encoded["input_ids"].to(self.device)
				attention_mask = encoded["attention_mask"].to(self.device)
				prompt_token_counts = attention_mask.sum(dim=1).tolist()

				gen_length = key[2]
				block_length = min(key[3], gen_length)
				if gen_length % block_length != 0:
					block_length = gen_length

				num_blocks = max(1, gen_length // block_length)
				steps = key[1]
				if steps % num_blocks != 0:
					steps = max(num_blocks, (steps // num_blocks) * num_blocks)
					if steps == 0:
						steps = num_blocks

				outputs = generate(
					self.model,
					input_ids,
					tokenizer=self.tokenizer,
					attention_mask=attention_mask,
					steps=steps,
					gen_length=gen_length,
					block_length=block_length,
					temperature=key[4],
					cfg_scale=key[5],
					remasking=key[6],
					mask_id=MASK_TOKEN_ID,
				)

				completion_ids = outputs[:, input_ids.shape[1] : input_ids.shape[1] + gen_length]
				completion_token_counts = [gen_length] * completion_ids.shape[0]
				texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

				created_ts = int(time.time())
				for idx, (fut, req, prompt_tokens, completion_tokens, text) in enumerate(
					zip(futures, requests, prompt_token_counts, completion_token_counts, texts)
				):
					if fut.done():
						continue

					response = ChatCompletionResponse(
						id=f"chatcmpl-{uuid.uuid4().hex}",
						object="chat.completion",
						created=created_ts,
						model=req.model,
						choices=[
							ChatChoice(
								index=0,
								message=ChatMessage(role="assistant", content=text.strip()),
								finish_reason="length",
							)
						],
						usage=Usage(
							prompt_tokens=int(prompt_tokens),
							completion_tokens=int(completion_tokens),
							total_tokens=int(prompt_tokens + completion_tokens),
						),
					)
					fut.set_result(response)

			except Exception as exc:  # pragma: no cover - propagate errors
				for fut in futures:
					if not fut.done():
						fut.set_exception(exc)


DEVICE = torch.device(os.getenv("LLADA_DEVICE", "cuda:6" if torch.cuda.is_available() else "cpu"))
MODEL_PATH = os.getenv("LLADA_MODEL", "GSAI-ML/LLaDA-8B-Instruct")
MAX_BATCH_SIZE = int(os.getenv("LLADA_MAX_BATCH", "32"))
MAX_WAIT_MS = float(os.getenv("LLADA_MAX_WAIT_MS", "20"))

batch_manager: BatchManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
	global batch_manager
	tokenizer = None
	model = None
	try:
		tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
		if tokenizer.padding_side != "left":
			tokenizer.padding_side = "left"
		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token

		model = AutoModel.from_pretrained(
			MODEL_PATH,
			trust_remote_code=True,
			torch_dtype=torch.bfloat16,
		).to(DEVICE)
		model.eval()

		batch_manager = BatchManager(
			model=model,
			tokenizer=tokenizer,
			device=DEVICE,
			max_batch_size=MAX_BATCH_SIZE,
			max_wait_ms=MAX_WAIT_MS,
		)
		await batch_manager.start()
		yield
	finally:
		if batch_manager:
			await batch_manager.stop()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
	if batch_manager is None:
		raise HTTPException(status_code=503, detail="Model not ready")
	if request.max_tokens <= 0:
		raise HTTPException(status_code=400, detail="max_tokens must be positive")
	# print('Submitting request to batch manager')
	return await batch_manager.submit(request)


if __name__ == "__main__":
	host = os.getenv("LLADA_HOST", "127.0.0.1")
	port = int(os.getenv("LLADA_PORT", "8088"))
	uvicorn.run("server:app", host=host, port=port, log_level="info")
