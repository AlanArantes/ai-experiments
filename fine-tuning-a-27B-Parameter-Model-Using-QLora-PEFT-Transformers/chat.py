import os
from dotenv import load_dotenv

# Load and sanitize environment variables
load_dotenv()
if "HF_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"].strip()

import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest


# ---------------------------------------------------------------------------
# 1. Schemas & Models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    model: Optional[str] = Field(
        default=None,
        description="Target model or adapter name (e.g., 'customer-support', 'coder', or None for base model)",
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(256, ge=1, le=4096)
    stop: Optional[list[str]] = None
    stream: bool = Field(
        default=False,
        description="Whether to stream response tokens via Server-Sent Events (SSE). Default is False (returns a complete JSON response).",
    )


# ---------------------------------------------------------------------------
# 2. Dynamic Adapter Registry
# ---------------------------------------------------------------------------
# Register known local adapters. lora_int_id MUST be unique per adapter.
ADAPTER_REGISTRY: Dict[str, LoRARequest] = {
    "qlora-adapter": LoRARequest(
        lora_name="qlora-adapter",
        lora_int_id=1,
        lora_path="./qlora-adapter-output",
    ),
}


# ---------------------------------------------------------------------------
# 3. Lifespan Manager (Engine Initialization & Teardown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-3B-Instruct",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_lora=True,
        max_loras=1,  # Max concurrent LoRAs computed per batch
        max_lora_rank=16,  # Accommodates the trained LoRA rank
        disable_log_requests=True,  # Cleaner logging in production
    )
    # Store engine in app state for access across endpoints
    app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield
    # Cleanup / graceful shutdown if needed
    del app.state.engine


app = FastAPI(title="vLLM Dynamic LoRA Streaming Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 4. SSE Streaming Generator with Early Abort
# ---------------------------------------------------------------------------
async def event_generator(
    engine: AsyncLLMEngine,
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest],
    http_request: Request,
) -> AsyncGenerator[str, None]:
    results_generator = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=lora_request,
    )

    prev_text = ""
    try:
        async for output in results_generator:
            # Check for client disconnect during streaming
            if await http_request.is_disconnected():
                await engine.abort(request_id)
                break

            current_text = output.outputs[0].text
            delta = current_text[len(prev_text) :]
            prev_text = current_text

            chunk = {
                "id": request_id,
                "delta": delta,
                "finish_reason": output.outputs[0].finish_reason,
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        # Abort the GPU work if an unhandled error interrupts generation
        await engine.abort(request_id)
        error_payload = {"error": str(e)}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 5. Production Endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/generate")
@app.post("/generate")
async def generate(req_body: GenerateRequest, request: Request):
    engine: AsyncLLMEngine = request.app.state.engine
    request_id = f"gen-{uuid.uuid4().hex}"

    # Resolve LoRA adapter
    lora_req = None
    if req_body.model:
        if req_body.model in ADAPTER_REGISTRY:
            lora_req = ADAPTER_REGISTRY[req_body.model]
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model/Adapter '{req_body.model}' not found. Available: {list(ADAPTER_REGISTRY.keys())}",
            )

    sampling_params = SamplingParams(
        temperature=req_body.temperature,
        top_p=req_body.top_p,
        max_tokens=req_body.max_tokens,
        stop=req_body.stop,
    )

    # 1. Streaming response via Server-Sent Events (SSE)
    if req_body.stream:
        return StreamingResponse(
            event_generator(
                engine=engine,
                request_id=request_id,
                prompt=req_body.prompt,
                sampling_params=sampling_params,
                lora_request=lora_req,
                http_request=request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disables proxy buffering (e.g., NGINX)
            },
        )

    # 2. Standard complete JSON response
    final_output = None
    try:
        async for output in engine.generate(
            prompt=req_body.prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_req,
        ):
            if await request.is_disconnected():
                await engine.abort(request_id)
                raise HTTPException(status_code=499, detail="Client disconnected")
            final_output = output
    except Exception as e:
        await engine.abort(request_id)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

    if final_output is None or not final_output.outputs:
        raise HTTPException(status_code=500, detail="Generation completed without output")

    output_choice = final_output.outputs[0]
    prompt_tokens = len(final_output.prompt_token_ids) if hasattr(final_output, "prompt_token_ids") and final_output.prompt_token_ids else None
    completion_tokens = len(output_choice.token_ids) if hasattr(output_choice, "token_ids") and output_choice.token_ids else None

    return {
        "id": request_id,
        "model": req_body.model or "base-model",
        "text": output_choice.text,
        "finish_reason": output_choice.finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (prompt_tokens + completion_tokens) if (prompt_tokens and completion_tokens) else None,
        } if prompt_tokens is not None else None,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
