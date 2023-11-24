from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from langport.protocol.worker_protocol import EmbeddingsTask, GenerationTask
from .core_node import app, create_background_tasks


@app.post("/chat_stream")
async def api_chat_stream(request: Request):
    params = await request.json()
    await app.node.acquire_model_semaphore()
    generator = app.node.generation_bytes_stream(GenerationTask(
        prompt=params["prompt"],
        temperature=params.get("temperature", 1.0),
        repetition_penalty=params.get("repetition_penalty", 0.0),
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", 0),
        max_tokens=params.get("max_tokens", 512),
        stop=params.get("stop", None),
        echo=params.get("echo", False),
        stop_token_ids=params.get("stop_token_ids", None),
    ))
    background_tasks = create_background_tasks(app.node)
    return StreamingResponse(generator, background=background_tasks)

@app.post("/chat")
async def api_chat(request: Request):
    params = await request.json()
    await app.node.acquire_model_semaphore()
    generator = await app.node.generation_stream(GenerationTask(
        prompt=params["prompt"],
        temperature=params.get("temperature", 1.0),
        repetition_penalty=params.get("repetition_penalty", 0.0),
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", 0),
        max_tokens=params.get("max_tokens", 512),
        stop=params.get("stop", None),
        echo=params.get("echo", False),
        stop_token_ids=params.get("stop_token_ids", None),
    ))
    completion = None
    for chunk in generator:
        completion = chunk
    background_tasks = create_background_tasks(app.node)
    return JSONResponse(content=completion.dict(), background=background_tasks)



@app.post("/completion_stream")
async def api_completion_stream(request: Request):
    params = await request.json()
    await app.node.acquire_model_semaphore()
    generator = app.node.generation_bytes_stream(GenerationTask(
        prompt=params["prompt"],
        temperature=params.get("temperature", 1.0),
        repetition_penalty=params.get("presence_penalty", 0.0),
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", 1),
        max_tokens=params.get("max_tokens", 512),
        stop=params.get("stop", None),
        echo=params.get("echo", False),
        stop_token_ids=params.get("stop_token_ids", None),
        logprobs=params.get("logprobs", None),
    ))
    background_tasks = create_background_tasks(app.node)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/completion")
async def api_completion(request: Request):
    params = await request.json()
    await app.node.acquire_model_semaphore()
    generator = await app.node.generation_stream(GenerationTask(
        prompt=params["prompt"],
        temperature=params.get("temperature", 1.0),
        repetition_penalty=params.get("presence_penalty", 0.0),
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", 1),
        max_tokens=params.get("max_tokens", 512),
        stop=params.get("stop", None),
        echo=params.get("echo", False),
        stop_token_ids=params.get("stop_token_ids", None),
        logprobs=params.get("logprobs", None),
    ))
    completion = None
    for chunk in generator:
        completion = chunk
    background_tasks = create_background_tasks(app.node)
    return JSONResponse(content=completion.dict(), background=background_tasks)
