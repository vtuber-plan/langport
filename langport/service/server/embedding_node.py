from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from langport.protocol.worker_protocol import EmbeddingsTask
from .core_node import app, create_background_tasks


@app.post("/embeddings")
async def api_embeddings(request: EmbeddingsTask):
    await app.node.acquire_model_semaphore()
    embedding = await app.node.get_embeddings(request)
    background_tasks = create_background_tasks(app.node)
    return JSONResponse(content=embedding.dict(), background=background_tasks)
