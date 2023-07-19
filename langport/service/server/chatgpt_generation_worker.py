import argparse
import os
import random
import uvicorn
import uuid

from langport.workers.generation_worker import GenerationModelWorker
from langport.utils import build_logger
from langport.routers.server.generation_node import app


# We suggest that concurrency == batch * thread (thread == 4)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--worker-address", type=str, default=None)
    parser.add_argument("--neighbors", type=str, nargs="*", default=[])

    
    parser.add_argument("--api-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api-key", type=str)

    parser.add_argument("--model-name", default="gpt-3.5-turbo", type=str, help="Optional display name")
    parser.add_argument("--limit-model-concurrency", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--stream-interval", type=int, default=2)
    args = parser.parse_args()

    node_id = str(uuid.uuid4())
    logger = build_logger("langport.service.chatgpt_generation_worker", f"chatgpt_generation_worker_{node_id}.log")
    logger.info(f"args: {args}")

    if args.port is None:
        args.port = random.randint(21001, 29001)

    if args.worker_address is None:
        args.worker_address = f"http://{args.host}:{args.port}"
    
    if args.model_name is None:
        args.model_name = os.path.basename(os.path.normpath(args.model_path))
    

    from langport.model.executor.generation.chatgpt import ChatGPTGenerationExecutor
    executor = ChatGPTGenerationExecutor(
        model_name=args.model_name,
        api_url=args.api_url,
        api_key=args.api_key
    )

    app.node = GenerationModelWorker(
        node_addr=args.worker_address,
        node_id=node_id,
        init_neighborhoods_addr=args.neighbors,
        executor=executor,
        limit_model_concurrency=args.limit_model_concurrency,
        max_batch=args.batch,
        stream_interval=args.stream_interval,
        logger=logger,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
