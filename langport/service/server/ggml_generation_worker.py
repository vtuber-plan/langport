import argparse
import os
import random
import uuid
import uvicorn

from langport.workers.generation_worker import GenerationModelWorker
from langport.model.model_args import add_model_args
from langport.utils import build_logger
from langport.routers.server.generation_node import app


# We suggest that concurrency == batch * thread (thread == 4)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--worker-address", type=str, default=None)
    parser.add_argument("--neighbors", type=str, nargs="*", default=[])

    add_model_args(parser)
    parser.add_argument("--model-name", type=str, help="Optional display name")
    parser.add_argument("--limit-model-concurrency", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--stream-interval", type=int, default=2)

    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-batch", type=int, default=1024)
    parser.add_argument("--last-n-tokens-size", type=int, default=1024)
    args = parser.parse_args()

    node_id = str(uuid.uuid4())
    logger = build_logger("ggml_generation_worker", f"ggml_generation_worker_{node_id}.log")
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.port is None:
        args.port = random.randint(21001, 29001)

    if args.worker_address is None:
        args.worker_address = f"http://{args.host}:{args.port}"
    
    if args.model_name is None:
        args.model_name = os.path.basename(os.path.normpath(args.model_path))
    
    from langport.model.executor.generation.llamacpp import LlamaCppGenerationExecutor
    executor = LlamaCppGenerationExecutor(
        model_name=args.model_name,
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        n_batch=args.n_batch,
        last_n_tokens_size=args.last_n_tokens_size
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
