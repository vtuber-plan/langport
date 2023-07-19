import argparse
import os
import random
import uuid
import warnings
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
    parser.add_argument("--stream-interval", type=int, default=2)

    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--lib", type=str, default=None, choices=["avx2", "avx", "basic"], help="The path to a shared library or one of avx2, avx, basic.")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "gpt2", "dolly-v2", "starcoder"], help="The type of model to use.")
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
    
    if args.load_8bit or args.load_4bit:
        warnings.warn("The ggml backend does not yet support quantization parameters.")

    if args.port is None:
        args.port = random.randint(21001, 29001)

    if args.worker_address is None:
        args.worker_address = f"http://{args.host}:{args.port}"
    
    if args.model_name is None:
        args.model_name = os.path.basename(os.path.normpath(args.model_path))
    
    from langport.model.executor.generation.ggml import GgmlGenerationExecutor
    executor = GgmlGenerationExecutor(
        model_name=args.model_name,
        model_path=args.model_path,
        context_length=args.context_length,
        gpu_layers=args.gpu_layers,
        lib=args.lib,
        model_type=args.model_type,
        chunk_size=args.chunk_size,
        threads=args.threads,
    )

    app.node = GenerationModelWorker(
        node_addr=args.worker_address,
        node_id=node_id,
        init_neighborhoods_addr=args.neighbors,
        executor=executor,
        limit_model_concurrency=args.limit_model_concurrency,
        max_batch=1,
        stream_interval=args.stream_interval,
        logger=logger,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
