import argparse
import os
import random
import uuid
import uvicorn

from langport.workers.embedding_worker import EmbeddingModelWorker
from langport.model.executor.embedding.huggingface import HuggingfaceEmbeddingExecutor
from langport.model.model_args import add_model_args
from langport.utils import build_logger
from langport.routers.server.embedding_node import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--worker-address", type=str, default=None)
    parser.add_argument("--neighbors", type=str, nargs="*", default=[])
    
    add_model_args(parser)
    parser.add_argument("--model-name", type=str, help="Optional display name")
    parser.add_argument("--limit-model-concurrency", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--sleep-time", type=int, default=-1, help="Offload model after n seconds")
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
    )
    args = parser.parse_args()

    node_id = str(uuid.uuid4())
    logger = build_logger("embedding_worker", f"embedding_worker_{node_id}.log")
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

    if args.load_8bit:
        quantization = "8bit"
    elif args.load_4bit:
        quantization = "4bit"
    else:
        quantization = None

    executor = HuggingfaceEmbeddingExecutor(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        quantization=quantization,
        cpu_offloading=args.cpu_offloading,
        deepspeed=args.deepspeed,
        gptq=args.gptq,
        group_size=args.group_size,
        trust_remote_code=args.trust_remote_code,
        offload_folder=args.offload_folder,
        sleep_time=args.sleep_time
    )

    app.node = EmbeddingModelWorker(
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
