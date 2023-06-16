import argparse
import uuid
import uvicorn

from langport.core.cluster_worker import ClusterWorker
from langport.utils import build_logger
from langport.routers.server.core_node import app
from langport.utils.cache_state import CacheState

logger = build_logger("langport.service.dummy_worker", "dummy_worker.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--neighbors", type=str, nargs="*", default=[])
    args = parser.parse_args()
    logger.info(f"args: {args}")

    node_id = str(uuid.uuid4())
    node_addr = f"http://{args.host}:{args.port}"
    app.node = ClusterWorker(
        node_addr=node_addr,
        node_id=node_id,
        init_neighborhoods_addr=args.neighbors,
        limit_model_concurrency=32,
        max_batch=1,
        stream_interval=2,
        logger=logger,
    )
    app.node.states["model_name"] = CacheState("dummy_node", 360)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
