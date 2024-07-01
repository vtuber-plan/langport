from typing import List, Optional, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from tenacity import retry, stop_after_attempt
from langport.core.cluster_worker import ClusterWorker
from langport.model.executor.base import BaseModelExecutor
from langport.model.executor.embedding import EmbeddingExecutor

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    EmbeddingWorkerResult,
    EmbeddingsTask,
    UsageInfo,
)
import traceback

import torch

from langport.constants import (
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    EMBEDDING_INFERENCE_INTERVAL,
    ErrorCode,
)
from langport.utils import server_error_msg, pretty_print_semaphore

class EmbeddingModelWorker(ClusterWorker):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        init_neighborhoods_addr: List[str],
        executor: EmbeddingExecutor,
        limit_model_concurrency: int,
        max_batch: int,
        stream_interval: int,
        logger
    ):
        super(EmbeddingModelWorker, self).__init__(
            node_addr=node_addr,
            node_id=node_id,
            init_neighborhoods_addr=init_neighborhoods_addr,
            limit_model_concurrency=limit_model_concurrency,
            max_batch=max_batch,
            stream_interval=stream_interval,
            logger=logger,
        )
        self.executor = executor
        workers = max(1, self.limit_model_concurrency)
        self.add_timer(
            "embeddings_inference", 
            EMBEDDING_INFERENCE_INTERVAL, 
            executor.inference, 
            args=(self,), 
            kwargs=None, 
            workers=workers
        )
        
        self.on_start("set_features", self.set_features)
        self.on_start("set_model_name", self.set_model_name)

    async def set_features(self):
        await self.set_local_state("features", ["embedding"], ttl=360)
    
    async def set_model_name(self):
        await self.set_local_state("model_name", self.executor.model_name, ttl=360)

    async def get_embeddings(self, task: EmbeddingsTask):
        input_tokens = len(self.executor.tokenize(task.input))
        context_length = self.executor.context_length

        if input_tokens > context_length:
            ooc_message = f"This model's maximum context length is {context_length} tokens. "
            f"However, you requested {input_tokens} tokens. "
            f"Please reduce the length of the messages or completion."
            self.logger.info(ooc_message)
            return BaseWorkerResult(task_id=task.task_id,
                                   type="error",
                                   message=ooc_message,
                                    error_code=ErrorCode.CONTEXT_OVERFLOW
                                   )

        await self.add_task(task)
        result = None
        try:
            async for chunk in self.fetch_task_result(task.task_id):
                result = chunk
        except Exception as e:
            self.logger.error(ooc_message)
            return BaseWorkerResult(task_id=task.task_id,
                        type="error",
                        message=str(e),
                        error_code=ErrorCode.INTERNAL_ERROR
            )

        return result
