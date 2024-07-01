import json
from typing import List

from langport.core.cluster_worker import ClusterWorker
from langport.model.executor.generation import GenerationExecutor

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    GenerationTask,
    GenerationWorkerResult,
    UsageInfo,
)

from langport.constants import (
    GENERATION_INFERENCE_INTERVAL,
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.utils import server_error_msg, pretty_print_semaphore

class GenerationModelWorker(ClusterWorker):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        init_neighborhoods_addr: List[str],
        executor: GenerationExecutor,
        limit_model_concurrency: int,
        max_batch: int,
        stream_interval: int,
        logger
    ):
        super(GenerationModelWorker, self).__init__(
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
            "generation_inference",
            GENERATION_INFERENCE_INTERVAL,
            self.executor.inference,
            args=[self,],
            kwargs=None,
            workers=workers,
        )
    
        self.on_start("set_features", self.set_features)
        self.on_start("set_model_name", self.set_model_name)

    async def set_features(self):
        await self.set_local_state("features", ["generation"], ttl=360)
    
    async def set_model_name(self):
        await self.set_local_state("model_name", self.executor.model_name, ttl=360)

    async def generation_stream(self, task: GenerationTask):
        prompt_tokens = len(self.executor.tokenize(task.prompt))
        max_tokens = task.max_tokens
        context_length = self.executor.context_length

        if context_length is not None and prompt_tokens + max_tokens > context_length:
            ooc_message = f"This model's maximum context length is {context_length} tokens. "
            f"However, you requested {max_tokens + prompt_tokens} tokens "
            f"({prompt_tokens} in the messages, "
            f"{max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion."
            self.logger.info(ooc_message)
            yield BaseWorkerResult(task_id=task.task_id,
                                   type="error",
                                   message=ooc_message,
                                    error_code=ErrorCode.CONTEXT_OVERFLOW
                                   )
            return

        await self.add_task(task)
        try:
            async for chunk in self.fetch_task_result(task.task_id):
                yield chunk
        except Exception as e:
            self.logger.error(str(e))
            yield BaseWorkerResult(task_id=task.task_id,
                        type="error",
                        message=str(e),
                        error_code=ErrorCode.INTERNAL_ERROR
            )

    async def generation_bytes_stream(self, task: GenerationTask):
        async for chunk in self.generation_stream(task):
            yield json.dumps(chunk.dict()).encode() + b"\0"
