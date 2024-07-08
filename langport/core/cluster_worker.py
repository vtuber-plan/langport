import asyncio
from collections import defaultdict
import logging
import json
import re
from typing import Dict, List, Optional

import queue
from langport.core.cluster_node import ClusterNode

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    BaseWorkerTask,
    WorkerAddressRequest,
    WorkerAddressResponse,
)

from langport.constants import (
    HEART_BEAT_EXPIRATION,
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.utils.evaluation import safe_eval

class ClusterWorker(ClusterNode):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        init_neighborhoods_addr: List[str],
        limit_model_concurrency: int,
        max_batch: int,
        stream_interval: int,
        logger: logging.Logger,
    ):
        super(ClusterWorker, self).__init__(
            node_addr=node_addr,
            node_id=node_id,
            init_neighborhoods_addr=init_neighborhoods_addr,
            logger=logger,
        )

        self.limit_model_concurrency = limit_model_concurrency
        self.max_batch = max_batch
        self.stream_interval = stream_interval

        self.global_counter = 0
        self.model_semaphore = None

        self.task_queue: queue.Queue[BaseWorkerTask] = queue.Queue()
        self.output_queue: Dict[str, queue.Queue[BaseWorkerResult]] = defaultdict(
            queue.Queue
        )

        self.on_start("set_queue_state", self.set_queue_state)
        self.on_start("set_features", self.set_features)
        self.on_start("set_speed", self.set_speed)

    async def set_queue_state(self):
        await self.set_local_state("queue_length", self.get_queue_length(), ttl=16)

    async def set_features(self):
        await self.set_local_state("features", [], ttl=360)
    
    async def set_speed(self):
        await self.set_local_state("speed", 1.0, ttl=360)

    def release_model_semaphore(self):
        self.model_semaphore.release()

    def acquire_model_semaphore(self):
        self.global_counter += 1
        if self.model_semaphore is None:
            self.model_semaphore = asyncio.Semaphore(self.limit_model_concurrency)
        return self.model_semaphore.acquire()

    def get_queue_length(self):
        if (
            self.model_semaphore is None
            or self.model_semaphore._value is None
            or self.model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.task_queue.qsize()
                + len(self.model_semaphore._waiters)
            )

    async def add_task(self, task: BaseWorkerTask):
        self.task_queue.put(task, block=True, timeout=WORKER_API_TIMEOUT)
        self.output_queue[task.task_id] = queue.Queue()
        await self.set_queue_state()

    async def fetch_task_result(self, task_id: str):
        result_queue = self.output_queue[task_id]
        retry_counter = 0
        while True:
            await self.set_queue_state()
            try:
                event = result_queue.get(block=False, timeout=None)
            except queue.Empty:
                await asyncio.sleep(0.01)
                retry_counter += 1
                # If client disconnected, stop to wait queue.
                if retry_counter > 60 * 100:
                    raise ValueError("Worker task execution timeout")
                else:
                    continue
            retry_counter = 0
            if event.type == "done":
                break
            elif event.type == "error":
                yield event
                break
            elif event.type == "finish":
                yield event
                break
            elif event.type == "data":
                yield event
            else:
                raise ValueError("Bad chunk type.")

        del self.output_queue[task_id]

    def get_num_tasks(self) -> int:
        return self.task_queue.qsize()

    def fetch_tasks(self, task_num: Optional[int]=None) -> List[BaseWorkerResult]:
        if task_num is None:
            task_num = self.max_batch
        task_batch = []
        while len(task_batch) < task_num:
            try:
                task = self.task_queue.get(block=False, timeout=None)
            except queue.Empty:
                break
            task_batch.append(task)
        return task_batch

    def push_task_result(self, task_id: str, response: BaseWorkerResult):
        self.output_queue[task_id].put(response, block=True, timeout=WORKER_API_TIMEOUT)

    async def api_get_worker_address(self, request: WorkerAddressRequest) -> WorkerAddressResponse:
        id_list, address_list, values = await self.get_worker_address(request.condition, request.expression)
        return WorkerAddressResponse(
            id_list=id_list,
            address_list=address_list,
            values=values,
        )
    
    async def get_worker_address(self, condition: str, expression: str) -> Optional[str]:
        condition_variables = re.findall(r'\{(.*?)\}', condition)
        expression_variables = re.findall(r'\{(.*?)\}', expression)

        worker_ids = []
        worker_address = []
        worker_values = []
        for w_id, w_info in self.neighborhoods.items():
            final_condition = condition
            final_condition_variables = {}
            for v_name in condition_variables:
                variable_value = await self.get_node_state(w_info.node_id, v_name)
                final_condition = final_condition.replace("{" + v_name + "}", v_name)
                final_condition_variables[v_name] = variable_value
            
            final_expression = expression
            final_expression_variables = {}
            for v_name in expression_variables:
                variable_value = await self.get_node_state(w_info.node_id, v_name)
                final_expression = final_expression.replace("{" + v_name + "}", v_name)
                final_expression_variables[v_name] = variable_value

            if safe_eval(final_condition, final_condition_variables):
                worker_ids.append(w_id)
                worker_address.append(w_info.node_addr)
                value_json = json.dumps(safe_eval(final_expression, final_expression_variables))
                worker_values.append(value_json)
        
        return worker_ids, worker_address, worker_values
                