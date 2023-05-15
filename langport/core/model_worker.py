import argparse
import asyncio
from collections import defaultdict
import dataclasses
import logging
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
import threading
import queue
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from tenacity import retry, stop_after_attempt
from langport.core.base_worker import BaseWorker
from langport.model.model_holder import LanguageModelHolder

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    BaseWorkerTask,
    RegisterWorkerRequest,
    RemoveWorkerRequest,
    WorkerStatus,
)

from langport.constants import (
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.model.model_adapter import load_model
from langport.utils import server_error_msg, pretty_print_semaphore
from langport.utils.interval_timer import IntervalTimer


class ModelWorker(BaseWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        worker_type: str,
        model_path: str,
        model_name: str,
        device: str,
        num_gpus: int,
        max_gpu_memory,
        load_8bit: bool,
        cpu_offloading: bool,
        limit_model_concurrency: int,
        max_batch: int,
        stream_interval: int,
        logger: logging.Logger,
    ):
        super(ModelWorker, self).__init__(
            controller_addr = controller_addr,
            worker_addr = worker_addr,
            worker_id = worker_id,
            worker_type = worker_type,
            logger = logger,
        )

        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.load_8bit = load_8bit
        self.cpu_offloading = cpu_offloading
        self.limit_model_concurrency = limit_model_concurrency
        self.max_batch = max_batch
        self.stream_interval = stream_interval
        
        self.global_counter = 0
        self.model_semaphore = None

        self.logger.info(
            f"Loading the model {self.model_name} on worker {worker_id} ..."
        )

        self.model_holder = LanguageModelHolder(
            model_path=model_path,
            model_name=model_name,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading
        )

        self.context_len = self.model_holder.context_len

        self.task_queue: queue.Queue[BaseWorkerTask] = queue.Queue()
        self.output_queue: Dict[str, queue.Queue[BaseWorkerResult]] = defaultdict(queue.Queue)

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
                self.limit_model_concurrency
                - self.model_semaphore._value
                + len(self.model_semaphore._waiters)
            )

    def get_status(self) -> WorkerStatus:
        return WorkerStatus(
            worker_id=self.worker_id,
            model_name=self.model_name,
            speed=1,
            queue_length=self.get_queue_length(),
        )
    
    def add_task(self, task: BaseWorkerTask):
        self.task_queue.put(task, block=True, timeout=WORKER_API_TIMEOUT)
        self.output_queue[task.task_id] = queue.Queue()
    
    def fetch_task_result(self, task_id: str):
        result_queue = self.output_queue[task_id]
        retry_counter = 0
        while True:
            try:
                event = result_queue.get(block=False, timeout=None)
            except queue.Empty:
                time.sleep(0.01)
                retry_counter += 1
                # If client disconnected, stop to wait queue.
                if retry_counter > 2000:
                    break
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

    def num_tasks(self) -> int:
        return self.task_queue.qsize()
    
    def fetch_tasks(self) -> List[BaseWorkerResult]:
        task_batch = []
        while len(task_batch) <= self.max_batch:
            try:
                task = self.task_queue.get(block=False, timeout=None)
            except queue.Empty:
                break
            task_batch.append(task)
        return task_batch
    
    def push_task_result(self, task_id: str, response: BaseWorkerResult):
        self.output_queue[task_id].put(response, block=True, timeout=WORKER_API_TIMEOUT)