import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import List, Optional, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from tenacity import retry, stop_after_attempt
from langport.core.model_worker import ModelWorker

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    EmbeddingWorkerResult,
    EmbeddingsTask,
    RegisterWorkerRequest,
    RemoveWorkerRequest,
    UsageInfo,
    WorkerStatus,
)

import torch

from langport.constants import (
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    EMBEDDING_INFERENCE_INTERVAL,
    ErrorCode,
)
from langport.model.model_adapter import load_model
from langport.utils import server_error_msg, pretty_print_semaphore


def inference_embeddings(worker: "ModelWorker"):
    if not worker.online:
        return
    tasks = worker.fetch_tasks()
    batch_size = len(tasks)
    if batch_size == 0:
        return
    
    prompts = [task.input for task in tasks]
    try:
        tokenizer = worker.model_holder.tokenizer
        model = worker.model_holder.model
        input_ids = tokenizer.encode(prompts, return_tensors="pt").to(worker.device)
        model_output = model(input_ids, output_hidden_states=True)
        is_chatglm = "chatglm" in str(type(model)).lower()
        if is_chatglm:
            data = model_output.hidden_states[-1].transpose(0, 1)
        else:
            data = model_output.hidden_states[-1]
        embeddings = torch.mean(data, dim=1)
        for i in range(batch_size):
            token_num = len(tokenizer(prompts[i]).input_ids)
            worker.push_task_result(
                tasks[i].task_id,
                EmbeddingWorkerResult(
                    task_id=tasks[i].task_id,
                    type="data",
                    embedding=embeddings[i].tolist(),
                    usage=UsageInfo(prompt_tokens=token_num, total_tokens=token_num),
                )
            )

    except torch.cuda.OutOfMemoryError:
        for i in range(batch_size):
            worker.push_task_result(
                tasks[i].task_id,
                BaseWorkerResult(
                    task_id=tasks[i].task_id,
                    type="error",
                    message="Cuda out of Memory Error"
                )
            )
    except Exception as e:
        for i in range(batch_size):
            worker.push_task_result(
                tasks[i].task_id,
                BaseWorkerResult(
                    task_id=tasks[i].task_id,
                    type="error",
                    message=str(e)
                )
            )
    
    for i in range(batch_size):
        worker.push_task_result(
            tasks[i].task_id,
            BaseWorkerResult(
                task_id=tasks[i].task_id,
                type="done",
            )
        )


class EmbeddingModelWorker(ModelWorker):
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
        logger,
    ):
        super(EmbeddingModelWorker, self).__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            worker_type=worker_type,
            model_path=model_path,
            model_name=model_name,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            limit_model_concurrency=limit_model_concurrency,
            max_batch=max_batch,
            stream_interval=stream_interval,
            logger=logger,
        )
        workers = max(1, 2 * self.limit_model_concurrency // self.max_batch)
        self.add_timer(
            "embeddings_inference", 
            EMBEDDING_INFERENCE_INTERVAL, 
            inference_embeddings, 
            args=(self,), 
            kwargs=None, 
            workers=workers
        )

    def get_embeddings(self, task: EmbeddingsTask):
        self.add_task(task)
        result = None
        for chunk in self.fetch_task_result(task.task_id):
            result = chunk
        return result
