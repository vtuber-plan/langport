import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from tenacity import retry, stop_after_attempt

from langport.protocol.worker_protocol import RegisterWorkerRequest, RemoveWorkerRequest, WorkerHeartbeat, WorkerStatus

import torch

from langport.constants import WORKER_API_TIMEOUT, WORKER_HEART_BEAT_INTERVAL, WORKER_HEART_BEAT_CHECK_INTERVAL, ErrorCode
from langport.model.model_adapter import load_model, add_model_args
from langport.core.inference import generate_stream
from langport.utils import server_error_msg, pretty_print_semaphore

def heart_beat_worker(controller: "ModelWorker"):
    last_time = time.time()
    while controller.online:
        time.sleep(WORKER_HEART_BEAT_CHECK_INTERVAL)
        now_time = time.time()
        if now_time - last_time > WORKER_HEART_BEAT_INTERVAL:
            try:
                controller.send_heart_beat()
            except requests.exceptions.RequestException as e:
                controller.logger.error(f"heart beat error: {e}")
            last_time = now_time


class ModelWorker(object):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_name: str,
        device: str,
        num_gpus: int,
        max_gpu_memory,
        load_8bit: bool,
        cpu_offloading: bool,
        limit_model_concurrency: int,
        stream_interval: int,
        logger,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]
        self.device = device
        self.limit_model_concurrency = limit_model_concurrency
        self.stream_interval=stream_interval
        self.logger = logger

        self.global_counter = 0
        self.model_semaphore = None


        self.logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.model, self.tokenizer = load_model(
            model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading
        )

        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        else:
            self.context_len = 2048

        self.generate_stream_func = generate_stream
    
        self.online = False
    
    def start(self):
        if self.online:
            return
        
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,)
        )
        self.heart_beat_thread.start()
        self.register_to_controller()

        self.online = True

    def stop(self):
        if not self.online:
            return
        
        self.remove_from_controller()

        self.online = False

    def register_to_controller(self):
        self.logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = RegisterWorkerRequest(
            worker_id=self.worker_id,
            worker_addr=self.worker_addr,
            check_heart_beat=True,
            worker_status=WorkerStatus(
                model_name=self.model_name,
                speed=1,
                queue_length=self.get_queue_length()
            )
        )
        r = requests.post(url, json=data.dict(), timeout=WORKER_API_TIMEOUT)
        assert r.status_code == 200
    
    def remove_from_controller(self):
        self.logger.info("Remove to controller")

        url = self.controller_addr + "/remove_worker"
        data = RemoveWorkerRequest(
            worker_id=self.worker_id
        )
        r = requests.post(url, json=data.dict(), timeout=WORKER_API_TIMEOUT)
        assert r.status_code == 200

    @retry(stop=stop_after_attempt(5))
    def send_heart_beat(self):
        self.logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {pretty_print_semaphore(self.model_semaphore)}. "
            f"global_counter: {self.global_counter}"
        )

        url = self.controller_addr + "/receive_heart_beat"

        ret = requests.post(
            url,
            json=WorkerHeartbeat(
                worker_id=self.worker_id,
                status=WorkerStatus(
                    model_name=self.model_name,
                    speed=1,
                    queue_length=self.get_queue_length(),
                )
            ),
            timeout=WORKER_API_TIMEOUT,
        )
        exist = ret.json()["exist"]

        if not exist:
            self.register_to_controller()

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

    def generate_stream(self, params):
        try:
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except ValueError as e:
            ret = {
                "text": str(e),
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        yield json.dumps(ret).encode() + b"\0"
    
    def generate(self, params):
        try:
            ret = {
                "text": "",
                "error_code": 0
            }
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret["text"] = output["text"]
            if "usage" in output:
                ret["usage"] = output["usage"]
            if "finish_reason" in output:
                ret["finish_reason"] = output["finish_reason"]
            if "logprobs" in output:
                ret["logprobs"] = output["logprobs"]
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except ValueError as e:
            ret = {
                "text": str(e),
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret
    
    def get_embeddings(self, params):
        try:
            tokenizer = self.tokenizer
            input_ids = tokenizer.encode(params["input"], return_tensors="pt").to(
                self.device
            )
            model_output = self.model(input_ids, output_hidden_states=True)
            is_chatglm = "chatglm" in str(type(self.model)).lower()
            if is_chatglm:
                data = (model_output.hidden_states[-1].transpose(0, 1))[0]
            else:
                data = model_output.hidden_states[-1][0]
            embedding = torch.mean(data, dim=0)
            return json.dumps(
                {
                    "embedding": embedding.tolist(),
                    "token_num": len(self.tokenizer(params["input"]).input_ids),
                }
            )
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            return json.dumps(ret).encode() + b"\0"

