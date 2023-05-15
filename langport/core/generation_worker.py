import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import Iterable, List, Optional, Union
import threading
import uuid
import traceback

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from tenacity import retry, stop_after_attempt
from langport.core.model_worker import ModelWorker
from langport.model.executor.base import BaseModelExecutor

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    EmbeddingWorkerResult,
    EmbeddingsTask,
    GenerationTask,
    GenerationWorkerResult,
    RegisterWorkerRequest,
    RemoveWorkerRequest,
    UsageInfo,
    WorkerStatus,
)

import torch

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxNewTokensCriteria,
)
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    TopKLogitsWarper,
)
from langport.constants import (
    GENERATION_INFERENCE_INTERVAL,
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.model.model_adapter import load_model
from langport.utils import server_error_msg, pretty_print_semaphore


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def stop_by_stopwords(
    output: str, rfind_start: int, stop: Optional[Union[str, List[str]]]
) -> int:
    if stop is not None:
        if isinstance(stop, str):
            pos = output.rfind(stop, rfind_start)
            if pos != -1:
                return pos
        elif isinstance(stop, Iterable):
            for each_stop in stop:
                pos = output.rfind(each_stop, rfind_start)
                if pos != -1:
                    return pos
        else:
            raise ValueError("Invalid stop field type.")
    return -1


@torch.inference_mode()
def batch_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    stream_interval: int,
    tasks: List[GenerationTask],
):
    batch_size = len(tasks)
    if batch_size == 0:
        return

    prompts = [task.prompt for task in tasks]
    max_new_tokens = max([task.max_new_tokens for task in tasks])

    # init logits_processor
    logits_processor_list = []
    for task in tasks:
        logits_processor = prepare_logits_processor(
            task.temperature, task.repetition_penalty, task.top_p, task.top_k
        )
        logits_processor_list.append(logits_processor)

    # prepare init inputs
    inputs = []
    length = []
    for i in range(batch_size):
        each_inputs = tokenizer(prompts[i], return_tensors="pt")
        each_input_ids = each_inputs.input_ids.squeeze(0)
        inputs.append(each_input_ids)
        length.append(len(each_input_ids))

    # padding to max(length)
    if tokenizer._pad_token is None:
        pad_fill_id = tokenizer.eos_token_id
    else:
        pad_fill_id = tokenizer.pad_token_id
    full_input_ids = torch.full(
        (batch_size, max(length)), pad_fill_id, dtype=torch.long, device=device
    )
    for i in range(batch_size):
        full_input_ids[i, : length[i]] = inputs[i]

    # needed variables
    input_ids = full_input_ids[:, : min(length)]
    if model.config.is_encoder_decoder:
        encoder_outputs = model.encoder(input_ids=full_input_ids)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            model.generation_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
    else:
        encoder_outputs = None
        decoder_input_ids = input_ids
    past_key_values = None

    # decode state
    is_stop = [False] * batch_size

    # step by step
    for step in range(max_new_tokens):
        if model.config.is_encoder_decoder:
            out = model(
                input_ids=input_ids,
                use_cache=True,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
            )
        else:
            out = model(
                input_ids=decoder_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
        logits = out.logits
        past_key_values = out.past_key_values

        new_ids = []
        current_len = input_ids.shape[1]
        for i in range(batch_size):
            if is_stop[i]:
                continue
            task = tasks[i]
            last_token_logits = logits[i][-1]

            logits_processor = logits_processor_list[i]
            if logits_processor:
                if task.repetition_penalty > 1.0:
                    tmp_output_ids = input_ids[i, :].unsqueeze(0)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[
                    0
                ]
            else:
                last_token_logits = logits[0, -1, :]

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if task.temperature < 1e-5 or task.top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            if current_len < length[i]:
                new_ids.append(full_input_ids[i, current_len])
            else:
                new_ids.append(token)

            if task.stop_token_ids is not None and token in task.stop_token_ids:
                is_stop[i] = True

            if token == tokenizer.eos_token_id or step == max_new_tokens - 1:
                is_stop[i] = True

        new_ids_tensor = torch.tensor(
            new_ids, dtype=torch.long, device=input_ids.device
        ).unsqueeze(1)

        input_ids = torch.cat(
            (input_ids, new_ids_tensor),
            dim=1,
        )
        decoder_input_ids = new_ids_tensor

        for i in range(batch_size):
            task = tasks[i]
            if step % stream_interval == 0 or is_stop[i]:
                if tasks[i].echo:
                    tmp_output_ids = input_ids[i, :]
                    rfind_start = length[i]
                else:
                    tmp_output_ids = input_ids[i, length[i] :]
                    rfind_start = 0
                output = tokenizer.decode(tmp_output_ids, skip_special_tokens=True)

                # stop by stopwords
                stop_pos = stop_by_stopwords(output, rfind_start, task.stop)
                if stop_pos != -1:
                    is_stop[i] = True
                    output = output[:stop_pos]

                # yield result
                yield GenerationWorkerResult(
                    task_id=task.task_id,
                    type="data",
                    text=output,
                    usage=UsageInfo(
                        prompt_tokens=length[i],
                        total_tokens=length[i] + step,
                        completion_tokens=step,
                    ),
                    finish_reason=None,
                )

            if is_stop[i]:
                if step == max_new_tokens - 1:
                    finish_reason = "length"
                else:
                    finish_reason = "stop"
                yield GenerationWorkerResult(
                    task_id=task.task_id,
                    type="finish",
                    text=output,
                    usage=UsageInfo(
                        prompt_tokens=length[i],
                        total_tokens=length[i] + step,
                        completion_tokens=step,
                    ),
                    finish_reason=finish_reason,
                )

        if all(is_stop):
            break

    del past_key_values


def inference_generation(worker: "ModelWorker", deadline_tick: bool):
    if not worker.online:
        return
    
    if not deadline_tick and worker.num_tasks() < worker.max_batch:
        return

    tasks = worker.fetch_tasks()
    batch_size = len(tasks)
    if batch_size == 0:
        return

    for chunk in batch_generation(
        worker.model_holder.model,
        worker.model_holder.tokenizer,
        worker.device,
        worker.stream_interval,
        tasks,
    ):
        worker.push_task_result(chunk.task_id, chunk)

    for task in tasks:
        worker.push_task_result(
            task.task_id, BaseWorkerResult(task_id=task.task_id, type="done")
        )


class GenerationModelWorker(ModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        worker_type: str,
        executor: BaseModelExecutor,
        limit_model_concurrency: int,
        max_batch: int,
        stream_interval: int,
        logger,
    ):
        super(GenerationModelWorker, self).__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            worker_type=worker_type,
            executor=executor,
            limit_model_concurrency=limit_model_concurrency,
            max_batch=max_batch,
            stream_interval=stream_interval,
            logger=logger,
        )
        workers = max(1, 2 * self.limit_model_concurrency // self.max_batch)
        self.add_timer(
            "generation_inference",
            GENERATION_INFERENCE_INTERVAL,
            inference_generation,
            args=[self, True],
            kwargs=None,
            workers=workers,
        )

        self.add_timer(
            "generation_inference",
            0.05,
            inference_generation,
            args=[self, False],
            kwargs=None,
            workers=workers,
        )

    def generation_stream(self, task: GenerationTask):
        self.add_task(task)
        for chunk in self.fetch_task_result(task.task_id):
            yield chunk

    def generation_bytes_stream(self, task: GenerationTask):
        for chunk in self.generation_stream(task):
            yield json.dumps(chunk.dict()).encode() + b"\0"
