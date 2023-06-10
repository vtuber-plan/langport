from typing import Iterable, List, Optional, Union

from langport.model.executor.huggingface import HuggingfaceExecutor

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    GenerationTask,
    GenerationWorkerResult,
    UsageInfo,
)

import torch

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    TopKLogitsWarper,
)
from langport.workers.generation_worker import GenerationModelWorker

from cachetools import LRUCache, TTLCache
from asyncache import cached

from typing import Optional

import torch

@cached(LRUCache(maxsize=64))
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
    # print(batch_size)
    
    # collect params
    prompts = [task.prompt for task in tasks]
    max_tokens = [task.max_tokens for task in tasks]

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
    start_infer_pos = min(length)
    input_ids = full_input_ids[:, : start_infer_pos].clone()
    if model.config.is_encoder_decoder:
        max_len = max(length)
        attention_mask = torch.ones(batch_size, max_len, dtype=torch.long, device=device)
        for i, each_length in enumerate(length):
            attention_mask[i, each_length:] = 0

        encoder_outputs = model.encoder(input_ids=full_input_ids, attention_mask=attention_mask)
        decoder_input_ids_list = [torch.full(
            (batch_size, 1),
            model.generation_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )]
    else:
        encoder_outputs = None
        decoder_input_ids_list = [input_ids]
    past_key_values = None

    # decode state
    is_stop = [False] * batch_size

    max_new_tokens = max([length[i] + max_tokens[i] for i in range(batch_size)]) - min(length)
    # step by step
    for step in range(max_new_tokens):
        # inference a step
        if len(decoder_input_ids_list) > 1:
            decoder_input_ids = torch.stack(decoder_input_ids_list, dim=1)
        elif len(decoder_input_ids_list) == 1:
            decoder_input_ids = decoder_input_ids_list[0]
        else:
            raise Exception("decoder_input_ids_list length is 0")

        if model.config.is_encoder_decoder:
            out = model.decoder(
                input_ids=decoder_input_ids,
                use_cache=True,
                encoder_outputs=encoder_outputs,
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
        decoder_input_ids_list = []

        # if we can skip some steps
        need_skip = []
        for i, task in enumerate(tasks):
            if not is_stop[i]:
                if step + start_infer_pos < length[i]:
                    need_skip.append(True)
                else:
                    need_skip.append(False)
        
        if all(need_skip) and len(need_skip) < batch_size:
            new_ids = []
            current_len = input_ids.shape[1]
            for i, task in enumerate(tasks):
                if is_stop[i]:
                    new_ids.append(pad_fill_id)
                else:
                    new_ids.append(full_input_ids[i, current_len])
            
            new_ids_tensor = torch.tensor(
                new_ids, dtype=torch.long, device=input_ids.device
            ).unsqueeze(1)

            input_ids = torch.cat(
                (input_ids, new_ids_tensor),
                dim=1,
            )

            decoder_input_ids_list.append(new_ids_tensor)
            continue

        # create new ids
        new_ids = []
        current_len = input_ids.shape[1]

        for i, task in enumerate(tasks):
            if is_stop[i]:
                new_ids.append(pad_fill_id)
                continue
            each_logits = logits[i, -1, :].unsqueeze(0)

            logits_processor = logits_processor_list[i]
            if logits_processor:
                if task.repetition_penalty > 1.0:
                    tmp_output_ids = input_ids[i, :].unsqueeze(0)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, each_logits)[0]
            else:
                last_token_logits = each_logits

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if task.temperature < 1e-5 or task.top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            if current_len < length[i]:
                new_token = full_input_ids[i, current_len]
            else:
                new_token = token
            new_ids.append(new_token)

            if task.stop_token_ids is not None and new_token in task.stop_token_ids:
                is_stop[i] = True

            if new_token == tokenizer.eos_token_id:
                is_stop[i] = True

            if current_len == length[i] + task.max_tokens - 1:
                is_stop[i] = True

        new_ids_tensor = torch.tensor(
            new_ids, dtype=torch.long, device=input_ids.device
        ).unsqueeze(1)

        input_ids = torch.cat(
            (input_ids, new_ids_tensor),
            dim=1,
        )

        decoder_input_ids_list = [new_ids_tensor]

        for i, task in enumerate(tasks):
            if not is_stop[i] and step % stream_interval != 0:
                continue

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
                    total_tokens=start_infer_pos + step,
                    completion_tokens=start_infer_pos + step - length[i],
                ),
                finish_reason=None,
            )

            if is_stop[i]:
                if current_len == length[i] + task.max_tokens - 1:
                    finish_reason = "length"
                else:
                    finish_reason = "stop"
                yield GenerationWorkerResult(
                    task_id=task.task_id,
                    type="finish",
                    text=output,
                    usage=UsageInfo(
                        prompt_tokens=length[i],
                        total_tokens=start_infer_pos + step,
                        completion_tokens=start_infer_pos + step - length[i],
                    ),
                    finish_reason=finish_reason,
                )

        if all(is_stop):
            break

    del past_key_values

class HuggingfaceGenerationExecutor(HuggingfaceExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        load_8bit: bool,
        cpu_offloading: bool,
        deepspeed: bool = False,
        trust_remote_code: bool = False
    ) -> None:
        super(HuggingfaceGenerationExecutor, self).__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading
        )
        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.adapter, self.model, self.tokenizer = self.load_model(
            model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, deepspeed, trust_remote_code
        )

        # self.model = torch.compile(self.model)

        if hasattr(self.model.config, "max_sequence_length"):
            self._context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self._context_len = self.model.config.max_position_embeddings
        else:
            self._context_len = 2048

    @property
    def context_length(self) -> int:
        return self._context_len
    
    def tokenize(self, text: str) -> List[int]:
        input_ids = self.tokenizer(text).input_ids
        return input_ids
    
    def inference(self, worker: "GenerationModelWorker"):
        if not worker.online:
            return

        tasks = worker.fetch_tasks()
        batch_size = len(tasks)
        if batch_size == 0:
            return

        # batch inference
        for chunk in batch_generation(
            self.model,
            self.tokenizer,
            self.device,
            worker.stream_interval,
            tasks,
        ):
            worker.push_task_result(chunk.task_id, chunk)

        for task in tasks:
            worker.push_task_result(
                task.task_id, BaseWorkerResult(task_id=task.task_id, type="done")
            )
  