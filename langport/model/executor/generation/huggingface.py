import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langport.model.executor.generation import BaseStreamer

from langport.model.executor.huggingface import HuggingfaceExecutor

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    GenerationTask,
    GenerationWorkerLogprobs,
    GenerationWorkerResult,
    UsageInfo,
)

import torch

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils import is_optimum_available
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

def token_to_unicode(token: str) -> str:
    utf8_bytes = token.encode("utf-8")
    # Convert the bytes to a string with \\x escape sequences
    escaped_bytes = "".join([f"\\x{b:02x}" for b in utf8_bytes])
    return escaped_bytes

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

class BatchingTask:
    def __init__(self, tasks: List[GenerationTask], tokenizer: PreTrainedTokenizer, device: str, is_encoder_decoder: bool) -> None:
        self.batch_size = len(tasks)
        if self.batch_size == 0:
            return
        
        self.tokenizer = tokenizer
        self.device = device
        self.is_encoder_decoder = is_encoder_decoder
        
        # collect params
        self.tasks = tasks
        self.prompts_ids = [self.tokenizer(task.prompt, return_tensors="pt").input_ids.squeeze(0) for task in tasks]
        self.prompts_ids_length = [len(i) for i in self.prompts_ids]
        self.min_prompts_length = min(self.prompts_ids_length)
        self.max_prompts_length = max(self.prompts_ids_length)
        self.max_tokens = [task.max_tokens for task in tasks]
        if self.tokenizer._pad_token is None:
            self.pad_fill_id = self.tokenizer.eos_token_id
        else:
            self.pad_fill_id = self.tokenizer.pad_token_id
        
        if self.pad_fill_id is None:
            self.pad_fill_id = 0

        # init logits_processor
        self.logits_processor_list = []
        for task in tasks:
            logits_processor = prepare_logits_processor(
                task.temperature, task.repetition_penalty, task.top_p, task.top_k
            )
            self.logits_processor_list.append(logits_processor)
        
        # variables used in the streaming process
        self.batch_tokens_cache: List[List[int]] = [[] for i in range(self.batch_size)]
        self.batch_tokens_probs_cache: List[List[float]] = [[] for i in range(self.batch_size)]
        self.batch_top_logprobs_cache: List[List[Dict[str, float]]] = [[] for i in range(self.batch_size)]
        self.stop: List[bool] = [False for i in range(self.batch_size)]

    def __len__(self):
        return self.batch_size
    
    def __call__(self, return_attention_mask=True, full_batch=False) -> Any:
        ids: List[torch.LongTensor] = []
        length: List[int] = []
        for i in range(self.batch_size):
            if not full_batch and self.is_stop(i):
                continue
            input_ids = torch.cat((self.prompts_ids[i], torch.LongTensor(self.batch_tokens_cache[i])))
            ids.append(input_ids)
            length.append(len(input_ids))
        
        dyn_batch_size = len(ids)
        # padding to max(length)
        full_input_ids: torch.LongTensor = torch.full(
            (dyn_batch_size, max(length)), self.pad_fill_id,
            dtype=torch.long, device=self.device
        )
        for i in range(dyn_batch_size):
            if self.is_encoder_decoder:
                full_input_ids[i, :length[i]] = ids[i]
            else:
                full_input_ids[i, -length[i]:] = ids[i] # padding side left
        if not return_attention_mask:
            return full_input_ids
        return full_input_ids, self._gen_attention_mask(length)
    
    def _gen_attention_mask(self, length: List[int]) -> torch.Tensor:
        mask = torch.full(
            (self.batch_size, max(length)), 0, dtype=torch.long, device=self.device
        )
        if self.is_encoder_decoder:
            for i in range(self.batch_size):
                mask[i, :length[i]] = 1
        else:
            for i in range(self.batch_size):
                mask[i, -length[i]:] = 1
        return mask
    
    def _check_idx(self, idx:int):
        if idx > self.batch_size:
            raise ValueError("Invalid batch index")
    
    def _check_batch_size(self, lenable):
        if len(lenable) != self.batch_size:
            raise ValueError("Different batch size")
    
    def get_prompt_ids(self, idx:int) -> List[int]:
        self._check_idx(idx)
        return self.prompts_ids[idx]
    
    def get_prompt_length(self, idx:int) -> int:
        return len(self.get_prompt_ids(idx))
    
    def get_logits_processor_list(self, idx:int) -> LogitsProcessorList:
        self._check_idx(idx)
        return self.logits_processor_list[idx]
    
    def get_generated_ids(self, idx: int) -> List[int]:
        self._check_idx(idx)
        return self.batch_tokens_cache[idx]
    
    def get_generated_length(self, idx: int) -> int:
        return len(self.get_generated_ids(idx))
    
    def get_generated_token_probs(self, idx: int) -> List[float]:
        self._check_idx(idx)
        return self.batch_tokens_probs_cache[idx]
    
    def get_generated_top_logprobs(self, idx: int) -> List[Dict[int, float]]:
        self._check_idx(idx)
        return self.batch_top_logprobs_cache[idx]
    
    def update_new_token(self, batch_token: List[int],
            token_probs: Optional[List[Optional[float]]]=None,
            top_logprobs: Optional[List[Optional[Dict[int, float]]]]=None
        ):
        self._check_batch_size(batch_token)
        if token_probs is not None:
            self._check_batch_size(token_probs)
        if top_logprobs is not None:
            self._check_batch_size(top_logprobs)
        
        for i, token in enumerate(batch_token):
            if self.is_stop(i):
                continue
            self.batch_tokens_cache[i].append(token)

            # auto check stop
            if token == self.tokenizer.eos_token_id:
                self.set_stop(i)
            if self.tasks[i].stop_token_ids is not None and token in self.tasks[i].stop_token_ids:
                self.set_stop(i)
            if self.get_generated_length(i) == self.max_tokens[i]:
                self.set_stop(i)
            
            if token_probs is not None and token_probs[i] is not None:
                self.batch_tokens_probs_cache[i].append(token_probs[i])
            if top_logprobs is not None and top_logprobs[i] is not None:
                self.batch_top_logprobs_cache[i].append(top_logprobs[i])
                
    def set_stop(self, idx:int):
        self._check_idx(idx)
        self.stop[idx] = True
    
    def is_stop(self, idx:int):
        self._check_idx(idx)
        return self.stop[idx]

class GenerationModel:
    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
    
    @torch.inference_mode()
    def generate(self, inputs: BatchingTask, 
                 max_new_tokens: int,
                 streamer: Optional[BaseStreamer]=None) -> torch.Tensor:

        if inputs.batch_size == 0:
            return

        full_input_ids, full_attention_mask = inputs(return_attention_mask=True, full_batch=True)
        if self.model.config.is_encoder_decoder:
            full_encoder_outputs = self.model.encoder(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            decoder_input_ids_list: List[torch.LongTensor] = [torch.full(
                (inputs.batch_size, 1),
                self.model.generation_config.decoder_start_token_id,
                dtype=torch.long,
                device=self.model.device,
            )]
        else:
            full_encoder_outputs = None
            decoder_input_ids_list: List[torch.LongTensor] = [full_input_ids]
        
        encoder_outputs = full_encoder_outputs
        attention_mask = full_attention_mask
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        bacth_mapping: List[int] = list(range(inputs.batch_size)) # dynamic batch

        # step by step
        for step in range(max_new_tokens):
            # inference a step
            if len(decoder_input_ids_list) > 1:
                decoder_input_ids = torch.stack(decoder_input_ids_list, dim=1)
            elif len(decoder_input_ids_list) == 1:
                decoder_input_ids = decoder_input_ids_list[0]
            else:
                raise Exception("decoder_input_ids_list length is 0")
            if self.model.config.is_encoder_decoder:
                out = self.model.decoder(
                    input_ids=decoder_input_ids,
                    use_cache=self.model.generation_config.use_cache,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                )
            else:
                if step > 0:
                    dynamic_attention_mask = torch.cat(
                        (attention_mask, 
                        torch.ones(
                            attention_mask.shape[0], step,
                            dtype=torch.long, device=attention_mask.device
                        )), dim=1
                    )
                else:
                    dynamic_attention_mask = attention_mask
                out = self.model(
                    input_ids=decoder_input_ids,
                    attention_mask=dynamic_attention_mask,
                    use_cache=self.model.generation_config.use_cache,
                    past_key_values=past_key_values,
                )
            logits = out.logits
            past_key_values = out.past_key_values
            decoder_input_ids_list = []

            new_ids = []
            # logprobs
            token_probs = [None] * inputs.batch_size
            top_logprobs = [None] * inputs.batch_size

            for task_i, task in enumerate(inputs.tasks):
                if inputs.is_stop(task_i):
                    new_ids.append(inputs.pad_fill_id)
                    continue
                batch_i = bacth_mapping[task_i]
                each_logits = logits[batch_i, -1, :].unsqueeze(0)

                logits_processor = inputs.get_logits_processor_list(task_i)
                if logits_processor:
                    if task.repetition_penalty > 1.0:
                        tmp_output_ids = decoder_input_ids[batch_i, :].unsqueeze(0)
                    else:
                        tmp_output_ids = None
                    last_token_logits = logits_processor(tmp_output_ids, each_logits)[0]
                else:
                    last_token_logits = each_logits[0]

                if self.model.device.type == "mps":
                    # Switch to CPU by avoiding some bugs in mps backend.
                    last_token_logits = last_token_logits.float().to("cpu")

                if task.temperature < 1e-5 or task.top_p < 1e-8:  # greedy
                    token = int(torch.argmax(last_token_logits))
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    sampled_tensor = torch.multinomial(probs, num_samples=2, replacement=False)
                    token = int(sampled_tensor[0].item())
                
                if task.logprobs is not None:
                    token_probs[task_i] = each_logits[0, token].item()
                    top_values, top_indices = torch.topk(each_logits[0, :], task.logprobs, dim=-1, largest=True, sorted=True)
                    item = {}
                    for top_i in range(len(top_values)):
                        item[top_indices[top_i].item()] = top_values[top_i].item()
                    top_logprobs[task_i] = item
                new_ids.append(token)
            
            is_stop_before = [s for s in inputs.stop]
            # update state
            inputs.update_new_token(new_ids, token_probs=token_probs, top_logprobs=top_logprobs)
            if streamer:
                streamer.put(new_ids)
            is_stop_after = [s for s in inputs.stop]
            stop_event = is_stop_before != is_stop_after
            
            # setup next step input and cache
            if self.model.config.is_encoder_decoder or self.model.generation_config.use_cache:
                # print("use cache!")
                dynamic_new_ids = [d for i, d in enumerate(new_ids) if not inputs.is_stop(i)]
                new_ids_tensor = torch.tensor(dynamic_new_ids, dtype=torch.long, device=decoder_input_ids.device).unsqueeze(1)
                decoder_input_ids_list = [new_ids_tensor]
            else:
                decoder_input_ids = inputs(return_attention_mask=False, full_batch=False)
                decoder_input_ids_list = [decoder_input_ids]
            
            # shrink encoder_outputs
            if self.model.config.is_encoder_decoder and stop_event:
                last_hidden_state = encoder_outputs[0]
                new_pos = 0
                for task_i in range(inputs.batch_size):
                    if inputs.is_stop(task_i):
                        continue
                    src_pos = bacth_mapping[task_i]
                    if src_pos != new_pos:
                        last_hidden_state[new_pos, ...] = last_hidden_state[src_pos,...]
                    new_pos += 1
                last_hidden_state = last_hidden_state[:new_pos,...]
                encoder_outputs = (last_hidden_state, )
            
            # clip cache
            if stop_event and past_key_values is not None:
                shrink_past_key_values = []
                for layer_i, layer in enumerate(past_key_values):
                    layer_cache = []
                    for cache_i, cache in enumerate(layer):
                        new_pos = 0
                        for task_i in range(inputs.batch_size):
                            if inputs.is_stop(task_i):
                                continue
                            src_pos = bacth_mapping[task_i]
                            if src_pos != new_pos:
                                cache[new_pos, ...] = cache[src_pos,...]
                            new_pos += 1
                        layer_cache.append(cache[:new_pos,...])
                    
                    shrink_past_key_values.append(tuple(layer_cache))
                past_key_values = tuple(shrink_past_key_values)
            # clip attention_mask
            if stop_event and attention_mask is not None:
                new_pos = 0
                for task_i in range(inputs.batch_size):
                    if inputs.is_stop(task_i):
                        continue
                    src_pos = bacth_mapping[task_i]
                    if src_pos != new_pos:
                        attention_mask[new_pos,...] = attention_mask[src_pos,...]
                    new_pos += 1
                attention_mask = attention_mask[:new_pos,...]
            # update bacth mapping
            if stop_event:
                new_pos = 0
                for task_i in range(inputs.batch_size):
                    if inputs.is_stop(task_i):
                        bacth_mapping[task_i] = -1
                    else:
                        bacth_mapping[task_i] = new_pos
                        new_pos += 1

            if all(inputs.stop):
                break
        
        # stop all
        for i in range(inputs.batch_size):
            if not inputs.is_stop(i):
                inputs.set_stop(i)
        if streamer:
            streamer.end()

        del past_key_values

class GenerationWorkerStreamer(BaseStreamer):
    def __init__(self,
                 task_batch: BatchingTask,
                 tokenizer: PreTrainedTokenizer,
                 worker: "GenerationModelWorker") -> None:
        self.task_batch = task_batch
        self.tokenizer = tokenizer
        self.worker = worker
        self.stream_interval = worker.stream_interval

        self.done = [False for i in range(task_batch.batch_size)]
    
    @cached(cache=LRUCache(maxsize=8192))
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def get_text_offset(self, text: str, tokens: List[str]):
        if self.tokenizer.is_fast:
            text_offset = [-1] * len(tokens)
            batch_encoding = self.tokenizer([text])
            for token_i in range(len(tokens)):
                span = batch_encoding.token_to_chars(0, token_i)
                if span is None:
                    continue
                start, end = span
                text_offset[token_i] = start
        else:
            text_offset = []
            for token_i in range(0, len(tokens)):
                if token_i == 0:
                    text_offset.append(-1)
                    continue
                prefix_text = self.convert_tokens_to_string(tuple(tokens[:token_i]))
                if text.startswith(prefix_text):
                    text_offset.append(len(prefix_text))
                else:
                    text_offset.append(-1)
            
            last_id = len(text)
            for token_i in reversed(range(0, len(tokens))):
                if text_offset[token_i] == -1:
                    text_offset[token_i] = last_id
                else:
                    last_id = text_offset[token_i]
        return text_offset
    
    def put(self, value):
        for i in range(self.task_batch.batch_size):
            generated_len = self.task_batch.get_generated_length(i)
            if (self.done[i] or generated_len % self.stream_interval != 0) and self.done[i]==self.task_batch.is_stop(i):
                continue
            task = self.task_batch.tasks[i]

            token_ids = self.task_batch.get_generated_ids(i)

            if len(token_ids) != 0:
                last_token = token_ids[-1]
                if last_token == self.tokenizer.eos_token_id:
                    token_ids = token_ids[:-1]
                tasks = self.task_batch.tasks
                if tasks[i].stop_token_ids is not None and last_token in tasks[i].stop_token_ids:
                    token_ids = token_ids[:-1]

            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            # text = self.convert_tokens_to_string(tuple(tokens))

            # get logprobs
            token_logprobs = self.task_batch.get_generated_token_probs(i)
            top_logprobs = self.task_batch.get_generated_top_logprobs(i)
            if top_logprobs is not None:
                top_logprobs_new = []
                for prob in top_logprobs:
                    top_logprobs_new.append({self.tokenizer.convert_ids_to_tokens(k): v for k, v in prob.items()})
                top_logprobs = top_logprobs_new

            # remove stop words
            stop_pos = stop_by_stopwords(text, 0, task.stop)
            if stop_pos != -1:
                self.task_batch.set_stop(i)

                # remove tokens after stop pos
                text = text[:stop_pos]
            
            prompt_len = self.task_batch.get_prompt_length(i)
            
            # logprobs
            if self.task_batch.tasks[i].logprobs is not None:
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
                # get offset mapping from token to text
                text_offset = self.get_text_offset(text, tokens)
                if stop_pos != -1:
                    token_stop_pos = len(tokens)
                    for token_i in reversed(range(0, len(text_offset))):
                        if text_offset[token_i] < stop_pos:
                            token_stop_pos = token_i + 1
                            break
                    # remove tokens after stop pos
                    tokens = tokens[:token_stop_pos]
                    if token_logprobs is not None:
                        token_logprobs = token_logprobs[:token_stop_pos]
                    if top_logprobs is not None:
                        top_logprobs = top_logprobs[:token_stop_pos]
                    text_offset = text_offset[:token_stop_pos]

                logprobs = GenerationWorkerLogprobs(
                    tokens=tokens,
                    token_logprobs=token_logprobs,
                    top_logprobs=top_logprobs,
                    text_offset=text_offset,
                )
            else:
                logprobs = None
            
            # push task to queue
            if self.task_batch.is_stop(i):
                if generated_len == self.task_batch.max_tokens[i]:
                    finish_reason = "length"
                else:
                    finish_reason = "stop"
                self.worker.push_task_result(task.task_id,
                    GenerationWorkerResult(
                        task_id=task.task_id,
                        type="finish",
                        text=text,
                        usage=UsageInfo(
                            prompt_tokens=prompt_len,
                            total_tokens=prompt_len + generated_len,
                            completion_tokens=generated_len,
                        ),
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                    )
                )
                self.worker.push_task_result(task.task_id,
                    BaseWorkerResult(task_id=task.task_id, type="done")
                )
                self.done[i] = True
            else:
                self.worker.push_task_result(task.task_id,
                    GenerationWorkerResult(
                        task_id=task.task_id,
                        type="data",
                        text=text,
                        usage=UsageInfo(
                            prompt_tokens=prompt_len,
                            total_tokens=prompt_len + generated_len,
                            completion_tokens=generated_len,
                        ),
                        logprobs=logprobs,
                        finish_reason=None,
                    )
                )

    def end(self):
        # check all done
        self.put(None)
            

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

class HuggingfaceGenerationExecutor(HuggingfaceExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        quantization: Optional[str],
        cpu_offloading: bool,
        deepspeed: bool = False,
        gptq: bool = False,
        group_size: Optional[int] = None,
        trust_remote_code: bool = False,
        offload_folder: Optional[str] = None,
        sleep_time: Optional[int] = 30,
    ) -> None:
        super(HuggingfaceGenerationExecutor, self).__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            quantization=quantization,
            cpu_offloading=cpu_offloading
        )
        self.last_call_time = time.time()

        self.deepspeed = deepspeed
        self.gptq = gptq
        self.group_size = group_size
        self.trust_remote_code = trust_remote_code
        self.offload_folder = offload_folder
        self.sleep_time = sleep_time

        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.adapter, self.model, self.tokenizer = self.load_model(
            model_path, device, num_gpus, max_gpu_memory, quantization, cpu_offloading, deepspeed, gptq, group_size, trust_remote_code, offload_folder
        )
        self.model.eval()

        # self.model = torch.compile(self.model)

        if hasattr(self.model.config, "max_sequence_length"):
            self._context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self._context_len = self.model.config.max_position_embeddings
        else:
            self._context_len = 2048
        
        self.current_batch = 2
    
    def _record_call_time(self):
        self.last_call_time = time.time()
    
    def sleep(self):
        self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.sleeping = True

    def wakeup(self):
        if self.model is not None:
            return
        self.adapter, self.model, self.tokenizer = self.load_model(
            self.model_path,
            self.device,
            self.num_gpus,
            self.max_gpu_memory,
            self.quantization,
            self.cpu_offloading,
            self.deepspeed,
            self.gptq,
            self.group_size,
            self.trust_remote_code,
            self.offload_folder
        )
        self.model.eval()
        self.sleeping = False

    @property
    def context_length(self) -> int:
        return self._context_len
    
    def tokenize(self, text: str) -> List[int]:
        input_ids = self.tokenizer(text).input_ids
        return input_ids
    
    def inference(self, worker: "GenerationModelWorker"):
        call_interval = time.time() - self.last_call_time
        if not self.sleeping and self.sleep_time > 0 and call_interval > self.sleep_time:
            self.sleep()

        if not worker.online:
            return

        tasks = worker.fetch_tasks(task_num=self.current_batch)

        if len(tasks) == 0:
            return
        
        self._record_call_time()
        if self.sleeping:
            self.wakeup()
        
        # batch inference
        tasks = sorted(tasks, key=lambda x:len(x.prompt), reverse=True)
        inputs = BatchingTask(tasks, self.tokenizer, self.device, self.model.config.is_encoder_decoder)
        
        if torch.cuda.is_available() and "cuda" in self.device:
            if self.device == "cuda":
                device = "cuda:0"
            else:
                device = self.device
            
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(device)
                if free_mem < total_mem * 0.3:
                    new_batch = self.current_batch * 2
                elif free_mem < total_mem * 0.8:
                    new_batch = self.current_batch + 1
                elif free_mem > total_mem * 0.95:
                    new_batch = self.current_batch - 1
                if len(tasks) == self.current_batch:
                    self.current_batch = new_batch
            except RuntimeError as e:
                traceback.print_exc()
        else:
            self.current_batch = worker.max_batch

        streamer = GenerationWorkerStreamer(inputs, self.tokenizer, worker)
        model = GenerationModel(self.model)
        max_new_tokens = max(inputs.max_tokens)
        model.generate(inputs, max_new_tokens, streamer)
  