import os
import time
import traceback
from typing import List, Optional

import torch
from huggingface_hub import hf_hub_download
from langport.model.executor.huggingface import HuggingfaceExecutor
from langport.protocol.worker_protocol import BaseWorkerResult, EmbeddingWorkerResult, EmbeddingsObject, UsageInfo
from langport.workers.embedding_worker import EmbeddingModelWorker
from langport.utils.itertools import batched

class HuggingfaceEmbeddingExecutor(HuggingfaceExecutor):
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
        super(HuggingfaceEmbeddingExecutor, self).__init__(
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

        if os.path.exists(model_path):
            if not os.path.exists(os.path.join(model_path, "modules.json")):
                modules_file = ""
            else:
                with open(os.path.join(model_path, "modules.json"), "r", encoding="utf-8") as f:
                    modules_file = f.read()
        else:
            modules_file = hf_hub_download(repo_id=model_path, filename="modules.json")
        if "sentence_transformers" in modules_file:
            self.adapter, self.model, self.tokenizer = self.load_sentence_transformer_model(
                model_path, device, num_gpus, max_gpu_memory, quantization, cpu_offloading,
                deepspeed, gptq, group_size, trust_remote_code, offload_folder
            )
            if hasattr(self.model, "max_seq_length"):
                self._context_len = self.model.max_seq_length
            else:
                self._context_len = 2048
        else:
            self.adapter, self.model, self.tokenizer = self.load_model(
                model_path, device, num_gpus, max_gpu_memory, quantization, cpu_offloading,
                deepspeed, gptq, group_size, trust_remote_code, offload_folder
            )
            if hasattr(self.model.config, "max_sequence_length"):
                self._context_len = self.model.config.max_sequence_length
            elif hasattr(self.model.config, "max_position_embeddings"):
                self._context_len = self.model.config.max_position_embeddings
            else:
                self._context_len = 2048
    
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

    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def inference_batch(self, prompts: List[str]) -> List[str]:
        tokenizer = self.tokenizer
        model = self.model
        if model.__class__.__module__ + '.' + model.__class__.__name__ != 'sentence_transformers.SentenceTransformer.SentenceTransformer':
            encoded_prompts = tokenizer(prompts, return_tensors="pt", padding="longest").to(self.device)
            input_ids = encoded_prompts.input_ids
            if model.config.is_encoder_decoder:
                decoder_input_ids = torch.full(
                    (len(prompts), 1),
                    model.generation_config.decoder_start_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                model_output = model(input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                data = model_output.decoder_hidden_states[-1]
            elif model.config.is_decoder:
                model_output = model(input_ids, output_hidden_states=True)
                is_chatglm = "chatglm" in str(type(model)).lower()
                if is_chatglm:
                    data = model_output.hidden_states[-1].transpose(0, 1)
                else:
                    data = model_output.hidden_states[-1]
            else:
                data = model(**encoded_prompts)
            # embeddings = torch.mean(data, dim=1)
            embeddings = self._mean_pooling(data, encoded_prompts['attention_mask']).cpu()
        else:
            embeddings = model.encode(prompts, show_progress_bar=False)
        return embeddings


    @torch.inference_mode()
    def inference(self, worker: "EmbeddingModelWorker"):
        call_interval = time.time() - self.last_call_time
        if not self.sleeping and self.sleep_time > 0 and call_interval > self.sleep_time:
            self.sleep()

        if not worker.online:
            return
        tasks = worker.fetch_tasks()
        batch_size = len(tasks)
        if batch_size == 0:
            return
        
        self._record_call_time()
        if self.sleeping:
            self.wakeup()

        # print(batch_size)
        prompts = []
        prompts_index = []
        for task_i, task in enumerate(tasks):
            task_input = task.input
            if isinstance(task_input, str):
                prompts.append(task_input)
                prompts_index.append(task_i)
            elif isinstance(task_input, list):
                prompts.extend(task_input)
                prompts_index.extend([task_i] * len(task_input))
            else:
                raise Exception("Invalid prompt type...")

        try:
            batch_prompts = batched(prompts, worker.max_batch)
            embeddings = []
            for each_batch in batch_prompts:
                batch_embeddings = self.inference_batch(each_batch)
                embeddings.extend(batch_embeddings)
            # ValueError: Asking to pad but the tokenizer does not have a padding token.
            # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`
            # or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
            if self.tokenizer._pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            for task_i, cur_task in enumerate(tasks):
                token_num = 0
                embedding_list = []
                for prompt_i in range(len(prompts)):
                    if prompts_index[prompt_i] == task_i:
                        token_num += len(self.tokenizer(prompts[prompt_i]).input_ids)
                        embedding_list.append(EmbeddingsObject(index=task_i, embedding=embeddings[prompt_i].tolist()))
                worker.push_task_result(
                    cur_task.task_id,
                    EmbeddingWorkerResult(
                        task_id=cur_task.task_id,
                        type="data",
                        embeddings=embedding_list,
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
            traceback.print_exc()
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
