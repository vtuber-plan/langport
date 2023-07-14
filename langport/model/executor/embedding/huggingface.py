import traceback
from typing import List, Optional

import torch
from langport.model.executor.huggingface import HuggingfaceExecutor
from langport.protocol.worker_protocol import BaseWorkerResult, EmbeddingWorkerResult, UsageInfo
from langport.workers.embedding_worker import EmbeddingModelWorker


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
        trust_remote_code: bool = False,
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
        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.adapter, self.model, self.tokenizer = self.load_model(
            model_path, device, num_gpus, max_gpu_memory, quantization, cpu_offloading, deepspeed, trust_remote_code
        )

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
    
    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    @torch.inference_mode()
    def inference(self, worker: "EmbeddingModelWorker"):
        if not worker.online:
            return
        tasks = worker.fetch_tasks()
        batch_size = len(tasks)
        if batch_size == 0:
            return
        # print(batch_size)

        prompts = [task.input for task in tasks]
        try:
            tokenizer = self.tokenizer
            model = self.model

            # ValueError: Asking to pad but the tokenizer does not have a padding token.
            # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`
            # or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
            if tokenizer._pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            encoded_prompts = tokenizer(prompts, return_tensors="pt", padding="longest").to(self.device)
            input_ids = encoded_prompts.input_ids
            if model.config.is_encoder_decoder:
                decoder_input_ids = torch.full(
                    (batch_size, 1),
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
            embeddings = self._mean_pooling(data, encoded_prompts['attention_mask'])
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
