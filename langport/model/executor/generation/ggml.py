import threading
from typing import List, Optional
from langport.model.executor.ggml import GgmlExecutor, GgmlTokenizer
from ctransformers import LLM
from langport.protocol.worker_protocol import BaseWorkerResult, GenerationTask, GenerationWorkerResult, UsageInfo
from langport.workers.generation_worker import GenerationModelWorker


def stream_generation(
    model: LLM,
    tokenizer: GgmlTokenizer,
    stream_interval: int,
    tasks: List[GenerationTask],
):
    batch_size = len(tasks)
    if batch_size == 0:
        return

    # todo: add stop words support
    for i, task in enumerate(tasks):
        output = ""

        if task.echo:
            output = task.prompt
        else:
            tokens = tokenizer.encode(task.prompt)
            prompt_length = len(tokens)
            output_ids = []

            # Compatible with some models
            top_k = 10 if task.top_k <= 1 else task.top_k
            repetition_penalty = 1.01 if task.repetition_penalty == 0.0 else task.repetition_penalty
            model.config.max_new_tokens = task.max_tokens

            finish_reason = "stop"
            n_tokens = 0
            for token in model.generate(
                            tokens, top_k=top_k, top_p=task.top_p, batch_size=model.config.batch_size,
                            threads=model.config.threads, temperature=task.temperature, 
                            last_n_tokens=256, repetition_penalty=repetition_penalty, reset=True):
                n_tokens += 1
                output_ids.append(token)
                if n_tokens == task.max_tokens:
                    output = tokenizer.decode(output_ids)
                    finish_reason = "length"
                    yield GenerationWorkerResult(
                        task_id=task.task_id,
                        type="finish",
                        text=output,
                        usage=UsageInfo(
                            prompt_tokens=prompt_length,
                            total_tokens=prompt_length + n_tokens,
                            completion_tokens=n_tokens,
                        ),
                        finish_reason=finish_reason,
                    )
                    break

                if n_tokens % stream_interval != 0:
                    continue
                output = tokenizer.decode(output_ids)

                # yield result
                yield GenerationWorkerResult(
                    task_id=task.task_id,
                    type="data",
                    text=output,
                    usage=UsageInfo(
                        prompt_tokens=prompt_length,
                        total_tokens=prompt_length + n_tokens,
                        completion_tokens=n_tokens,
                    ),
                    finish_reason=None,
                )

            # token == eos is checked in model.generate
            if finish_reason == "stop":
                output = tokenizer.decode(output_ids)
                yield GenerationWorkerResult(
                    task_id=task.task_id,
                    type="finish",
                    text=output,
                    usage=UsageInfo(
                        prompt_tokens=prompt_length,
                        total_tokens=prompt_length + n_tokens,
                        completion_tokens=n_tokens,
                    ),
                    finish_reason="stop",
                )


class GgmlGenerationExecutor(GgmlExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        context_length: int,
        gpu_layers: int,
        chunk_size: int,
        threads: int,
        model_type: str = "llama",
        lib: Optional[str] = None,
    ) -> None:
        n_gpu = 1 if gpu_layers > 0 else 0
        super(GgmlGenerationExecutor, self).__init__(
            model_name=model_name,
            model_path=model_path,
            device="cpu",
            num_gpus=n_gpu,
            max_gpu_memory=None,
            gpu_layers=gpu_layers,
            chunk_size=chunk_size,
            threads=threads,
            lib=lib,
            model_type=model_type,
        )
        self.n_ctx = context_length
        self.adapter, self.model, self.tokenizer = self.load_model(model_path, from_pretrained_kwargs={})
        self.lock = threading.Lock()

    @property
    def context_length(self) -> int:
        return self.n_ctx

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def inference(self, worker: "GenerationModelWorker"):
        if not worker.online:
            return

        tasks = worker.fetch_tasks()
        batch_size = len(tasks)
        if batch_size == 0:
            return
        
        self.lock.acquire()

        # batch inference
        try:
            for chunk in stream_generation(
                self.model,
                self.tokenizer,
                worker.stream_interval,
                tasks,
            ):
                worker.push_task_result(chunk.task_id, chunk)

            for task in tasks:
                worker.push_task_result(
                    task.task_id, BaseWorkerResult(task_id=task.task_id, type="done")
                )
        finally:
            self.lock.release()
    