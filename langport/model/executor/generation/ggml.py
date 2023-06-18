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
            top_k = 40 if task.top_k <= 1 else task.top_k
            repetition_penalty = 1.17647 if task.repetition_penalty == 0.0 else task.repetition_penalty

            for j, token in enumerate(model.generate(tokens, top_k=top_k, top_p=task.top_p,
                                                  temperature=task.temperature, repetition_penalty=repetition_penalty)):
                output_ids.append(token)
                if tokenizer.is_eos_token(token) or prompt_length + j == task.max_tokens - 1:
                    output = tokenizer.decode(output_ids)
                    if tokenizer.is_eos_token(token):
                        finish_reason = "stop"
                    else:
                        finish_reason = "length"
                    yield GenerationWorkerResult(
                        task_id=task.task_id,
                        type="finish",
                        text=output,
                        usage=UsageInfo(
                            prompt_tokens=prompt_length,
                            total_tokens=prompt_length + j,
                            completion_tokens=j,
                        ),
                        finish_reason=finish_reason,
                    )
                    break

                if j%stream_interval!=0:
                    continue
                output = tokenizer.decode(output_ids)

                # yield result
                yield GenerationWorkerResult(
                    task_id=task.task_id,
                    type="data",
                    text=output,
                    usage=UsageInfo(
                        prompt_tokens=prompt_length,
                        total_tokens=prompt_length + j,
                        completion_tokens=j,
                    ),
                    finish_reason=None,
                )


class GgmlGenerationExecutor(GgmlExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        context_length: int,
        gpu_layers: int,
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
            lib=lib,
            model_type=model_type,
        )
        self.n_ctx = context_length
        self.adapter, self.model, self.tokenizer = self.load_model(model_path, from_pretrained_kwargs={})

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

        # batch inference
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
    