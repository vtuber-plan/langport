from typing import List, Optional

from llama_cpp import Llama, LlamaTokenizer

from langport.model.executor.llamacpp import LlamaCppExecutor
from langport.model.model_adapter import get_model_adapter
from langport.model.executor.base import BaseModelExecutor, LocalModelExecutor
from langport.protocol.worker_protocol import BaseWorkerResult, GenerationTask, GenerationWorkerResult, UsageInfo
from langport.workers.generation_worker import GenerationModelWorker


def batch_generation(
    model: Llama,
    tokenizer: LlamaTokenizer,
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
            tokens = tokenizer.encode(" " + task.prompt + " ")
            prompt_length = len(tokens)
            output_ids = []

            for j, token in enumerate(model.generate(tokens, top_k=40, top_p=task.top_p, 
                                                  temp=task.temperature, repeat_penalty=1.17647)):
                output_ids.append(token)
                if token == model.token_eos() or len(tokens) + j == task.max_tokens - 1:
                    output = tokenizer.decode(output_ids)
                    if token == model.token_eos():
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


class LlamaCppGenerationExecutor(LlamaCppExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        seed: int,
        n_batch: int,
        last_n_tokens_size: int
    ) -> None:
        super(LlamaCppGenerationExecutor, self).__init__(
            model_name=model_name,
            model_path=model_path,
            device="cpu",
            num_gpus=1,
            max_gpu_memory=None,
        )
        self.n_ctx = n_ctx
        self.adapter = get_model_adapter(model_path)
        self.model, self.tokenizer = self.load_model(model_path, {"n_ctx":n_ctx,
                                                                          "n_gpu_layers":n_gpu_layers, 
                                                                          "seed":seed, 
                                                                          "n_batch":n_batch, 
                                                                          "last_n_tokens_size":last_n_tokens_size,})

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
        for chunk in batch_generation(
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
    