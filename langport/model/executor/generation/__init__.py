from queue import Queue
from typing import List, Optional
from langport.model.executor.base import BaseModelExecutor
# from langport.workers.generation_worker import GenerationModelWorker

class GenerationExecutor(BaseModelExecutor):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        load_8bit: bool,
        cpu_offloading: bool,
    ) -> None:
        super(GenerationExecutor, self).__init__(
            model_path=model_path,
            model_name=model_name,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading
        )

    def inference(self, worker: "GenerationModelWorker"):
        raise NotImplementedError("GenerationExecutor inference not implemented")

class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    Same as huggingface transformers
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens
        value: Tensor | int
        """
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()


class BatchStreamer(BaseStreamer):
    def __init__(self, tokenizer, stream_interval: int, skip_prompt: bool = False, **decode_kwargs) -> None:
        """tokenizer must have `decode [int]->str` method
        """
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.is_stop:List[bool] = []
        self.step = 0
        self.output_batch_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True

    def put(self, value):
        self.step += 1
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        value_list = value.tolist()
        self.token_cache.append(value_list)
        output_batch = []
        for i in range(len(value)):
            if len(self.is_stop)==0:
                self.is_stop.append(value_list[i] == self.tokenizer.eos_token)
            else:
                self.is_stop[i] = value_list[i] == self.tokenizer.eos_token
            if self.step % self.stream_interval == 0:
                tokens = [x[i] for x in self.token_cache]
                text = self.tokenizer.decode(tokens, **self.decode_kwargs)
                output_batch.append((text,self.is_stop[i]))
        if len(output_batch) != 0:
            self.output_batch_queue.put(output_batch)
            print(output_batch)

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        self.next_tokens_are_prompt = True
        self.token_cache.clear()
        self.is_stop.clear()
        self.step = 0
        self.output_batch_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        """
        [(str,bool),(str,bool),...]
        """
        value = self.output_batch_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value