

from typing import Optional
from llama_cpp import Llama
from langport.model.executor.base import LocalModelExecutor


class LlamaCppExecutor(LocalModelExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        load_8bit: bool = False,
        cpu_offloading: bool = False,
    ) -> None:
        super(LlamaCppExecutor, self).__init__(
            model_name = model_name,
            model_path = model_path,
            device = device,
            num_gpus = num_gpus,
            max_gpu_memory = max_gpu_memory,
            load_8bit = load_8bit,
            cpu_offloading = cpu_offloading,
        )
 
    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model = Llama(model_path, **from_pretrained_kwargs)
        return model, model.tokenizer()
