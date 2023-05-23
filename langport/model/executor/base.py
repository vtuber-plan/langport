
from typing import List, Optional


class BaseModelExecutor(object):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        load_8bit: bool = False,
        cpu_offloading: bool = False,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.load_8bit = load_8bit
        self.cpu_offloading = cpu_offloading

    @property
    def context_length(self) -> int:
        return 2048

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("executor tokenizer method is not implemented.")