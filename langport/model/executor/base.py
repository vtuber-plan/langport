from typing import List, Optional

class BaseModelExecutor(object):
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    @property
    def context_length(self) -> int:
        return 2048

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("executor tokenizer method is not implemented.")

class LocalModelExecutor(BaseModelExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        quantization: Optional[str] = None,
        cpu_offloading: bool = False,
    ) -> None:
        super(LocalModelExecutor, self).__init__(
            model_name=model_name,
        )

        self.model_path = model_path
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.quantization = quantization
        self.cpu_offloading = cpu_offloading

        self.sleeping = False
    
    def sleep(self):
        pass

    def wakeup(self):
        pass
        
    @property
    def context_length(self) -> int:
        return 2048

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("executor tokenizer method is not implemented.")

class RemoteModelExecutor(BaseModelExecutor):
    def __init__(
        self,
        model_name: str,
        api_url: str,
        api_key: str,
    ) -> None:
        super(RemoteModelExecutor, self).__init__(
            model_name=model_name,
        )
        self.api_url = api_url
        self.api_key = api_key

    @property
    def context_length(self) -> int:
        return 2048

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("executor tokenizer method is not implemented.")
