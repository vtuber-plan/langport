

from typing import Optional
from langport.model.executor.base import BaseModelExecutor
# from langport.workers.embedding_worker import EmbeddingModelWorker


class EmbeddingExecutor(BaseModelExecutor):
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
        super(EmbeddingExecutor, self).__init__(
            model_path=model_path,
            model_name=model_name,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading
        )
        self.adapter = None
        self.model = None
        self.tokenizer = None
    
    def inference(self, worker: "EmbeddingModelWorker"):
        raise NotImplementedError("EmbeddingExecutor inference not implemented")
        
    