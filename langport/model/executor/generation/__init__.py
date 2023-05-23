

from typing import Optional
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