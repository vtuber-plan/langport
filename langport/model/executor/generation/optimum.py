
from typing import List, Optional
from langport.model.executor.generation.huggingface import BatchingTask, GenerationModel, GenerationWorkerStreamer
from langport.model.executor.optimum import OptimumExecutor
from langport.workers.generation_worker import GenerationModelWorker


class OptimumGenerationExecutor(OptimumExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        load_8bit: bool,
        cpu_offloading: bool,
        trust_remote_code: bool = False
    ) -> None:
        super(OptimumGenerationExecutor, self).__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading
        )
        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.adapter, self.model, self.tokenizer = self.load_model(
            model_path, {}
        )

        # self.model = torch.compile(self.model)

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
    
    def inference(self, worker: "GenerationModelWorker"):
        if not worker.online:
            return

        tasks = worker.fetch_tasks()

        # batch inference
        inputs = BatchingTask(tasks, self.tokenizer, self.device, self.model.config.is_encoder_decoder)
        if inputs.batch_size == 0:
            return
        streamer = GenerationWorkerStreamer(inputs, self.tokenizer, worker)
        model = GenerationModel(self.model)
        max_new_tokens = max(inputs.max_tokens)
        model.generate(inputs, max_new_tokens, streamer)
  