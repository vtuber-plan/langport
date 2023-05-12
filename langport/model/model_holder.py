from langport.model.model_adapter import load_model


class LanguageModelHolder(object):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: str,
        num_gpus: int,
        max_gpu_memory,
        load_8bit: bool,
        cpu_offloading: bool,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.load_8bit = load_8bit
        self.cpu_offloading = cpu_offloading

        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.adapter, self.model, self.tokenizer = load_model(
            model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading
        )
    
    @property
    def context_len(self) -> int:
        if hasattr(self.model.config, "max_sequence_length"):
            return self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        else:
            return 2048