from typing import List, Optional
from ctransformers import AutoModelForCausalLM, LLM, AutoConfig, Config
from langport.model.executor.base import LocalModelExecutor
from langport.model.model_adapter import get_model_adapter

class GgmlTokenizer:
    def __init__(self, model:LLM) -> None:
        self.model = model
    
    def encode(self, text: str) -> List[int]:
        return self.model.tokenize(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.model.detokenize(tokens)
    
    def is_eos_token(self, token: int) -> bool:
        return self.model.is_eos_token(token)
        

class GgmlExecutor(LocalModelExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        lib: Optional[str] = None,
        gpu_layers: int = 0,
        model_type: str = 'llama',
        chunk_size: int = 1024,
        threads: int = -1,
        quantization: Optional[str] = None,
        cpu_offloading: bool = False,
    ) -> None:
        super(GgmlExecutor, self).__init__(
            model_name = model_name,
            model_path = model_path,
            device = device,
            num_gpus = num_gpus,
            max_gpu_memory = max_gpu_memory,
            quantization = quantization,
            cpu_offloading = cpu_offloading,
        )
        self.gpu_layers = gpu_layers
        # ctransformers has a bug
        self.lib = lib
        self.model_type = model_type
        self.chunk_size = chunk_size
        self.threads = threads
 

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        adapter = get_model_adapter(model_path)
        config = Config()
        setattr(config, 'stream', True)
        setattr(config, 'gpu_layers', self.gpu_layers)
        setattr(config, 'batch_size', self.chunk_size)
        setattr(config, 'threads', self.threads)
        auto_config = AutoConfig(config=config, model_type=self.model_type)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                   config=auto_config,
                                                   lib=self.lib,
                                                   )
        tokenizer = GgmlTokenizer(model)

        return adapter, model, tokenizer
