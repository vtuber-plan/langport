import os
from typing import List, Optional
from langport.model.executor.base import LocalModelExecutor
from langport.model.model_adapter import get_model_adapter

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
        

class OptimumExecutor(LocalModelExecutor):
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
        super(OptimumExecutor, self).__init__(
            model_name = model_name,
            model_path = model_path,
            device = device,
            num_gpus = num_gpus,
            max_gpu_memory = max_gpu_memory,
            quantization = quantization,
            cpu_offloading = cpu_offloading,
        )

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        adapter = get_model_adapter(model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if os.path.exists(os.path.join(model_path, "decoder_model.onnx")):
            export_onnx = False
        else:
            export_onnx = True
        
        use_gpu = False # self.device == "cuda"
        if use_gpu:
            provider = "CUDAExecutionProvider"
        else:
            provider = "CPUExecutionProvider"
        model = ORTModelForCausalLM.from_pretrained(model_path, export=export_onnx, provider=provider)

        return adapter, model, tokenizer
