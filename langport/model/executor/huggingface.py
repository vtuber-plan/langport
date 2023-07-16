from typing import Optional

from langport.model.adapters.dolly_v2 import DollyV2Adapter
from langport.model.adapters.rwkv import RwkvAdapter
from langport.model.adapters.t5 import T5Adapter
from langport.model.adapters.text2vec import BertAdapter
from langport.model.adapters.chatglm import ChatGLMAdapter

from langport.model.executor.base import LocalModelExecutor

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    BertTokenizer,
    BertModel,
)


import math
from typing import Optional
import warnings
import psutil

import torch
from langport.model.compression import load_compress_model, default_compression_config, bit4_compression_config
from langport.model.svd import load_svd_model
from langport.model.executor.base import BaseModelExecutor
from langport.model.model_adapter import get_model_adapter, raise_warning_for_incompatible_cpu_offloading_configuration
from langport.model.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
from langport.utils import get_gpu_memory



class HuggingfaceExecutor(LocalModelExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        quantization: Optional[str],
        cpu_offloading: bool,
        deepspeed: bool = False,
    ) -> None:
        super(HuggingfaceExecutor, self).__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            quantization=quantization,
            cpu_offloading=cpu_offloading
        )
    
      
    def _load_hf_model(self, adapter, model_path: str, from_pretrained_kwargs: dict):
        if isinstance(adapter, DollyV2Adapter):
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
            # 50277 means "### End"
            tokenizer.eos_token_id = 50277
        elif isinstance(adapter, RwkvAdapter):
            from langport.model.models.rwkv_model import RwkvModel
            model = RwkvModel(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-160m", use_fast=True
            )
        elif isinstance(adapter, T5Adapter):
            tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        elif isinstance(adapter, BertAdapter):
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertModel.from_pretrained(model_path, **from_pretrained_kwargs)
        elif isinstance(adapter, ChatGLMAdapter):
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            if "trust_remote_code" in from_pretrained_kwargs:
                from_pretrained_kwargs.pop("trust_remote_code")
            model = AutoModel.from_pretrained(
                model_path, low_cpu_mem_usage=True, trust_remote_code=True, **from_pretrained_kwargs
            )
        else:
            trust_remote_code = from_pretrained_kwargs.get("trust_remote_code", False)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )

        return model, tokenizer

    def load_model(
        self,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str] = None,
        quantization: Optional[str] = None,
        cpu_offloading: bool = False,
        deepspeed: bool = False,
        trust_remote_code: bool = False,
        debug: bool = False,
    ):
        """Load a model from Hugging Face."""
        adapter = get_model_adapter(model_path)

        # Handle device mapping
        cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
            device, quantization!=None, cpu_offloading
        )
        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        elif device == "mps":
            kwargs = {"torch_dtype": torch.float16}
            # Avoid bugs in mps backend by not using in-place operations.
            replace_llama_attn_with_non_inplace_operations()
        else:
            raise ValueError(f"Invalid device: {device}")

        kwargs["trust_remote_code"] = trust_remote_code

        if cpu_offloading:
            # raises an error on incompatible platforms
            from transformers import BitsAndBytesConfig

            if "max_memory" in kwargs:
                kwargs["max_memory"]["cpu"] = (
                    str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
                )
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit_fp32_cpu_offload=cpu_offloading
            )
            kwargs["load_in_8bit"] = quantization!=None
            # Load model
            model, tokenizer = self._load_hf_model(adapter, model_path, kwargs)
        elif quantization is not None:
            if num_gpus != 1:
                warnings.warn(
                    "n-bit quantization is not supported for multi-gpu inference."
                )
            else:
                if "8" in quantization:
                    model, tokenizer = load_compress_model(
                        model_path=model_path, device=device, torch_dtype=kwargs["torch_dtype"], compression_config=default_compression_config, trust_remote_code=trust_remote_code
                    )
                elif "4" in quantization:
                    model, tokenizer = load_compress_model(
                        model_path=model_path, device=device, torch_dtype=kwargs["torch_dtype"], compression_config=bit4_compression_config, trust_remote_code=trust_remote_code
                    )
                else:
                    model, tokenizer = load_compress_model(
                        model_path=model_path, device=device, torch_dtype=kwargs["torch_dtype"], compression_config=default_compression_config, trust_remote_code=trust_remote_code
                    )
                # return adapter, model, tokenizer
        else:
            # Load model
            model, tokenizer = self._load_hf_model(adapter, model_path, kwargs)

        if deepspeed:
            from transformers.deepspeed import HfDeepSpeedConfig
            import deepspeed

            dtype = torch.float16
            # FIXME: deepspeed quantization
            if quantization is not None:
                dtype = torch.int8
            config = {
                # "mp_size": 1,        # Number of GPU
                "dtype": dtype, # dtype of the weights (fp16)
                "replace_method": "auto", # Lets DS autmatically identify the layer to replace
                "replace_with_kernel_inject": True, # replace the model with the kernel injector
                "enable_cuda_graph": True,
                "tensor_parallel": {
                    "enabled": True,
                    "tp_size": 1,
                },
            }
            ds_engine = deepspeed.init_inference(model=model, config=config)
            model = ds_engine.module
        else:
            if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device == "mps":
                model.to(device)

        if debug:
            print(model)

        return adapter, model, tokenizer
