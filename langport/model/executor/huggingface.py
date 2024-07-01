from functools import partial
import os
from typing import Optional

from langport.model.adapters.dolly_v2 import DollyV2Adapter
from langport.model.adapters.openbuddy import OpenBuddyAdapter
from langport.model.adapters.qwen import QwenAdapter
from langport.model.adapters.rwkv import RwkvAdapter
from langport.model.adapters.t5 import T5Adapter
from langport.model.adapters.text2vec import BertAdapter
from langport.model.adapters.chatglm import ChatGLMAdapter

from langport.model.executor.base import LocalModelExecutor

import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    BertTokenizer,
    BertModel,
)

from transformers.utils.quantization_config import QuantizationMethod
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from accelerate import init_empty_weights

import math
from typing import Optional
import warnings
import psutil

import torch
from langport.model.compression import load_compress_model, default_compression_config, bit4_compression_config
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
        elif isinstance(adapter, QwenAdapter):
            trust_remote_code = from_pretrained_kwargs.get("trust_remote_code", True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
            # allowed_special work around
            tokenizer.tokenize = partial(tokenizer.tokenize, allowed_special="all")
        elif isinstance(adapter, OpenBuddyAdapter):
            trust_remote_code = from_pretrained_kwargs.get("trust_remote_code", False)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            ) # , offload_folder="offload"
        else:
            # GPTQ quanted mode work around
            trust_remote_code = from_pretrained_kwargs.get("trust_remote_code", False)
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            if hasattr(config, "quantization_config"):
                quantization_method_from_config = config.quantization_config.get(
                    "quant_method", QuantizationMethod.BITS_AND_BYTES
                )
                if quantization_method_from_config == QuantizationMethod.GPTQ:
                    no_split_module_classes = ["LlamaDecoderLayer", "GPTJBlock", "GPT2Block", "GPTBigCodeBlock", "GPTNeoBlock"]
                    device_map = from_pretrained_kwargs.get("device_map", "auto")
                    if config.quantization_config.get("bits", None) == 4:
                        torch_dtype = torch.quint4x2
                    elif config.quantization_config.get("bits", None) == 8:
                        torch_dtype = torch.int8
                    else:
                        torch_dtype = torch.int8
                    with init_empty_weights():
                        model = AutoModelForCausalLM.from_config(config)
                    device_map = infer_auto_device_map(
                        model, max_memory=None, no_split_module_classes=no_split_module_classes, dtype=torch_dtype
                    )
                    from_pretrained_kwargs["device_map"] = device_map
                    
            trust_remote_code = from_pretrained_kwargs.get("trust_remote_code", False)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True,**from_pretrained_kwargs
            )

        return model, tokenizer

    def load_sentence_transformer_model(
        self,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str] = None,
        quantization: Optional[str] = None,
        cpu_offloading: bool = False,
        deepspeed: bool = False,
        gptq: bool = False,
        group_size: Optional[int] = None,
        trust_remote_code: bool = False,
        offload_folder: Optional[str] = None,
        debug: bool = False,
    ):
        """Load a model from Hugging Face."""
        from sentence_transformers import SentenceTransformer
        adapter = get_model_adapter(model_path)

        kwargs = {}
        if device == "cpu":
            kwargs["torch_dtype"] = torch.float32
        elif device == "cuda":
            kwargs["torch_dtype"] = "auto"
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    if len(available_gpu_memory) == 0:
                        kwargs["device_map"] = "auto"
                    elif all([mem == available_gpu_memory[0] for mem in available_gpu_memory]):
                        kwargs["device_map"] = "balanced"
                    else:
                        kwargs["max_memory"] = {
                            i: str(int(available_gpu_memory[i] * 0.55)) + "GiB"
                            for i in range(num_gpus)
                        }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        elif device == "mps":
            kwargs["torch_dtype"] = torch.float16
            # Avoid bugs in mps backend by not using in-place operations.
            replace_llama_attn_with_non_inplace_operations()
        else:
            raise ValueError(f"Invalid device: {device}")

        kwargs["trust_remote_code"] = trust_remote_code
        if offload_folder is not None:
            kwargs["offload_folder"] = offload_folder

        model = SentenceTransformer(model_path, device=device, trust_remote_code=trust_remote_code, model_kwargs=kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return adapter, model, tokenizer

    def load_model(
        self,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str] = None,
        quantization: Optional[str] = None,
        cpu_offloading: bool = False,
        deepspeed: bool = False,
        gptq: bool = False,
        group_size: Optional[int] = None,
        trust_remote_code: bool = False,
        offload_folder: Optional[str] = None,
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
                    if len(available_gpu_memory) == 0:
                        kwargs[
                            "device_map"
                        ] = "auto"
                    elif all([mem == available_gpu_memory[0] for mem in available_gpu_memory]):
                        kwargs[
                            "device_map"
                        ] = "balanced"
                    else:
                        kwargs["max_memory"] = {
                            i: str(int(available_gpu_memory[i] * 0.55)) + "GiB"
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
        if offload_folder is not None:
            kwargs["offload_folder"] = offload_folder

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
        elif quantization is not None or gptq:
            if group_size is None:
                group_size = 128
            if gptq:
                if quantization is None:
                    quantization = "8bit"
                import datasets
                from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
                if "gptq" in model_path.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **kwargs)
                    model = AutoGPTQForCausalLM.from_quantized(model_path, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **kwargs)
                    if tokenizer._pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                    if "4" in quantization:
                        quantize_config = BaseQuantizeConfig(
                            bits=4,  # quantize model to 4-bit
                            group_size=group_size,  # it is recommended to set the value to 128
                            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
                        )
                    elif "8" in quantization:
                        quantize_config = BaseQuantizeConfig(
                            bits=8,  # quantize model to 4-bit
                            group_size=group_size,  # it is recommended to set the value to 128
                            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
                        )
                    else:
                        quantize_config = BaseQuantizeConfig(
                            bits=8,  # quantize model to 4-bit
                            group_size=group_size,  # it is recommended to set the value to 128
                            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
                        )
                    # load un-quantized model, by default, the model will always be loaded into CPU memory
                    # temp_kwargs = {k: v for k,v in kwargs.items() if k not in ["max_memory", "device_map"]}
                    temp_kwargs = {k: v for k,v in kwargs.items()}
                    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config, low_cpu_mem_usage=True, **temp_kwargs)
                    quant_dataset = datasets.load_dataset("Vtuber-plan/quantdata-10k")
                    train_quant_dataset = quant_dataset["train"]
                    examples = []
                    for data in train_quant_dataset:
                        examples.append(tokenizer(data["text"], max_length=512, padding="longest", truncation=True, return_tensors='pt'))
                    model.quantize(examples, batch_size=1, cache_examples_on_gpu=False)
                    temp_model_path = os.path.join(offload_folder, os.path.basename(model_path))
                    model.save_quantized(temp_model_path, use_safetensors=True)
                    model = AutoGPTQForCausalLM.from_quantized(temp_model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                if num_gpus != 1:
                    warnings.warn(
                        "n-bit quantization is not supported for multi-gpu inference."
                    )
                    if "8" in quantization:
                        model, tokenizer = load_compress_model(
                            model_path=model_path, device="auto", compression_config=default_compression_config, **kwargs
                        )
                    elif "4" in quantization:
                        model, tokenizer = load_compress_model(
                            model_path=model_path, device="auto", compression_config=bit4_compression_config, **kwargs
                        )
                    else:
                        model, tokenizer = load_compress_model(
                            model_path=model_path, device="auto", compression_config=default_compression_config, **kwargs
                        )
                else:
                    if "8" in quantization:
                        # model, tokenizer = load_compress_model(
                        #     model_path=model_path, device=device, compression_config=default_compression_config, **kwargs
                        # )
                        kwargs["load_in_8bit"] = True
                        model, tokenizer = self._load_hf_model(adapter, model_path, kwargs)
                    elif "4" in quantization:
                        # model, tokenizer = load_compress_model(
                        #     model_path=model_path, device=device, compression_config=bit4_compression_config, **kwargs
                        # )
                        kwargs["load_in_4bit"] = True
                        model, tokenizer = self._load_hf_model(adapter, model_path, kwargs)
                    else:
                        model, tokenizer = load_compress_model(
                            model_path=model_path, device=device, compression_config=default_compression_config, **kwargs
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
            if (device == "cuda" and num_gpus == 1 and not cpu_offloading and quantization is None) or device == "mps":
                model.to(device)

        if debug:
            print(model)

        return adapter, model, tokenizer
