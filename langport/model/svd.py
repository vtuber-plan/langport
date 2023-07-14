import gc
import glob
import os

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from langport.model.compression import get_compressed_list

class SVDLinear(nn.Module):
    """Compressed SVD Linear Layer."""

    def __init__(self, weight=None, bias=None, dtype=torch.float32, device=None, k_ratio: float=0.8):
        super().__init__()
        self.u_k = None
        self.v_k = None
        if weight is None:
            self.weight = None
        elif isinstance(weight, Tensor):
            weight = weight.data.to(torch.float32).to(device)
            # print(weight.shape)
            k = int(k_ratio * weight.shape[1])
            u, s, v = torch.linalg.svd(weight, full_matrices=False)
            self.u_k = u[:, :k].clone().to(dtype)
            self.v_k = v[:k, :].clone().to(dtype)
            self.weight = None
        else:
            self.weight = weight
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            bias = self.bias.to(self.weight.dtype)
        else:
            bias = self.bias
        if self.u_k is not None and self.v_k is not None:
            return F.linear(input.to(self.u_k.dtype), torch.matmul(self.u_k, self.v_k), bias)
        else:
            return F.linear(input.to(self.weight.dtype), self.weight, bias)


def apply_compressed_weight(module, target_dtype, target_device, prefix=""):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            setattr(
                module,
                attr_str,
                SVDLinear(
                    target_attr.weight, target_attr.bias, target_dtype, target_device
                ),
            )
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        apply_compressed_weight(
            child, target_dtype, target_device, child_prefix
        )


def load_svd_model(model_path, device, torch_dtype, trust_remote_code: bool=True):
    # partially load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    apply_compressed_weight(model, torch_dtype, device)

    model.to(torch_dtype).to(device)

    return model, tokenizer