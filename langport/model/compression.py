import dataclasses
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


@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""

    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


default_compression_config = CompressionConfig(
    num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True
)

bit4_compression_config = CompressionConfig(
    num_bits=4, group_size=256, group_dim=1, symmetric=False, enabled=True
)

bit2_compression_config = CompressionConfig(
    num_bits=2, group_size=64, group_dim=1, symmetric=False, enabled=True
)

class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, compression_config, weight=None, bias=None, device=None):
        super().__init__()
        self.config = compression_config
        if weight is None:
            self.weight = None
        elif isinstance(weight, Tensor):
            self.weight = compress(weight.data.to(device), self.config)
        else:
            self.weight = weight
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        weight = decompress(self.weight, self.config)
        if self.bias is not None:
            bias = self.bias.to(weight.dtype)
        else:
            bias = self.bias
        return F.linear(input.to(weight.dtype), weight, bias)


def compress_module(module, target_device, config):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                CLinear(config, target_attr.weight, target_attr.bias, target_device),
            )
    for name, child in module.named_children():
        compress_module(child, target_device, config)


def get_compressed_list(module, prefix=""):
    compressed_list = []
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            compressed_list.append(full_name)
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        for each in get_compressed_list(child, child_prefix):
            compressed_list.append(each)
    return compressed_list


def apply_compressed_weight(module, compressed_state_dict, target_device, config, prefix=""):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            setattr(
                module,
                attr_str,
                CLinear(
                    config, compressed_state_dict[full_name], target_attr.bias, target_device
                ),
            )
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        apply_compressed_weight(
            child, compressed_state_dict, target_device, config, child_prefix
        )


def load_compress_model(model_path, device, torch_dtype, compression_config: CompressionConfig=default_compression_config, trust_remote_code: bool=True):
    # partially load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    base_pattern = os.path.join(model_path, "pytorch_model-*.bin")
    files = glob.glob(base_pattern)

    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        linear_weights = get_compressed_list(model)

    compressed_state_dict = {}

    for filename in tqdm(files):
        tmp_state_dict = torch.load(filename)
        for name in tmp_state_dict:
            if name in linear_weights:
                tensor = tmp_state_dict[name].to(device).data.to(torch_dtype)
                compressed_state_dict[name] = compress(
                    tensor, compression_config
                )
            else:
                compressed_state_dict[name] = tmp_state_dict[name].to(device)
            tmp_state_dict[name] = None
            tensor = None
            gc.collect()
            torch.cuda.empty_cache()

    for name in model.state_dict():
        if name not in linear_weights:
            set_module_tensor_to_device(
                model, name, device, value=compressed_state_dict[name]
            )
    apply_compressed_weight(model, compressed_state_dict, device, compression_config)

    model.to(device)

    return model, tokenizer

def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (
        original_shape[:group_dim]
        + (num_groups, group_size)
        + original_shape[group_dim + 1 :]
    )

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = (
            original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1 :]
        )
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim,
        )
    data = tensor.view(new_shape)

    if num_bits >= 8:
        final_shape = new_shape
    elif num_bits > 2:
        final_shape = [i for i in new_shape]
        final_shape[group_dim + 1] = final_shape[group_dim + 1] // 2
        final_data = torch.zeros(size=final_shape, dtype=torch.uint8, device=data.device)
    else:
        final_shape = [i for i in new_shape]
        final_shape[group_dim + 1] = final_shape[group_dim + 1] // 4
        final_data = torch.zeros(size=final_shape, dtype=torch.uint8, device=data.device)

    # Quantize
    if num_bits >= 8:
        if symmetric:
            B = 2 ** (num_bits - 1) - 1
            scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
            data = data * scale
            data = data.clamp_(-B, B).round_().to(torch.int8)
            return data, scale, original_shape
        else:
            B = 2**num_bits - 1
            mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
            mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

            scale = B / (mx - mn)
            data = data - mn
            data.mul_(scale)

            data = data.clamp_(0, B).round_().to(torch.uint8)
            return data, mn, scale, original_shape
    elif num_bits > 2:
        B = 2**num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        left_half = [slice(None) for i in new_shape]
        left_half[group_dim + 1] = slice(0, group_size//2)
        final_data = torch.bitwise_left_shift(data[left_half], 4)

        right_half = [slice(None) for i in new_shape]
        right_half[group_dim + 1] = slice(group_size//2, group_size)
        final_data = final_data.bitwise_or_(data[right_half])
        
        return final_data, mn, scale, original_shape
    else:
        B = 2**num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        first_half = [slice(None) for i in new_shape]
        first_half[group_dim + 1] = slice(0, group_size//4)
        final_data = torch.bitwise_left_shift(data[first_half], 6)

        second_half = [slice(None) for i in new_shape]
        second_half[group_dim + 1] = slice(group_size//4, group_size//4*2)
        final_data = final_data.bitwise_or_(torch.bitwise_left_shift(data[second_half], 4))
        
        third_half = [slice(None) for i in new_shape]
        third_half[group_dim + 1] = slice(group_size//4*2, group_size//4*3)
        final_data = final_data.bitwise_or_(torch.bitwise_left_shift(data[third_half], 2))

        fourth_half = [slice(None) for i in new_shape]
        fourth_half[group_dim + 1] = slice(group_size//4*3, group_size)
        final_data = final_data.bitwise_or_(data[fourth_half])

        return final_data, mn, scale, original_shape

def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )

    if num_bits >= 8:
        # Dequantize
        if symmetric:
            data, scale, original_shape = packed_data
            data = data / scale
        else:
            data, mn, scale, original_shape = packed_data
            data = data / scale
            data.add_(mn)
    elif num_bits > 2:
        data, mn, scale, original_shape = packed_data
        new_shape = [i for i in data.shape]
        new_shape[group_dim + 1] = 2 * new_shape[group_dim + 1]
        int8_data = torch.zeros(size=new_shape, dtype=torch.uint8, device=data.device)

        left_half = [slice(None) for i in new_shape]
        left_half[group_dim + 1] = slice(0, group_size//2)
        left_mask = torch.tensor([0b11110000], dtype=torch.uint8, device=data.device)
        int8_data[left_half] = torch.bitwise_right_shift(torch.bitwise_and(data, left_mask), 4)

        right_half = [slice(None) for i in new_shape]
        right_half[group_dim + 1] = slice(group_size//2, group_size)
        right_mask = torch.tensor([0b00001111], dtype=torch.uint8, device=data.device)
        int8_data[right_half] = torch.bitwise_and(data, right_mask)

        data = int8_data / scale
        data.add_(mn)
    else:
        data, mn, scale, original_shape = packed_data
        new_shape = [i for i in data.shape]
        new_shape[group_dim + 1] = 4 * new_shape[group_dim + 1]
        int8_data = torch.zeros(size=new_shape, dtype=torch.uint8, device=data.device)

        first_half = [slice(None) for i in new_shape]
        first_half[group_dim + 1] = slice(0, group_size//4)
        first_mask = torch.tensor([0b11000000], dtype=torch.uint8, device=data.device)
        int8_data[first_half] = torch.bitwise_right_shift(torch.bitwise_and(data, first_mask), 6)

        second_half = [slice(None) for i in new_shape]
        second_half[group_dim + 1] = slice(group_size//4, group_size//4*2)
        second_mask = torch.tensor([0b00110000], dtype=torch.uint8, device=data.device)
        int8_data[second_half] = torch.bitwise_right_shift(torch.bitwise_and(data, second_mask), 4)
        
        third_half = [slice(None) for i in new_shape]
        third_half[group_dim + 1] = slice(group_size//4*2, group_size//4*3)
        third_mask = torch.tensor([0b00001100], dtype=torch.uint8, device=data.device)
        int8_data[third_half] = torch.bitwise_right_shift(torch.bitwise_and(data, third_mask), 2)

        fourth_half = [slice(None) for i in new_shape]
        fourth_half[group_dim + 1] = slice(group_size//4*3, group_size)
        fourth_mask = torch.tensor([0b00000011], dtype=torch.uint8, device=data.device)
        int8_data[fourth_half] = torch.bitwise_and(data, fourth_mask)

        data = int8_data / scale
        data.add_(mn)
    
    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim]
            + (original_shape[group_dim] + pad_len,)
            + original_shape[group_dim + 1 :]
        )
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)
