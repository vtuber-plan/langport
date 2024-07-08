import dataclasses
import gc
import glob
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple, Union
import transformers
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

try:
    import cupy
    cupy_available = True
except ImportError:
    cupy_available = False
    print("Info: Install cupy to get better quantization performance.")

@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""

    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True

# Symmetric False need small group_size
default_compression_config = CompressionConfig(
    num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True
)

bit4_compression_config = CompressionConfig(
    num_bits=4, group_size=32, group_dim=1, symmetric=True, enabled=True
)

bit2_compression_config = CompressionConfig(
    num_bits=2, group_size=4, group_dim=1, symmetric=False, enabled=True
)

class CLinear(nn.Module):
    """Compressed Linear Layer."""
    __constants__ = ['config']
    config: CompressionConfig
    weight: Union[Tensor, Tuple]
    bias: Tensor

    def __init__(self, compression_config: CompressionConfig, weight=None, bias=None, device=None):
        super().__init__()
        self.config = compression_config
        if weight is None:
            self.weight = None
        elif isinstance(weight, Tensor):
            self.weight = compress(weight.data.to(device), self.config)
        else:
            self.weight = weight
        self.bias = bias
        self.bias_dtype = {}

    def forward(self, input: Tensor) -> Tensor:
        weight = decompress(self.weight, self.config, input.dtype)
        if input.dtype in self.bias_dtype:
            bias = self.bias_dtype[input.dtype]
        else:
            if self.bias is not None:
                bias = self.bias.to(input.dtype)
                self.bias_dtype[input.dtype] = bias
            else:
                bias = self.bias
        return F.linear(input, weight, bias)
    
    def extra_repr(self) -> str:
        return 'num_bits={}, group_size={}, group_dim={}, symmetric={}, bias={}'.format(
            self.config.num_bits,
            self.config.group_size,
            self.config.group_dim,
            self.config.symmetric,
            self.bias is not None
        )

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


def apply_compressed_weight(module, compressed_state_dict, config, prefix=""):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            param_device = "cpu"
            if isinstance(compressed_state_dict[full_name], torch.Tensor):
                param_device = compressed_state_dict[full_name].device
            elif isinstance(compressed_state_dict[full_name], tuple):
                param_device = compressed_state_dict[full_name][0].device
            setattr(
                module,
                attr_str,
                CLinear(
                    config, compressed_state_dict[full_name], target_attr.bias, param_device
                ),
            )
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        apply_compressed_weight(
            child, compressed_state_dict, config, child_prefix
        )

from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
from accelerate.utils import find_tied_parameters, check_tied_parameters_in_config, check_tied_parameters_on_same_device, load_state_dict
from accelerate.utils import get_balanced_memory, WEIGHTS_NAME, SAFE_WEIGHTS_NAME, offload_weight, save_offload_index
from accelerate.utils import load_offloaded_weights, retie_parameters
from collections import defaultdict

def find_device_map(module_name: str, device_map, default=None):
    target_name = ""
    for name, device in device_map.items():
        if module_name.startswith(name):
            if len(name) > len(target_name):
                target_name = name
    
    if len(target_name) == 0:
        return default
    else:
        device = device_map[target_name]
        if isinstance(device, int):
            return f"cuda:{device}"
        else:
            return device


def load_compressed_checkpoint_in_model(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    device_map: Optional[Dict[str, Union[int, str, torch.device]]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: bool = False,
    offload_buffers: bool = False,
    keep_in_fp32_modules: List[str] = None,
    offload_8bit_bnb: bool = False,
):
    if offload_8bit_bnb:
        from accelerate.utils.bnb import quantize_and_offload_8bit

    tied_params = find_tied_parameters(model)

    if check_tied_parameters_in_config(model) and len(tied_params) == 0:
        print(
            "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."
        )

    check_tied_parameters_on_same_device(tied_params, device_map)

    if offload_folder is None and device_map is not None and "disk" in device_map.values():
        raise ValueError(
            "At least one of the model submodule will be offloaded to disk, please pass along an `offload_folder`."
        )
    elif offload_folder is not None and device_map is not None and "disk" in device_map.values():
        os.makedirs(offload_folder, exist_ok=True)

    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)

    checkpoint_files = None
    index_filename = None
    if os.path.isfile(checkpoint):
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]
    elif os.path.isdir(checkpoint):
        # check if the whole state dict is present
        potential_state_bin = [f for f in os.listdir(checkpoint) if f == WEIGHTS_NAME]
        potential_state_safetensor = [f for f in os.listdir(checkpoint) if f == SAFE_WEIGHTS_NAME]
        if len(potential_state_bin) == 1:
            checkpoint_files = [os.path.join(checkpoint, potential_state_bin[0])]
        elif len(potential_state_safetensor) == 1:
            checkpoint_files = [os.path.join(checkpoint, potential_state_safetensor[0])]
        else:
            # otherwise check for sharded checkpoints
            potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
            if len(potential_index) == 0:
                raise ValueError(
                    f"{checkpoint} is not a folder containing a `.index.json` file or a {WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
                )
            elif len(potential_index) == 1:
                index_filename = os.path.join(checkpoint, potential_index[0])
            else:
                raise ValueError(
                    f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                )
    else:
        raise ValueError(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got {checkpoint}."
        )

    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename, "r") as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

    # Logic for missing/unexepected keys goes here.

    offload_index = {}
    if offload_state_dict:
        state_dict_folder = tempfile.mkdtemp()
        state_dict_index = {}

    buffer_names = [name for name, _ in model.named_buffers()]
    for checkpoint_file in checkpoint_files:
        checkpoint = load_state_dict(checkpoint_file, device_map=device_map)
        if device_map is None:
            model.load_state_dict(checkpoint, strict=False)
        else:
            for param_name, param in checkpoint.items():
                # skip SCB parameter (for 8-bit serialization)
                if "SCB" in param_name:
                    continue

                module_name = param_name

                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    # TODO: group all errors and raise at the end.
                    raise ValueError(f"{param_name} doesn't have any device set.")
                param_device = device_map[module_name]
                new_dtype = dtype
                if dtype is not None and torch.is_floating_point(param):
                    if keep_in_fp32_modules is not None and dtype == torch.float16:
                        proceed = False
                        for key in keep_in_fp32_modules:
                            if ((key in param_name) and (key + "." in param_name)) or key == param_name:
                                proceed = True
                                break
                        if proceed:
                            new_dtype = torch.float32

                if "weight" in param_name and param_name.replace("weight", "SCB") in checkpoint.keys():
                    if param.dtype == torch.int8:
                        fp16_statistics = checkpoint[param_name.replace("weight", "SCB")]
                else:
                    fp16_statistics = None

                if param_device == "disk":
                    if offload_buffers or param_name not in buffer_names:
                        if new_dtype is None:
                            new_dtype = param.dtype
                        if offload_8bit_bnb:
                            quantize_and_offload_8bit(
                                model, param, param_name, new_dtype, offload_folder, offload_index, fp16_statistics
                            )
                            continue
                        else:
                            set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, offload_folder, index=offload_index)
                elif param_device == "cpu" and offload_state_dict:
                    if new_dtype is None:
                        new_dtype = param.dtype
                    if offload_8bit_bnb:
                        quantize_and_offload_8bit(
                            model, param, param_name, new_dtype, state_dict_folder, state_dict_index, fp16_statistics
                        )
                    else:
                        set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, state_dict_folder, index=state_dict_index)
                else:
                    set_module_tensor_to_device(
                        model,
                        param_name,
                        param_device,
                        value=param,
                        dtype=new_dtype,
                        fp16_statistics=fp16_statistics,
                    )

        # Force Python to clean up.
        del checkpoint
        gc.collect()

    save_offload_index(offload_index, offload_folder)

    # Load back offloaded state dict on CPU
    if offload_state_dict:
        load_offloaded_weights(model, state_dict_index, state_dict_folder)
        shutil.rmtree(state_dict_folder)

    retie_parameters(model, tied_params)

def load_compress_model(model_path, device, compression_config: CompressionConfig=default_compression_config, **kwargs):
    torch_dtype = kwargs.get("torch_dtype", torch.float16)
    trust_remote_code = kwargs.get("trust_remote_code", False)
    max_memory = kwargs.get("max_memory", None)
    # partially load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **kwargs)
    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        model: nn.Module = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        linear_weights = get_compressed_list(model)
    
    if device == "auto":
        no_split_module_classes = ["LlamaDecoderLayer", "GPTJBlock", "GPT2Block", "GPTBigCodeBlock", "GPTNeoBlock"]
        device_map = None
        if device != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=torch.int8,
                low_zero=(device_map == "balanced_low_0"),
            )
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, dtype=torch.int8
        )
    else:
        device_map = defaultdict(lambda:device)

    compressed_state_dict = {}
    base_pattern = os.path.join(model_path, "pytorch_model-*.bin")
    files = glob.glob(base_pattern)
    for filename in tqdm(files):
        tmp_state_dict = torch.load(filename)
        for name in tmp_state_dict:
            param_device = find_device_map(name, device_map, device)
            if name in linear_weights:
                tensor = tmp_state_dict[name].to(param_device).data.to(torch_dtype)
                compressed_state_dict[name] = compress(
                    tensor, compression_config
                )
            else:
                compressed_state_dict[name] = tmp_state_dict[name].to(param_device)
            tmp_state_dict[name] = None
            tensor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for name, buffer in model.named_buffers():
        if name not in linear_weights:
            param_device = find_device_map(name, device_map, device)
            if name in compressed_state_dict:
                set_module_tensor_to_device(
                    model, name, param_device, value=compressed_state_dict[name]
                )
            else:
                set_module_tensor_to_device(
                    model, name, param_device, value=buffer
                )

    for name in model.state_dict():
        if name not in linear_weights:
            param_device = find_device_map(name, device_map, device)
            if name in compressed_state_dict:
                set_module_tensor_to_device(
                    model, name, param_device, value=compressed_state_dict[name]
                )
    apply_compressed_weight(model, compressed_state_dict, compression_config)

    if device != "auto":
        model.to(device)
    else:
        dispatch_model(
            model,
            device_map=device_map,
            offload_dir=None,
            offload_buffers=False,
            skip_keys=None,
            preload_module_classes=None,
        )
        model.tie_weights()

    for name, tensor in model.state_dict().items():
        print(f"State Name: {name}, Device: {tensor.device}")

    return model, tokenizer

"""
def load_compress_model(model_path, device, torch_dtype, compression_config: CompressionConfig=default_compression_config, trust_remote_code: bool=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=trust_remote_code)
    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )
        model: nn.Module = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)

    # Load the checkpoint and dispatch it to the right devices
    model = load_checkpoint_and_dispatch(
        model, model_path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"],
    )
    
    return model, tokenizer
"""

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
    B = 2 ** (num_bits - 1) - 1
    if num_bits >= 8:
        if symmetric:
            mn = None
            scale = B * torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0].pow_(-1)
            data = data * scale
            data = data.clamp_(-B, B).round_().type(torch.int8)
        else:
            mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
            mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

            scale = B / (mx - mn)
            data = data - mn
            data.mul_(scale)
            data = data.clamp_(0, B).round_().type(torch.uint8)

        inv_scale = 1.0 / scale
        return data, mn, inv_scale, original_shape
    elif num_bits > 2:
        if symmetric:
            mn = None
            scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
            data = data * scale
            data = data.clamp_(-B, B).round_().to(torch.int16)
            data = (data + B).to(torch.uint8)
        else:
            mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
            mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

            scale = B / (mx - mn)
            data = data - mn
            data.mul_(scale)

            data = data.clamp_(0, B).round_().to(torch.uint8)
        
        left_half = [slice(None) for i in new_shape]
        left_half[group_dim + 1] = slice(0, group_size//2)
        final_data.bitwise_or_(torch.bitwise_left_shift(data[left_half], 4))

        right_half = [slice(None) for i in new_shape]
        right_half[group_dim + 1] = slice(group_size//2, group_size)
        final_data.bitwise_or_(data[right_half])

        inv_scale = 1.0 / scale
        return final_data, mn, inv_scale, original_shape
    else:
        if symmetric:
            mn = None
            scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
            data = data * scale
            data = data.clamp_(-B, B).round_().to(torch.int16)
            data = (data + B).to(torch.uint8)
        else:
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

        inv_scale = 1.0 / scale
        return final_data, mn, inv_scale, original_shape

@torch.jit.script
def unpack_int4(data: torch.Tensor, group_dim: int, group_size: int, new_shape: List[int], dtype: torch.dtype):
    left_half = torch.bitwise_right_shift(data, 4)
    right_half = torch.bitwise_and(data, 0b00001111)
    float_data = torch.concat((left_half, right_half), dim=group_dim + 1).to(dtype)
    return float_data

def decompress(packed_data, config, dtype=torch.float32):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )

    data, mn, inv_scale, original_shape = packed_data
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    B = 2 ** (num_bits - 1) - 1
    if num_bits >= 8:
        # Dequantize
        if symmetric:
            # data = data.to(dtype) * inv_scale.to(dtype)
            data = data.type(dtype)
            data.mul_(inv_scale)
        else:
            data = data.to(dtype)
            data = data.mul_(inv_scale).add_(mn)
    elif num_bits > 2:
        new_shape = [i for i in data.shape]
        new_shape[group_dim + 1] = group_size

        float_data = torch.empty(size=new_shape, dtype=dtype, device=data.device)
        left_half = [slice(None) for i in new_shape]
        left_half[group_dim + 1] = slice(0, group_size//2)
        float_data[left_half] = torch.bitwise_right_shift(data, 4)
        
        right_half = [slice(None) for i in new_shape]
        right_half[group_dim + 1] = slice(group_size//2, group_size)
        float_data[right_half] = torch.bitwise_and(data, 0b00001111)

        # float_data = unpack_int4(data, group_dim, group_size, new_shape, dtype)

        if symmetric:
            float_data = float_data.sub_(B)
            data = float_data.mul_(inv_scale)
        else:
            # data = float_data.mul_(inv_scale).add_(mn)
            data = torch.addcmul(mn, float_data, inv_scale, value=1.0)
    else:
        new_shape = [i for i in data.shape]
        new_shape[group_dim + 1] = 4 * new_shape[group_dim + 1]
        float_data = torch.empty(size=new_shape, dtype=dtype, device=data.device)

        first_half = [slice(None) for i in new_shape]
        first_half[group_dim + 1] = slice(0, group_size//4)
        first_mask = torch.tensor([0b11000000], dtype=torch.uint8, device=data.device)
        float_data[first_half] = torch.bitwise_right_shift(torch.bitwise_and(data, first_mask), 6)

        second_half = [slice(None) for i in new_shape]
        second_half[group_dim + 1] = slice(group_size//4, group_size//4*2)
        second_mask = torch.tensor([0b00110000], dtype=torch.uint8, device=data.device)
        float_data[second_half] = torch.bitwise_right_shift(torch.bitwise_and(data, second_mask), 4)
        
        third_half = [slice(None) for i in new_shape]
        third_half[group_dim + 1] = slice(group_size//4*2, group_size//4*3)
        third_mask = torch.tensor([0b00001100], dtype=torch.uint8, device=data.device)
        float_data[third_half] = torch.bitwise_right_shift(torch.bitwise_and(data, third_mask), 2)

        fourth_half = [slice(None) for i in new_shape]
        fourth_half[group_dim + 1] = slice(group_size//4*3, group_size)
        fourth_mask = torch.tensor([0b00000011], dtype=torch.uint8, device=data.device)
        float_data[fourth_half] = torch.bitwise_and(data, fourth_mask)
        
        if symmetric:
            float_data = float_data.sub_(B)
            data = float_data.mul_(inv_scale)
        else:
            data = float_data.mul_(inv_scale).add_(mn)
    
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
