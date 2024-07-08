"""Model adapter registration."""

import importlib
import pkgutil
import sys
import os
from typing import List
import warnings
from functools import cache

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings


class BaseAdapter:
    """The base and the default model adapter."""

    def match(self, model_path: str) -> bool:
        return True

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("zero_shot")
        return ConversationHistory(
            system="A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            messages=[],
            offset=2,
            settings=settings,
        )


# A global registry for all model adapters
model_adapters: List[BaseAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_path: str) -> BaseAdapter:
    """Get a model adapter for a model_path."""
    for adapter in model_adapters:
        if adapter.match(model_path):
            print(f"Using model adapter {adapter.__class__.__name__}")
            return adapter
    raise ValueError(f"No valid model adapter for {model_path}")


def raise_warning_for_incompatible_cpu_offloading_configuration(
    device: str, load_8bit: bool, cpu_offloading: bool
):
    if cpu_offloading:
        if not load_8bit:
            warnings.warn(
                "The cpu-offloading feature can only be used while also using 8-bit-quantization.\n"
                "Use '--load-8bit' to enable 8-bit-quantization\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if not "linux" in sys.platform:
            warnings.warn(
                "CPU-offloading is only supported on linux-systems due to the limited compatability with the bitsandbytes-package\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if device != "cuda":
            warnings.warn(
                "CPU-offloading is only enabled when using CUDA-devices\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
    return cpu_offloading


def get_conversation_template(model_path: str) -> ConversationHistory:
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

adapters_path = os.path.join(os.path.dirname(__file__), "adapters")
for module_loader, name, ispkg in pkgutil.iter_modules([adapters_path]):
    # print(module_loader, name, ispkg)
    importlib.import_module(".adapters." + name, __package__)

# Register all adapters.
for cls in BaseAdapter.__subclasses__():
    register_model_adapter(cls)

# After all adapters, try the default base adapter.
register_model_adapter(BaseAdapter)
