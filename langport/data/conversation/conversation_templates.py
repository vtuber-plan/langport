

import importlib
import os
import pkgutil
from typing import Dict

from langport.data.conversation import Conversation


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


templates_path = os.path.join(os.path.dirname(__file__), "templates")
for module_loader, name, ispkg in pkgutil.iter_modules([templates_path]):
    # print(module_loader, name, ispkg)
    module = importlib.import_module(".templates." + name, __package__)
    for name, values in vars(module).items():
        if not isinstance(values, Conversation):
            continue
        register_conv_template(values)

