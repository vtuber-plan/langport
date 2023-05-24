

import importlib
import os
import pkgutil
from typing import Dict

from langport.data.conversation import ConversationHistory, ConversationSettings


# A global registry for all conversation settings
conv_settings: Dict[str, ConversationSettings] = {}


def register_conv_settings(template: ConversationSettings, override: bool = False):
    """Register a new ConversationSettings template."""
    if not override:
        assert template.name not in conv_settings, f"{template.name} has been registered."
    conv_settings[template.name] = template


def get_conv_settings(name: str) -> ConversationSettings:
    """Get a ConversationSettings template."""
    return conv_settings[name].copy()


settings_path = os.path.join(os.path.dirname(__file__), "settings")
for module_loader, name, ispkg in pkgutil.iter_modules([settings_path]):
    # print(module_loader, name, ispkg)
    module = importlib.import_module(".settings." + name, __package__)
    for name, values in vars(module).items():
        if not isinstance(values, ConversationSettings):
            continue
        register_conv_settings(values)

