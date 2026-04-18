"""Pragma platform resource implementations."""

from pragma_provider.resources.config import ConfigResource, ConfigResourceConfig, ConfigResourceOutputs
from pragma_provider.resources.file import File, FileConfig, FileOutputs
from pragma_provider.resources.secret import Secret, SecretConfig, SecretOutputs


__all__ = [
    "ConfigResource",
    "ConfigResourceConfig",
    "ConfigResourceOutputs",
    "File",
    "FileConfig",
    "FileOutputs",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
]
