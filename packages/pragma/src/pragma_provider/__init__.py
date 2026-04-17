"""Pragma platform provider for built-in resources.

Provides pragma/secret, pragma/config, and pragma/file resources for managing
secrets, non-sensitive configuration, and file storage declaratively through
Pragmatiks.
"""

from pragma_sdk import Provider

from pragma_provider.resources.config import ConfigResource, ConfigResourceConfig, ConfigResourceOutputs
from pragma_provider.resources.file import File, FileConfig, FileOutputs
from pragma_provider.resources.secret import Secret, SecretConfig, SecretOutputs


pragma = Provider()

pragma.resource("secret")(Secret)
pragma.resource("config")(ConfigResource)
pragma.resource("file")(File)

__all__ = [
    "pragma",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
    "ConfigResource",
    "ConfigResourceConfig",
    "ConfigResourceOutputs",
    "File",
    "FileConfig",
    "FileOutputs",
]
