"""Resource definitions for kubernetes provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from kubernetes_provider.resources.configmap import (
    ConfigMap,
    ConfigMapConfig,
    ConfigMapOutputs,
)
from kubernetes_provider.resources.deployment import (
    Deployment,
    DeploymentConfig,
    DeploymentOutputs,
)
from kubernetes_provider.resources.secret import (
    Secret,
    SecretConfig,
    SecretOutputs,
)
from kubernetes_provider.resources.service import (
    Service,
    ServiceConfig,
    ServiceOutputs,
)
from kubernetes_provider.resources.statefulset import (
    StatefulSet,
    StatefulSetConfig,
    StatefulSetOutputs,
)


__all__ = [
    "ConfigMap",
    "ConfigMapConfig",
    "ConfigMapOutputs",
    "Deployment",
    "DeploymentConfig",
    "DeploymentOutputs",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
    "Service",
    "ServiceConfig",
    "ServiceOutputs",
    "StatefulSet",
    "StatefulSetConfig",
    "StatefulSetOutputs",
]
