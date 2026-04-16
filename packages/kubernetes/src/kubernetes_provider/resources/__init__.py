"""Resource definitions for kubernetes provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from kubernetes_provider.resources.config import (
    ConfigConfig,
    ConfigOutputs,
    KubernetesConfig,
)
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
from kubernetes_provider.resources.namespace import (
    Namespace,
    NamespaceConfig,
    NamespaceOutputs,
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
    "ConfigConfig",
    "ConfigMap",
    "ConfigMapConfig",
    "ConfigMapOutputs",
    "ConfigOutputs",
    "Deployment",
    "DeploymentConfig",
    "DeploymentOutputs",
    "KubernetesConfig",
    "Namespace",
    "NamespaceConfig",
    "NamespaceOutputs",
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
