"""Kubernetes provider for Pragmatiks.

Provides generic Kubernetes resources using lightkube for managing
workloads against any cluster reachable through a ``kubernetes/config``
resource.
"""

from pragma_sdk import Provider

from kubernetes_provider.client import build_kubeconfig_from_gke, create_client_from_gke
from kubernetes_provider.resources import (
    ConfigConfig,
    ConfigMap,
    ConfigMapConfig,
    ConfigMapOutputs,
    ConfigOutputs,
    Deployment,
    DeploymentConfig,
    DeploymentOutputs,
    KubernetesConfig,
    Namespace,
    NamespaceConfig,
    NamespaceOutputs,
    Secret,
    SecretConfig,
    SecretOutputs,
    Service,
    ServiceConfig,
    ServiceOutputs,
    StatefulSet,
    StatefulSetConfig,
    StatefulSetOutputs,
)


kubernetes = Provider()

kubernetes.resource("config")(KubernetesConfig)
kubernetes.resource("deployment")(Deployment)
kubernetes.resource("service")(Service)
kubernetes.resource("configmap")(ConfigMap)
kubernetes.resource("secret")(Secret)
kubernetes.resource("statefulset")(StatefulSet)
kubernetes.resource("namespace")(Namespace)

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
    "build_kubeconfig_from_gke",
    "create_client_from_gke",
    "kubernetes",
]
