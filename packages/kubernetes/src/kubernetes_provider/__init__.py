"""Kubernetes provider for Pragmatiks.

Provides generic Kubernetes resources using lightkube for managing
workloads in GKE clusters using user-provided credentials.
"""

from pragma_sdk import Provider

from kubernetes_provider.client import create_client_from_gke
from kubernetes_provider.resources import (
    ConfigMap,
    ConfigMapConfig,
    ConfigMapOutputs,
    Deployment,
    DeploymentConfig,
    DeploymentOutputs,
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


kubernetes = Provider(name="kubernetes")

kubernetes.resource("deployment")(Deployment)
kubernetes.resource("service")(Service)
kubernetes.resource("configmap")(ConfigMap)
kubernetes.resource("secret")(Secret)
kubernetes.resource("statefulset")(StatefulSet)

__all__ = [
    "kubernetes",
    "create_client_from_gke",
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
