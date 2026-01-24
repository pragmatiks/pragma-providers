"""Agno Agent resource - deploys AI agents to GKE via kubectl."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pragma_sdk import Config, Dependency, Outputs, Resource

if TYPE_CHECKING:
    from gcp_provider import GKE


class AgentConfig(Config):
    """Configuration for an Agno agent deployment.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        model: Model dependency (anthropic/messages or openai/chat) for agent LLM.
        embeddings: Optional embeddings dependency for RAG capabilities.
        vector_store: Optional vector store dependency (qdrant collection) for RAG.
        instructions: System instructions for the agent.
        image: Container image for the agent deployment.
        replicas: Number of agent replicas.
    """

    cluster: Dependency["GKE"]
    model: Dependency
    embeddings: Dependency | None = None
    vector_store: Dependency | None = None
    instructions: str | None = None
    image: str = "ghcr.io/agno-ai/agno:latest"
    replicas: int = 1


class AgentOutputs(Outputs):
    """Outputs from Agno agent deployment.

    Attributes:
        url: In-cluster URL for agent API.
        ready: Whether the Deployment is ready.
    """

    url: str
    ready: bool


# Polling configuration for Deployment readiness
_POLL_INTERVAL_SECONDS = 5
_MAX_POLL_ATTEMPTS = 60  # 60 * 5s = 5 minutes max wait


class Agent(Resource[AgentConfig, AgentOutputs]):
    """Agno AI agent deployed to GKE via kubectl.

    Deploys an Agno agent as a Kubernetes Deployment with a Service.
    Uses cluster credentials from the GKE dependency to authenticate
    with the Kubernetes API. Model, embeddings, and vector store URLs
    are passed as environment variables to the container.

    Lifecycle:
        - on_create: Apply Deployment + Service, wait for ready
        - on_update: Update Deployment with new config
        - on_delete: Delete Deployment and Service
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "agent"

    def _get_deployment_name(self) -> str:
        """Get Kubernetes Deployment name based on resource name."""
        return f"agno-{self.name}"

    def _get_namespace(self) -> str:
        """Get Kubernetes namespace."""
        return "default"

    async def _get_kubeconfig(self) -> str:
        """Build kubeconfig from GKE cluster credentials.

        Returns:
            Kubeconfig content as string.
        """
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs
        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        kubeconfig = f"""
apiVersion: v1
kind: Config
clusters:
- name: gke-cluster
  cluster:
    server: https://{outputs.endpoint}
    certificate-authority-data: {outputs.cluster_ca_certificate}
contexts:
- name: gke-context
  context:
    cluster: gke-cluster
    user: gke-user
current-context: gke-context
users:
- name: gke-user
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: gke-gcloud-auth-plugin
      installHint: Install gke-gcloud-auth-plugin for kubectl auth
      provideClusterInfo: true
"""
        return kubeconfig

    async def _get_env_vars(self) -> list[dict[str, str]]:
        """Build environment variables for the agent container.

        Resolves model, embeddings, and vector_store dependencies
        to extract their URLs.

        Returns:
            List of env var dicts with name and value keys.
        """
        env_vars: list[dict[str, str]] = []

        # Model URL (required)
        model = await self.config.model.resolve()
        if model.outputs is not None and hasattr(model.outputs, "url"):
            env_vars.append({"name": "AGNO_MODEL_URL", "value": model.outputs.url})

        # Embeddings URL (optional)
        if self.config.embeddings is not None:
            embeddings = await self.config.embeddings.resolve()
            if embeddings.outputs is not None and hasattr(embeddings.outputs, "url"):
                env_vars.append({"name": "AGNO_EMBEDDINGS_URL", "value": embeddings.outputs.url})

        # Vector store URL (optional)
        if self.config.vector_store is not None:
            vector_store = await self.config.vector_store.resolve()
            if vector_store.outputs is not None and hasattr(vector_store.outputs, "url"):
                env_vars.append({"name": "AGNO_VECTOR_STORE_URL", "value": vector_store.outputs.url})

        # Instructions (optional)
        if self.config.instructions:
            env_vars.append({"name": "AGNO_INSTRUCTIONS", "value": self.config.instructions})

        return env_vars

    def _build_deployment_manifest(self, env_vars: list[dict[str, str]]) -> dict[str, Any]:
        """Build Kubernetes Deployment manifest.

        Args:
            env_vars: Environment variables for the container.

        Returns:
            Deployment manifest as dict.
        """
        deployment_name = self._get_deployment_name()
        namespace = self._get_namespace()

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "namespace": namespace,
                "labels": {
                    "app": deployment_name,
                    "pragma.io/provider": "agno",
                    "pragma.io/resource": "agent",
                    "pragma.io/name": self.name,
                },
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": deployment_name,
                    },
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": deployment_name,
                        },
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "agent",
                                "image": self.config.image,
                                "ports": [{"containerPort": 8080}],
                                "env": env_vars,
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080,
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 10,
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080,
                                    },
                                    "initialDelaySeconds": 15,
                                    "periodSeconds": 20,
                                },
                            },
                        ],
                    },
                },
            },
        }

    def _build_service_manifest(self) -> dict[str, Any]:
        """Build Kubernetes Service manifest.

        Returns:
            Service manifest as dict.
        """
        deployment_name = self._get_deployment_name()
        namespace = self._get_namespace()

        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": deployment_name,
                "namespace": namespace,
                "labels": {
                    "app": deployment_name,
                    "pragma.io/provider": "agno",
                    "pragma.io/resource": "agent",
                    "pragma.io/name": self.name,
                },
            },
            "spec": {
                "selector": {
                    "app": deployment_name,
                },
                "ports": [
                    {
                        "port": 80,
                        "targetPort": 8080,
                        "protocol": "TCP",
                    },
                ],
                "type": "ClusterIP",
            },
        }

    async def _run_command(self, cmd: list[str], kubeconfig_path: str, name: str) -> subprocess.CompletedProcess:
        """Run a CLI command with kubeconfig.

        Args:
            cmd: Full command with arguments.
            kubeconfig_path: Path to kubeconfig file.
            name: Command name for error messages.

        Returns:
            Completed process result.

        Raises:
            RuntimeError: If command fails.
        """
        env = {"KUBECONFIG": kubeconfig_path, "PATH": os.environ.get("PATH", "")}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            ),
        )

        if result.returncode != 0:
            msg = f"{name} command failed: {result.stderr}"
            raise RuntimeError(msg)

        return result

    async def _run_kubectl(self, args: list[str], kubeconfig_path: str) -> subprocess.CompletedProcess:
        """Run kubectl command with kubeconfig."""
        return await self._run_command(["kubectl", *args], kubeconfig_path, "kubectl")

    async def _apply_manifest(self, manifest: dict[str, Any], kubeconfig_path: str) -> None:
        """Apply a Kubernetes manifest via kubectl.

        Args:
            manifest: Kubernetes manifest as dict.
            kubeconfig_path: Path to kubeconfig file.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest, f)
            manifest_path = f.name

        try:
            await self._run_kubectl(["apply", "-f", manifest_path], kubeconfig_path)
        finally:
            Path(manifest_path).unlink(missing_ok=True)

    async def _delete_resource(self, kind: str, name: str, kubeconfig_path: str) -> None:
        """Delete a Kubernetes resource.

        Args:
            kind: Resource kind (e.g., "deployment", "service").
            name: Resource name.
            kubeconfig_path: Path to kubeconfig file.
        """
        namespace = self._get_namespace()
        try:
            await self._run_kubectl(
                ["delete", kind, name, "-n", namespace, "--ignore-not-found"],
                kubeconfig_path,
            )
        except RuntimeError:
            pass  # Resource may not exist

    async def _wait_for_ready(self, kubeconfig_path: str) -> bool:
        """Wait for Deployment to be ready.

        Args:
            kubeconfig_path: Path to kubeconfig file.

        Returns:
            True when Deployment is ready.

        Raises:
            TimeoutError: If Deployment doesn't become ready in time.
        """
        namespace = self._get_namespace()
        deployment_name = self._get_deployment_name()

        for _ in range(_MAX_POLL_ATTEMPTS):
            try:
                result = await self._run_kubectl(
                    [
                        "get",
                        "deployment",
                        deployment_name,
                        "-n",
                        namespace,
                        "-o",
                        "jsonpath={.status.readyReplicas}",
                    ],
                    kubeconfig_path,
                )
                ready_replicas = int(result.stdout.strip()) if result.stdout.strip() else 0
                if ready_replicas >= self.config.replicas:
                    return True
            except (RuntimeError, ValueError):
                pass  # Deployment may not exist yet or parsing failed

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = f"Deployment {deployment_name} did not become ready within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS}s"
        raise TimeoutError(msg)

    def _dependency_changed(self, prev: Dependency | None, curr: Dependency | None) -> bool:
        """Check if an optional dependency changed.

        Args:
            prev: Previous dependency (may be None).
            curr: Current dependency (may be None).

        Returns:
            True if the dependency changed (added, removed, or different ID).
        """
        if (prev is None) != (curr is None):
            return True
        if prev is not None and curr is not None and prev.id != curr.id:
            return True
        return False

    def _build_outputs(self) -> AgentOutputs:
        """Build outputs with in-cluster service URL."""
        deployment_name = self._get_deployment_name()
        namespace = self._get_namespace()

        url = f"http://{deployment_name}.{namespace}.svc.cluster.local"

        return AgentOutputs(
            url=url,
            ready=True,
        )

    async def on_create(self) -> AgentOutputs:
        """Deploy Agno agent to GKE via kubectl.

        Idempotent: Uses kubectl apply which handles both
        initial creation and updates.

        Returns:
            AgentOutputs with in-cluster service URL.
        """
        kubeconfig = await self._get_kubeconfig()
        env_vars = await self._get_env_vars()

        deployment = self._build_deployment_manifest(env_vars)
        service = self._build_service_manifest()

        with tempfile.TemporaryDirectory() as tmpdir:
            kubeconfig_path = Path(tmpdir) / "kubeconfig"
            kubeconfig_path.write_text(kubeconfig)

            await self._apply_manifest(deployment, str(kubeconfig_path))
            await self._apply_manifest(service, str(kubeconfig_path))
            await self._wait_for_ready(str(kubeconfig_path))

        return self._build_outputs()

    async def on_update(self, previous_config: AgentConfig) -> AgentOutputs:
        """Update Agno agent deployment.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            AgentOutputs with in-cluster service URL.
        """
        # Check for immutable field changes
        if previous_config.cluster.id != self.config.cluster.id:
            msg = "Cannot change cluster; delete and recreate resource"
            raise ValueError(msg)

        # If nothing significant changed, return existing outputs
        if self.outputs is not None:
            previous_dict = previous_config.model_dump(exclude={"cluster", "model", "embeddings", "vector_store"})
            current_dict = self.config.model_dump(exclude={"cluster", "model", "embeddings", "vector_store"})
            deps_changed = (
                previous_config.model.id != self.config.model.id
                or self._dependency_changed(previous_config.embeddings, self.config.embeddings)
                or self._dependency_changed(previous_config.vector_store, self.config.vector_store)
            )
            if previous_dict == current_dict and not deps_changed:
                return self.outputs

        kubeconfig = await self._get_kubeconfig()
        env_vars = await self._get_env_vars()

        deployment = self._build_deployment_manifest(env_vars)
        service = self._build_service_manifest()

        with tempfile.TemporaryDirectory() as tmpdir:
            kubeconfig_path = Path(tmpdir) / "kubeconfig"
            kubeconfig_path.write_text(kubeconfig)

            await self._apply_manifest(deployment, str(kubeconfig_path))
            await self._apply_manifest(service, str(kubeconfig_path))
            await self._wait_for_ready(str(kubeconfig_path))

        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete Agno agent Deployment and Service.

        Idempotent: Succeeds if resources don't exist.
        """
        kubeconfig = await self._get_kubeconfig()
        deployment_name = self._get_deployment_name()

        with tempfile.TemporaryDirectory() as tmpdir:
            kubeconfig_path = Path(tmpdir) / "kubeconfig"
            kubeconfig_path.write_text(kubeconfig)

            await self._delete_resource("deployment", deployment_name, str(kubeconfig_path))
            await self._delete_resource("service", deployment_name, str(kubeconfig_path))
