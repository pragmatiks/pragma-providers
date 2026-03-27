"""Vercel Deployment resource."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField

from vercel_provider.client import create_vercel_client, raise_for_status


class DeploymentConfig(Config):
    """Configuration for a Vercel deployment.

    Attributes:
        access_token: Vercel API token for authentication.
        project_id: Vercel project ID to deploy. Typically a dependency
            reference to a Vercel project resource.
        project_name: Vercel project name. Used as the deployment target.
        git_ref: Git reference (branch, tag, or SHA) to deploy.
            When None, deploys the default branch.
        target: Deployment target environment (``production`` or ``preview``).
        team_id: Vercel team ID. Required for team-owned projects.
    """

    access_token: SensitiveField[str]
    project_id: ImmutableField[str]
    project_name: ImmutableField[str]
    git_ref: Field[str] | None = None
    target: Field[str] = "production"
    team_id: Field[str] | None = None


class DeploymentOutputs(Outputs):
    """Outputs from Vercel deployment creation.

    Attributes:
        deployment_id: Unique deployment identifier assigned by Vercel.
        url: Deployment URL.
        state: Deployment state (e.g., ``READY``, ``BUILDING``, ``ERROR``).
        ready_state: Final readiness state of the deployment.
        project_id: Project ID this deployment belongs to.
    """

    deployment_id: str
    url: str
    state: str
    ready_state: str
    project_id: str


_POLL_INTERVAL_SECONDS = 10
_MAX_POLL_ATTEMPTS = 60  # 60 * 10s = 10 minutes max wait


class Deployment(Resource[DeploymentConfig, DeploymentOutputs]):
    """Vercel deployment resource.

    Triggers and manages deployments for a Vercel project via the REST API.
    Deployments are triggered by creating a new deployment with a Git reference
    or by redeploying the latest deployment.

    Lifecycle:
        - on_create: Triggers a new deployment and polls until it reaches a
          terminal state (READY or ERROR). Not idempotent -- duplicate calls
          create duplicate deployments.
        - on_update: Triggers a new deployment with the updated configuration.
          Vercel deployments are immutable; updates create new deployments.
        - on_delete: Deletes the deployment. Idempotent -- succeeds if the
          deployment does not exist.

    Example::

        resources:
          - name: my-app-deploy
            provider: vercel
            type: deployment
            config:
              access_token:
                provider: pragma
                resource: secret
                name: vercel-token
                field: outputs.value
              project_id:
                provider: vercel
                resource: project
                name: my-app
                field: outputs.project_id
              project_name:
                provider: vercel
                resource: project
                name: my-app
                field: outputs.name
              target: production
    """

    def _query_params(self) -> dict[str, str]:
        """Build common query parameters for API requests.

        Returns:
            Dictionary of query parameters including team_id if configured.
        """
        params: dict[str, str] = {}

        if self.config.team_id:
            params["teamId"] = self.config.team_id

        return params

    async def _wait_for_ready(self, client: httpx.AsyncClient, deployment_id: str) -> dict[str, Any]:
        """Poll deployment status until it reaches a terminal state.

        Args:
            client: Authenticated Vercel API client.
            deployment_id: Deployment ID to poll.

        Returns:
            Deployment data dictionary from the API.

        Raises:
            RuntimeError: If the deployment reaches a non-READY terminal state
                (ERROR or CANCELED).
            TimeoutError: If the deployment does not reach a terminal state in time.
        """
        terminal_states = {"READY", "ERROR", "CANCELED"}

        for _ in range(_MAX_POLL_ATTEMPTS):
            response = await client.get(f"/v13/deployments/{deployment_id}", params=self._query_params())

            if response.status_code in {401, 403, 404}:
                await raise_for_status(response)

            if response.is_success:
                data = response.json()
                ready_state = data.get("readyState", "")

                if ready_state in terminal_states:
                    if ready_state != "READY":
                        msg = f"Deployment {deployment_id} reached terminal state {ready_state}"
                        raise RuntimeError(msg)

                    return data

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = (
            f"Deployment {deployment_id} did not complete within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS} seconds"
        )
        raise TimeoutError(msg)

    def _build_outputs(self, data: dict[str, Any]) -> DeploymentOutputs:
        """Build outputs from Vercel deployment API response data.

        Args:
            data: Raw deployment data from the Vercel API.

        Returns:
            DeploymentOutputs with deployment metadata.
        """
        return DeploymentOutputs(
            deployment_id=data["id"],
            url=data.get("url", ""),
            state=data.get("state", "UNKNOWN"),
            ready_state=data.get("readyState", "UNKNOWN"),
            project_id=data.get("projectId", self.config.project_id),
        )

    async def _trigger_deployment(self) -> DeploymentOutputs:
        """Trigger a new deployment and wait for completion.

        Returns:
            DeploymentOutputs with deployment details.
        """
        client = create_vercel_client(self.config.access_token)

        try:
            body: dict[str, Any] = {
                "name": self.config.project_name,
                "project": self.config.project_id,
                "target": self.config.target,
            }

            if self.config.git_ref is not None:
                body["gitSource"] = {
                    "ref": self.config.git_ref,
                    "type": "github",
                }

            response = await client.post("/v13/deployments", params=self._query_params(), json=body)
            await raise_for_status(response)
            deployment_data = response.json()
            deployment_id = deployment_data["id"]

            deployment_data = await self._wait_for_ready(client, deployment_id)

            return self._build_outputs(deployment_data)
        finally:
            await client.aclose()

    async def on_create(self) -> DeploymentOutputs:
        """Trigger a new deployment and wait until complete.

        Returns:
            DeploymentOutputs with deployment details.
        """
        return await self._trigger_deployment()

    async def on_update(self, previous_config: DeploymentConfig) -> DeploymentOutputs:
        """Trigger a new deployment with updated configuration.

        Vercel deployments are immutable, so updates always create a new
        deployment.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            DeploymentOutputs with new deployment details.
        """
        return await self._trigger_deployment()

    async def on_delete(self) -> None:
        """Delete the Vercel deployment.

        Idempotent: Succeeds if the deployment does not exist.
        """
        if self.outputs is None:
            return

        client = create_vercel_client(self.config.access_token)

        try:
            response = await client.delete(
                f"/v13/deployments/{self.outputs.deployment_id}",
                params=self._query_params(),
            )

            if response.status_code == 404:
                return

            await raise_for_status(response)
        finally:
            await client.aclose()

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs
