"""Vercel Deployment resource."""

from __future__ import annotations

import asyncio
import time
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
        team_id: Vercel team ID. Required for team-owned projects. Immutable.
    """

    access_token: SensitiveField[str]
    project_id: ImmutableField[str]
    project_name: ImmutableField[str]
    git_ref: Field[str] | None = None
    target: Field[str] = "production"
    team_id: ImmutableField[str] | None = None


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


POLL_INTERVAL_SECONDS = 10
READY_TIMEOUT_SECONDS = 900
RESOURCE_ID_META_KEY = "pragma_resource"
GIT_REF_META_KEY = "pragma_git_ref"
TARGET_META_KEY = "pragma_target"
IN_FLIGHT_STATES = frozenset({"QUEUED", "INITIALIZING", "BUILDING"})
TERMINAL_STATES = frozenset({"READY", "ERROR", "CANCELED"})
CURRENT_STATES = frozenset({"READY", *IN_FLIGHT_STATES})


class Deployment(Resource[DeploymentConfig, DeploymentOutputs]):
    """Vercel deployment resource.

    Triggers and manages one deployment for a Vercel project via the REST API.
    The deployment is located by its ``meta.pragma_resource`` stamp, set to the resource ID.

    Lifecycle:
        - on_create: Triggers a new deployment and polls until it is READY.
        - on_observe: Finds this resource's deployment.
        - on_update: Keeps the deployment if it is current, otherwise replaces it.
        - on_delete: Deletes the deployment. Succeeds if there is none.

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
                (ERROR or CANCELED); the message carries Vercel's error code and message.
            TimeoutError: If the deployment is not in a terminal state within ``READY_TIMEOUT_SECONDS``; the
                message carries the last ready state and the last HTTP status.
            httpx.HTTPStatusError: If Vercel answers 401, 403 or 404.
        """
        last_http_status: int | None = None
        last_ready_state: str | None = None
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS

        while True:
            response = await client.get(f"/v13/deployments/{deployment_id}", params=self._query_params())

            if response.status_code in {401, 403, 404}:
                await raise_for_status(response)

            last_http_status = response.status_code

            if response.is_success:
                data = response.json()
                last_ready_state = data.get("readyState")

                if last_ready_state == "READY":
                    return data

                if last_ready_state in TERMINAL_STATES:
                    msg = (
                        f"Deployment {deployment_id} reached terminal state {last_ready_state} "
                        f"(errorCode={data.get('errorCode')} errorMessage={data.get('errorMessage')})"
                    )
                    raise RuntimeError(msg)

            if time.monotonic() >= deadline:
                break

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        msg = (
            f"Deployment {deployment_id} did not complete within {READY_TIMEOUT_SECONDS} seconds "
            f"(last_ready_state={last_ready_state} last_http_status={last_http_status})"
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

    async def _trigger_deployment(self, client: httpx.AsyncClient) -> DeploymentOutputs:
        """Trigger a new deployment and wait for completion.

        Args:
            client: Authenticated Vercel API client.

        Returns:
            DeploymentOutputs with deployment details.
        """
        body: dict[str, Any] = {
            "name": self.config.project_name,
            "project": self.config.project_id,
            "target": self.config.target,
            "meta": {
                RESOURCE_ID_META_KEY: self.id,
                GIT_REF_META_KEY: self.config.git_ref or "",
                TARGET_META_KEY: self.config.target,
            },
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

    def is_live_stamped_deployment(self, deployment: dict[str, Any]) -> bool:
        """Tell whether a listed deployment carries this resource's ID and is not deleted.

        Args:
            deployment: Deployment entry from the Vercel deployments list.

        Returns:
            ``True`` when it carries this resource's ID and is not deleted.
        """
        meta = deployment.get("meta", {})

        return meta.get(RESOURCE_ID_META_KEY) == self.id and deployment.get("readyState") != "DELETED"

    def is_current(self, deployment: dict[str, Any]) -> bool:
        """Tell whether a deployment is READY or in flight, built from the current configuration.

        Args:
            deployment: Deployment data from the Vercel API.

        Returns:
            ``True`` when READY, queued or building, and built from the configured ``git_ref`` and ``target``.
        """
        meta = deployment.get("meta", {})

        return (
            deployment.get("readyState") in CURRENT_STATES
            and meta.get(GIT_REF_META_KEY) == (self.config.git_ref or "")
            and meta.get(TARGET_META_KEY) == self.config.target
        )

    async def find_deployment_id(self, client: httpx.AsyncClient) -> str | None:
        """Find the ID of the deployment stamped with this resource's ID.

        Args:
            client: Authenticated Vercel API client.

        Returns:
            The deployment ID, or ``None`` when the project has no such deployment.

        Raises:
            RuntimeError: If more than one deployment carries this resource's ID.
        """
        params = {
            **self._query_params(),
            "projectId": self.config.project_id,
            f"meta-{RESOURCE_ID_META_KEY}": self.id,
            "limit": "100",
        }
        stamped: list[str] = []

        while True:
            response = await client.get("/v7/deployments", params=params)
            await raise_for_status(response)
            page = response.json()

            stamped.extend(
                deployment["uid"] for deployment in page["deployments"] if self.is_live_stamped_deployment(deployment)
            )

            next_page = page["pagination"]["next"]

            if next_page is None:
                break

            params["until"] = str(next_page)

        if len(stamped) > 1:
            msg = f"Found {len(stamped)} Vercel deployments for {self.id}: {', '.join(stamped)}; expected at most one"
            raise RuntimeError(msg)

        return stamped[0] if stamped else None

    async def fetch_deployment(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """Fetch the deployment stamped with this resource's ID.

        Args:
            client: Authenticated Vercel API client.

        Returns:
            Deployment data from the Vercel API, or ``None`` when there is none.
        """
        deployment_id = await self.find_deployment_id(client)

        if deployment_id is None:
            return None

        response = await client.get(f"/v13/deployments/{deployment_id}", params=self._query_params())
        await raise_for_status(response)

        return response.json()

    async def delete_deployment(self, client: httpx.AsyncClient, deployment_id: str) -> None:
        """Delete a deployment; succeeds if Vercel no longer has it.

        Args:
            client: Authenticated Vercel API client.
            deployment_id: ID of the deployment to delete.
        """
        response = await client.delete(f"/v13/deployments/{deployment_id}", params=self._query_params())

        if response.status_code == 404:
            return

        await raise_for_status(response)

    async def on_create(self) -> DeploymentOutputs:
        """Trigger a new deployment and wait until complete.

        Returns:
            DeploymentOutputs with deployment details.
        """
        async with create_vercel_client(self.config.access_token) as client:
            return await self._trigger_deployment(client)

    async def on_observe(self) -> DeploymentOutputs | None:
        """Read the deployment stamped with this resource's ID.

        Returns:
            DeploymentOutputs of that deployment, or ``None`` when there is none.
        """
        async with create_vercel_client(self.config.access_token) as client:
            deployment = await self.fetch_deployment(client)

        if deployment is None:
            return None

        return self._build_outputs(deployment)

    async def on_update(self, previous_config: DeploymentConfig | None) -> DeploymentOutputs:
        """Keep the current deployment if it is up to date, otherwise replace it.

        A current deployment still in flight is awaited until READY rather than replaced.
        Replacement deletes the old deployment first, so nothing is served until the new one is READY.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            DeploymentOutputs of the kept or new deployment.
        """
        async with create_vercel_client(self.config.access_token) as client:
            deployment = await self.fetch_deployment(client)

            if deployment is None:
                return await self._trigger_deployment(client)

            if not self.is_current(deployment):
                await self.delete_deployment(client, deployment["id"])

                return await self._trigger_deployment(client)

            if deployment.get("readyState") in IN_FLIGHT_STATES:
                deployment = await self._wait_for_ready(client, deployment["id"])

            return self._build_outputs(deployment)

    async def on_delete(self) -> None:
        """Delete this resource's deployment; succeeds if there is none."""
        async with create_vercel_client(self.config.access_token) as client:
            deployment_id = await self.find_deployment_id(client)

            if deployment_id is None:
                return

            await self.delete_deployment(client, deployment_id)
