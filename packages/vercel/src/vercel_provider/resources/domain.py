"""Vercel Domain resource."""

from __future__ import annotations

from typing import Literal

from pragma_sdk import Config, Field, HealthStatus, ImmutableField, Outputs, Resource, SensitiveField

from vercel_provider.client import create_vercel_client, fetch_optional_json, raise_for_status


class DomainConfig(Config):
    """Configuration for a custom domain on a Vercel project.

    Attributes:
        access_token: Vercel API token for authentication.
        project_id: Vercel project ID to attach the domain to.
        domain: Custom domain name (e.g., ``example.com``, ``app.example.com``).
        redirect: Optional domain to redirect to. When set, requests to this
            domain are redirected to the target domain.
        redirect_status_code: HTTP status code for the redirect (301, 302, 307
            or 308). Only applies when ``redirect`` is set.
        git_branch: Optional Git branch to associate with this domain.
            When set, deployments from this branch are served on the domain.
        team_id: Vercel team ID. Required for team-owned projects. Immutable.
    """

    access_token: SensitiveField[str]
    project_id: ImmutableField[str]
    domain: ImmutableField[str]
    redirect: Field[str] | None = None
    redirect_status_code: Field[Literal[301, 302, 307, 308]] | None = None
    git_branch: Field[str] | None = None
    team_id: ImmutableField[str] | None = None


class DomainOutputs(Outputs):
    """Outputs from Vercel domain configuration.

    Attributes:
        domain: The configured domain name.
        project_id: Project ID the domain is attached to.
        redirect: Redirect target domain, if configured.
        git_branch: Associated Git branch, if configured.
    """

    domain: str
    project_id: str
    redirect: str
    git_branch: str


class Domain(Resource[DomainConfig, DomainOutputs]):
    """Vercel custom domain resource.

    Adds and manages custom domains on Vercel projects via the REST API.
    Domains can serve project deployments directly or redirect to another
    domain. The domain is located by ``project_id``, ``domain`` and
    ``team_id``, all immutable.

    Lifecycle:
        - on_create: Adds the domain to the project. Verification may be
          required before the domain is active.
        - on_observe: Reads the domain from the project.
        - on_update: Applies the redirect and git branch settings.
        - on_delete: Removes the domain from the project. Succeeds if the
          domain is not attached.

    Example::

        resources:
          - name: my-domain
            provider: vercel
            type: domain
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
              domain: myapp.example.com
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

    def _build_outputs(self, data: dict) -> DomainOutputs:
        """Build outputs from Vercel domain API response data.

        Args:
            data: Raw domain data from the Vercel API.

        Returns:
            DomainOutputs with domain metadata.
        """
        return DomainOutputs(
            domain=data.get("name", self.config.domain),
            project_id=self.config.project_id,
            redirect=data.get("redirect") or "",
            git_branch=data.get("gitBranch") or "",
        )

    def build_settings_body(self) -> dict:
        """Build the redirect and git branch request fields.

        Unset settings are sent as ``null`` so an update clears them.

        Returns:
            Request body fields for adding or updating a project domain.
        """
        return {
            "redirect": self.config.redirect,
            "redirectStatusCode": self.config.redirect_status_code,
            "gitBranch": self.config.git_branch,
        }

    def domain_path(self) -> str:
        """Build the API path of this domain on its project.

        Returns:
            Path of the project domain endpoint.
        """
        return f"/v9/projects/{self.config.project_id}/domains/{self.config.domain}"

    async def on_create(self) -> DomainOutputs:
        """Add a custom domain to the Vercel project.

        Returns:
            DomainOutputs with domain details.
        """
        async with create_vercel_client(self.config.access_token) as client:
            response = await client.post(
                f"/v10/projects/{self.config.project_id}/domains",
                params=self._query_params(),
                json={"name": self.config.domain, **self.build_settings_body()},
            )
            await raise_for_status(response)

            return self._build_outputs(response.json())

    async def on_observe(self) -> DomainOutputs | None:
        """Read the domain from its project.

        Returns:
            DomainOutputs, or ``None`` if the domain is not attached.
        """
        async with create_vercel_client(self.config.access_token) as client:
            domain = await fetch_optional_json(client, self.domain_path(), self._query_params())

        if domain is None:
            return None

        return self._build_outputs(domain)

    async def on_update(self, previous_config: DomainConfig | None) -> DomainOutputs:
        """Apply the redirect and git branch settings to the attached domain.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            DomainOutputs with updated domain details.
        """
        async with create_vercel_client(self.config.access_token) as client:
            response = await client.patch(
                self.domain_path(),
                params=self._query_params(),
                json=self.build_settings_body(),
            )
            await raise_for_status(response)

            return self._build_outputs(response.json())

    async def on_delete(self) -> None:
        """Remove the custom domain from the project.

        Succeeds if the domain is not attached to the project.
        """
        async with create_vercel_client(self.config.access_token) as client:
            response = await client.delete(self.domain_path(), params=self._query_params())

            if response.status_code == 404:
                return

            await raise_for_status(response)

    async def health(self) -> HealthStatus:
        """Report whether Vercel has verified the domain.

        Returns:
            ``healthy`` when verified, ``degraded`` while verification is pending,
            ``unhealthy`` when the domain is not attached.
        """
        async with create_vercel_client(self.config.access_token) as client:
            data = await fetch_optional_json(client, self.domain_path(), self._query_params())

        if data is None:
            return HealthStatus(status="unhealthy", message=f"Domain {self.config.domain} is not attached")

        if data["verified"]:
            return HealthStatus(status="healthy")

        return HealthStatus(
            status="degraded",
            message=f"Domain {self.config.domain} is not verified yet",
            details={"verification": data.get("verification", [])},
        )
