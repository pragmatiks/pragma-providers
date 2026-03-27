"""Vercel Domain resource."""

from __future__ import annotations

from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField

from vercel_provider.client import create_vercel_client, raise_for_status


class DomainConfig(Config):
    """Configuration for a custom domain on a Vercel project.

    Attributes:
        access_token: Vercel API token for authentication.
        project_id: Vercel project ID to attach the domain to.
        domain: Custom domain name (e.g., ``example.com``, ``app.example.com``).
        redirect: Optional domain to redirect to. When set, requests to this
            domain are redirected to the target domain.
        redirect_status_code: HTTP status code for the redirect (301 or 307).
            Only applies when ``redirect`` is set.
        git_branch: Optional Git branch to associate with this domain.
            When set, deployments from this branch are served on the domain.
        team_id: Vercel team ID. Required for team-owned projects.
    """

    access_token: SensitiveField[str]
    project_id: ImmutableField[str]
    domain: ImmutableField[str]
    redirect: Field[str] | None = None
    redirect_status_code: Field[int] | None = None
    git_branch: Field[str] | None = None
    team_id: Field[str] | None = None


class DomainOutputs(Outputs):
    """Outputs from Vercel domain configuration.

    Attributes:
        domain: The configured domain name.
        project_id: Project ID the domain is attached to.
        verified: Whether the domain has been verified for use.
        redirect: Redirect target domain, if configured.
        git_branch: Associated Git branch, if configured.
    """

    domain: str
    project_id: str
    verified: bool
    redirect: str
    git_branch: str


class Domain(Resource[DomainConfig, DomainOutputs]):
    """Vercel custom domain resource.

    Adds and manages custom domains on Vercel projects via the REST API.
    Domains can serve project deployments directly or redirect to another
    domain.

    Lifecycle:
        - on_create: Adds the domain to the project. Verification may be
          required before the domain is active.
        - on_update: Updates the domain configuration (redirect, git branch).
          Domain name and project are immutable.
        - on_delete: Removes the domain from the project. Idempotent --
          succeeds if the domain is not attached.

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
            verified=data.get("verified", False),
            redirect=data.get("redirect") or "",
            git_branch=data.get("gitBranch") or "",
        )

    async def on_create(self) -> DomainOutputs:
        """Add a custom domain to the Vercel project.

        Returns:
            DomainOutputs with domain details and verification status.
        """
        client = create_vercel_client(self.config.access_token)

        try:
            body: dict = {"name": self.config.domain}

            if self.config.redirect is not None:
                body["redirect"] = self.config.redirect

            if self.config.redirect_status_code is not None:
                body["redirectStatusCode"] = self.config.redirect_status_code

            if self.config.git_branch is not None:
                body["gitBranch"] = self.config.git_branch

            response = await client.post(
                f"/v10/projects/{self.config.project_id}/domains",
                params=self._query_params(),
                json=body,
            )
            await raise_for_status(response)

            return self._build_outputs(response.json())
        finally:
            await client.aclose()

    async def on_update(self, previous_config: DomainConfig) -> DomainOutputs:
        """Update the domain configuration.

        Removes and re-adds the domain to apply changes, since the Vercel
        API does not support PATCH on project domains.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            DomainOutputs with updated domain details.

        Raises:
            RuntimeError: If no existing outputs are available for the domain.
        """
        if self.outputs is None:
            msg = "Cannot update domain without existing outputs"
            raise RuntimeError(msg)

        client = create_vercel_client(self.config.access_token)

        try:
            params = self._query_params()

            delete_response = await client.delete(
                f"/v9/projects/{self.config.project_id}/domains/{self.config.domain}",
                params=params,
            )

            if delete_response.status_code != 404:
                await raise_for_status(delete_response)

            body: dict = {"name": self.config.domain}

            if self.config.redirect is not None:
                body["redirect"] = self.config.redirect

            if self.config.redirect_status_code is not None:
                body["redirectStatusCode"] = self.config.redirect_status_code

            if self.config.git_branch is not None:
                body["gitBranch"] = self.config.git_branch

            create_response = await client.post(
                f"/v10/projects/{self.config.project_id}/domains",
                params=params,
                json=body,
            )
            await raise_for_status(create_response)

            return self._build_outputs(create_response.json())
        finally:
            await client.aclose()

    async def on_delete(self) -> None:
        """Remove the custom domain from the project.

        Idempotent: Succeeds if the domain is not attached to the project.
        """
        if self.outputs is None:
            return

        client = create_vercel_client(self.config.access_token)

        try:
            response = await client.delete(
                f"/v9/projects/{self.config.project_id}/domains/{self.config.domain}",
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
