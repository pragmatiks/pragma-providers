"""GCP Cloud SQL user resource."""

from __future__ import annotations

from typing import Any

from pragma_sdk import Config, Field, ImmutableDependency, ImmutableField, Outputs, Resource

from gcp_provider.resources.cloudsql.database_instance import (
    DatabaseInstance,
    DatabaseInstanceConfig,
    resolve_instance_config,
)
from gcp_provider.resources.cloudsql.helpers import (
    execute,
    get_credentials,
    get_sqladmin_service,
    run_instance_operation,
)
from gcp_provider.resources.polling import compute_operation_deadline


class UserConfig(Config):
    """Configuration for a Cloud SQL user.

    Attributes:
        instance: The Cloud SQL instance that hosts this user.
        username: Username for the database user.
        password: Password for the database user. Use Field[str] for secret injection.
    """

    instance: ImmutableDependency[DatabaseInstance]
    username: ImmutableField[str]
    password: Field[str]


class UserOutputs(Outputs):
    """Outputs from Cloud SQL user creation.

    Attributes:
        username: Username of the created user.
        instance_name: Name of the hosting instance.
        host: Host pattern for the user (% for all hosts).
    """

    username: str
    instance_name: str
    host: str


class User(Resource[UserConfig, UserOutputs]):
    """GCP Cloud SQL user resource.

    Creates and manages database users within a Cloud SQL instance. Requires
    a dependency on a ``gcp/cloudsql/database_instance`` resource, from which
    it inherits credentials and connection details.

    Lifecycle:
        - on_create: Creates the user in the target instance. Idempotent --
          succeeds if the user already exists.
        - on_observe: Reads the user by ``username``.
        - on_update: Applies the password in place; ``instance`` and ``username`` are immutable.
        - on_delete: Drops the user from the instance. Idempotent --
          succeeds silently if the user does not exist.

    Example::

        resources:
          - name: app-user
            provider: gcp
            type: cloudsql/user
            config:
              instance:
                provider: gcp
                resource: cloudsql/database_instance
                name: prod-db-instance
              username: app_service
              password:
                provider: pragma
                resource: secret
                name: db-password-secret
                field: outputs.password
    """

    async def on_create(self) -> UserOutputs:
        """Create the user in the Cloud SQL instance and wait until it exists.

        Idempotent: If the user already exists, returns its current state.

        Returns:
            UserOutputs with user details.

        Raises:
            RuntimeError: If Cloud SQL refuses the write because another operation runs on the instance,
                or the operation finished with errors.
            TimeoutError: If a pending operation, or the write's operation, is not done by the operation deadline.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))

        await run_instance_operation(
            service,
            instance_config,
            self.name,
            lambda: execute(
                service.users().insert(
                    project=instance_config.project_id,
                    instance=instance_config.instance_name,
                    body={
                        "name": self.config.username,
                        "password": self.config.password,
                        "project": instance_config.project_id,
                        "instance": instance_config.instance_name,
                    },
                ),
                ignore_exists=True,
            ),
            compute_operation_deadline(),
        )

        return await self.fetch_outputs(instance_config, service)

    async def on_observe(self) -> UserOutputs | None:
        """Read the user named ``username`` in the resolved instance.

        Returns:
            UserOutputs for the user, or None if it does not exist.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))

        user = await self.find_user(instance_config, service)

        if user is None:
            return None

        return self.build_outputs(instance_config, user)

    async def on_update(self, previous_config: UserConfig | None) -> UserOutputs:
        """Apply the configured password when it changed or the previous configuration is unknown.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            UserOutputs with updated user details.

        Raises:
            RuntimeError: If Cloud SQL refuses the write because another operation runs on the instance,
                or the operation finished with errors.
            TimeoutError: If a pending operation, or the write's operation, is not done by the operation deadline.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))

        if previous_config is None or previous_config.password != self.config.password:
            await run_instance_operation(
                service,
                instance_config,
                self.name,
                lambda: execute(
                    service.users().update(
                        project=instance_config.project_id,
                        instance=instance_config.instance_name,
                        name=self.config.username,
                        body={
                            "name": self.config.username,
                            "password": self.config.password,
                        },
                    )
                ),
                compute_operation_deadline(),
            )

        return await self.fetch_outputs(instance_config, service)

    async def on_delete(self) -> None:
        """Drop the user from the host it lives on and wait until it is gone.

        Idempotent: Succeeds if the user does not exist.

        Raises:
            RuntimeError: If users on several hosts share the username, Cloud SQL refuses the write because
                another operation runs on the instance, or the operation finished with errors.
            TimeoutError: If a pending operation, or the write's operation, is not done by the operation deadline.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))
        user = await self.find_user(instance_config, service)

        if user is None:
            return

        await run_instance_operation(
            service,
            instance_config,
            self.name,
            lambda: execute(
                service.users().delete(
                    project=instance_config.project_id,
                    instance=instance_config.instance_name,
                    name=self.config.username,
                    host=user.get("host"),
                ),
                ignore_404=True,
            ),
            compute_operation_deadline(),
        )

    async def fetch_outputs(self, instance_config: DatabaseInstanceConfig, service: Any) -> UserOutputs:
        """Read the user and build its outputs.

        Args:
            instance_config: Configuration of the hosting Cloud SQL instance.
            service: Cloud SQL Admin API service.

        Returns:
            UserOutputs with user details.

        Raises:
            RuntimeError: If the user does not exist.
        """
        user = await self.find_user(instance_config, service)

        if user is None:
            msg = f"User '{self.config.username}' not found"
            raise RuntimeError(msg)

        return self.build_outputs(instance_config, user)

    def build_outputs(self, instance_config: DatabaseInstanceConfig, user: dict) -> UserOutputs:
        """Build outputs from a user dict.

        Args:
            instance_config: Configuration of the hosting Cloud SQL instance.
            user: User as the Cloud SQL Admin API returns it.

        Returns:
            UserOutputs with user details.
        """
        return UserOutputs(
            username=user["name"],
            instance_name=instance_config.instance_name,
            host=user.get("host", "%"),
        )

    async def find_user(self, instance_config: DatabaseInstanceConfig, service: Any) -> dict | None:
        """Find the user in the instance by username.

        Args:
            instance_config: Configuration of the hosting Cloud SQL instance.
            service: Cloud SQL Admin API service.

        Returns:
            User dict if found, None otherwise.

        Raises:
            RuntimeError: If users on several hosts share the username.
        """
        result = await execute(
            service.users().list(
                project=instance_config.project_id,
                instance=instance_config.instance_name,
            ),
            ignore_404=True,
        )

        if result is None:
            return None

        matches = [user for user in result.get("items", []) if user.get("name") == self.config.username]

        if len(matches) > 1:
            msg = (
                f"{len(matches)} users named '{self.config.username}' exist on different hosts; "
                "delete all but one of them in the Cloud SQL console"
            )
            raise RuntimeError(msg)

        return matches[0] if matches else None
