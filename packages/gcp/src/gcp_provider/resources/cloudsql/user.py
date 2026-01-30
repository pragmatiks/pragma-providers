"""GCP Cloud SQL user resource."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field as PydanticField
from pragma_sdk import Config, Dependency, Field, Outputs, Resource

from gcp_provider.resources.cloudsql.database_instance import DatabaseInstance
from gcp_provider.resources.cloudsql.helpers import execute, get_credentials, get_sqladmin_service


class UserConfig(Config):
    """Configuration for a Cloud SQL user.

    Attributes:
        instance: The Cloud SQL instance that hosts this user.
        username: Username for the database user.
        password: Password for the database user. Use Field[str] for secret injection.
    """

    instance: Dependency[DatabaseInstance]
    username: str = PydanticField(json_schema_extra={"immutable": True})
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

    Creates and manages database users within a Cloud SQL instance.
    Password is mutable. Changing the instance triggers replacement.
    """

    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "cloudsql/user"

    async def on_create(self) -> UserOutputs:
        """Create user in the Cloud SQL instance.

        Idempotent: If user already exists, returns its current state.
        """
        inst = self.config.instance.config
        service = get_sqladmin_service(get_credentials(inst.credentials))

        await execute(
            service.users().insert(
                project=inst.project_id,
                instance=inst.instance_name,
                body={
                    "name": self.config.username,
                    "password": self.config.password,
                    "project": inst.project_id,
                    "instance": inst.instance_name,
                },
            ),
            ignore_exists=True,
        )

        user = await self._find_user(inst, service)

        return UserOutputs(
            username=user.get("name", self.config.username) if user else self.config.username,
            instance_name=inst.instance_name,
            host=user.get("host", "%") if user else "%",
        )

    async def on_update(self, previous_config: UserConfig) -> UserOutputs:
        """Handle user updates.

        If instance changed, delete from old instance and create in new one.
        Password changes are applied in place. Username cannot be changed.
        """
        if previous_config.username != self.config.username:
            msg = "Cannot change username; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.instance != self.config.instance:
            await self._delete(previous_config)
            return await self.on_create()

        inst = self.config.instance.config
        service = get_sqladmin_service(get_credentials(inst.credentials))

        if previous_config.password != self.config.password:
            await execute(
                service.users().update(
                    project=inst.project_id,
                    instance=inst.instance_name,
                    name=self.config.username,
                    body={
                        "name": self.config.username,
                        "password": self.config.password,
                    },
                )
            )

        user = await self._find_user(inst, service)

        if user is None:
            msg = f"User '{self.config.username}' not found"
            raise RuntimeError(msg)

        return UserOutputs(
            username=user.get("name", self.config.username),
            instance_name=inst.instance_name,
            host=user.get("host", "%"),
        )

    async def on_delete(self) -> None:
        """Delete user. Idempotent: succeeds if user doesn't exist."""
        await self._delete(self.config)

    async def _delete(self, config: UserConfig) -> None:
        """Delete user from instance. Idempotent: succeeds if not found."""
        inst = config.instance.config
        service = get_sqladmin_service(get_credentials(inst.credentials))

        await execute(
            service.users().delete(
                project=inst.project_id,
                instance=inst.instance_name,
                name=config.username,
            ),
            ignore_404=True,
        )

    async def _find_user(self, inst: Any, service: Any) -> dict | None:
        """Find user in instance by username."""
        result = await execute(
            service.users().list(
                project=inst.project_id,
                instance=inst.instance_name,
            ),
            ignore_404=True,
        )

        if result is None:
            return None

        for user in result.get("items", []):
            if user.get("name") == self.config.username:
                return user

        return None
