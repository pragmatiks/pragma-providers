"""Agno prompt resource for reusable instruction templates."""

from __future__ import annotations

import re
from typing import ClassVar

from pragma_sdk import Config, Outputs, Resource
from pydantic import model_validator


class PromptConfig(Config):
    """Configuration for a reusable prompt template.

    Attributes:
        instructions: List of instruction lines to join with newlines.
        variables: Variable values for template interpolation.
        template: Template string with {{variable}} placeholders.
    """

    instructions: list[str] = []
    variables: dict[str, str] = {}
    template: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> PromptConfig:
        """Validate that content is provided and variables are complete.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If neither instructions nor template provided,
                or if template references undefined variables.
        """
        if not self.instructions and not self.template:
            msg = "At least one of 'instructions' or 'template' must be provided"
            raise ValueError(msg)

        if self.template:
            placeholders = set(re.findall(r"\{\{(\w+)\}\}", self.template))
            missing = placeholders - set(self.variables.keys())
            if missing:
                msg = f"Missing variables for template placeholders: {sorted(missing)}"
                raise ValueError(msg)

        return self


class PromptOutputs(Outputs):
    """Outputs from prompt resource.

    Attributes:
        text: Rendered prompt text.
        instruction_count: Number of instruction lines in the rendered text.
    """

    text: str
    instruction_count: int


class Prompt(Resource[PromptConfig, PromptOutputs]):
    """Reusable prompt template resource.

    Stateless resource (Pragmatiks addition, not in Agno SDK) that enables
    prompt reuse and versioning. Dependent resources can call render() to
    get the formatted prompt text.

    Template syntax:
        Use {{variable}} for interpolation. Variables must be defined
        in the 'variables' dict.

    Rendering:
        - If only instructions: join with newlines
        - If only template: interpolate variables
        - If both: interpolate template, append to instructions

    Lifecycle:
        - on_create: Render and return outputs
        - on_update: Re-render with new config
        - on_delete: No-op (stateless)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "prompt"

    def render(self) -> str:
        """Render the prompt text.

        Called by dependent resources (e.g., agno/agent) that need
        the formatted prompt for their instructions.

        Returns:
            Rendered prompt text with variables interpolated.
        """
        parts: list[str] = []

        if self.config.instructions:
            parts.append("\n".join(self.config.instructions))

        if self.config.template:
            rendered = self.config.template
            for key, value in self.config.variables.items():
                rendered = rendered.replace(f"{{{{{key}}}}}", value)
            parts.append(rendered)

        return "\n".join(parts)

    def _build_outputs(self) -> PromptOutputs:
        """Build outputs from current config.

        Returns:
            PromptOutputs with rendered text and instruction count.
        """
        text = self.render()
        instruction_count = len(text.strip().split("\n")) if text.strip() else 0

        return PromptOutputs(text=text, instruction_count=instruction_count)

    async def on_create(self) -> PromptOutputs:
        """Create resource and return rendered outputs.

        Returns:
            PromptOutputs with rendered text and instruction count.
        """
        return self._build_outputs()

    async def on_update(self, previous_config: PromptConfig) -> PromptOutputs:  # noqa: ARG002
        """Update resource and return re-rendered outputs.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            PromptOutputs with updated rendered text.
        """
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
