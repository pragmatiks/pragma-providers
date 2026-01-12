# gcp Provider

GCP provider for Pragmatiks

## Resources

Define your resources in `src/gcp_provider/resources.py`:

```python
from pragma_sdk import Resource, Config, Outputs, Field
from typing import ClassVar

class MyResourceConfig(Config):
    name: Field[str]
    size: Field[int] = 10

class MyResourceOutputs(Outputs):
    url: str
    created_at: str

class MyResource(Resource[MyResourceConfig, MyResourceOutputs]):
    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "my_resource"

    async def on_create(self) -> MyResourceOutputs:
        # Create the resource
        return MyResourceOutputs(url="...", created_at="...")

    async def on_update(self, previous_config: MyResourceConfig) -> MyResourceOutputs:
        # Update the resource
        return self.outputs

    async def on_delete(self) -> None:
        # Delete the resource
        pass
```

## Development

### Testing

Test your resource lifecycle methods with ProviderHarness:

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest tests/
```

### Writing Tests

Use `ProviderHarness` to test lifecycle methods:

```python
from pragma_sdk.provider import ProviderHarness
from gcp_provider import ExampleResource, ExampleConfig

async def test_create():
    harness = ProviderHarness()
    result = await harness.invoke_create(
        ExampleResource,
        name="test",
        config=ExampleConfig(name="my-resource", size=10),
    )
    assert result.success
    assert result.outputs.url is not None
```

## Deployment

Push your provider to Pragmatiks platform:

```bash
pragma provider push
```

The platform handles all runtime infrastructure - you just push your code.

## Updating This Project

This project was created with [Copier](https://copier.readthedocs.io/). To update to the latest template:

```bash
copier update
```
