"""Checks a provider's publishable wheel against the workspace wheels built for one release.

Usage:
    python provider_wheel.py name <pyproject_path>
    python provider_wheel.py release-set <wheel_dir> <distribution>
    python provider_wheel.py report <distribution> <report_path>
    python provider_wheel.py embeds <wheel_dir> <distribution>
    python provider_wheel.py floor-tags <workspace_dir> <distribution>
    python provider_wheel.py unreleased-floors <wheel_dir>

Commands:
    name: prints the distribution name a provider's pyproject.toml declares.
    release-set: prints, one per line, the paths of the distribution's wheel and of every wheel in wheel_dir
        that its Requires-Dist reaches, directly or through the chain.
    report: exits 1, printing the report, unless the pragma_sdk.introspection report at report_path succeeded.
    embeds: exits 1 unless the distribution's pragma.embeds keys equal the distributions in its release set,
        itself excluded, whose wheel declares a pragma.provider entry point; it never embeds itself.
    floor-tags: prints one record per line for every workspace sibling the
        distribution reaches at its lowest allowed version, as `release <sibling> <tag>` lines: each sibling
        requirement resolves to the lowest tag whose version satisfies it, read again at that tag for the
        sibling's own sibling requirements, and a sibling reached twice keeps the higher tag. A requirement
        no tag satisfies is a floor not released yet: it prints a GitHub Actions warning to stderr and a
        `tree <sibling>` line, whose in-tree package stands in and is read in the working tree for its own
        sibling requirements.

    unreleased-floors: prints, one per line, a uv override `<name> @ file://<wheel>` for every wheel in
        wheel_dir that some wheel there requires with a floor above its version, and a GitHub Actions
        warning to stderr for each such requirement. A floor above the in-tree version is not released
        yet: the in-tree wheel is the sibling's last released one until CI bumps it, so it stands in.
        A requirement that fails on any other clause gets no override and stays strict.

A requirement is followed when it has no marker or its marker holds for this interpreter with no extra.
Needs packaging installed; floor-tags also needs git and every `<provider>-v<version>` tag fetched.

Exit codes:
    0 when the check passes, 1 when it fails, 2 on a usage error.
"""

import json
import subprocess
import sys
import tomllib
import zipfile
from configparser import ConfigParser
from dataclasses import dataclass
from email.parser import HeaderParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


PREFIX = "check_provider_wheel"
DISTRIBUTION_PREFIX = "pragmatiks-"


@dataclass(frozen=True)
class BuiltWheel:
    """Metadata of one built wheel that the checks read.

    Attributes:
        path: Wheel file path.
        version: The wheel's version.
        followed_requirements: The requirements followed under the marker rule.
        requirements: Canonical names of the requirements followed under the marker rule.
        entry_point_groups: Entry point group name to its entry names.
    """

    path: Path
    version: Version
    followed_requirements: tuple[Requirement, ...]
    requirements: tuple[str, ...]
    entry_point_groups: dict[str, set[str]]


def read_dist_info_member(archive: zipfile.ZipFile, member_name: str) -> str | None:
    """Read a `.dist-info` member of a wheel.

    Args:
        archive: The open wheel.
        member_name: File name inside `.dist-info`, such as `METADATA`.

    Returns:
        The member's text, or None when the wheel has no such member.
    """
    for name in archive.namelist():
        if name.endswith(f".dist-info/{member_name}") and name.count("/") == 1:
            return archive.read(name).decode()
    return None


def build_followed_requirements(metadata_requirements: list[str]) -> tuple[Requirement, ...]:
    """Return the requirements the marker rule follows.

    Args:
        metadata_requirements: `Requires-Dist` values.

    Returns:
        The requirements with no marker or a marker that holds with no extra.
    """
    requirements = (Requirement(text) for text in metadata_requirements)
    return tuple(
        requirement
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""})
    )


def load_built_wheels(wheel_directory: str) -> dict[str, BuiltWheel]:
    """Read every wheel in a directory.

    Args:
        wheel_directory: Directory of built wheels.

    Returns:
        Each wheel keyed by its canonical distribution name.
    """
    wheels = {}
    for wheel_path in Path(wheel_directory).glob("*.whl"):
        with zipfile.ZipFile(wheel_path) as archive:
            metadata = HeaderParser().parsestr(read_dist_info_member(archive, "METADATA") or "")
            entry_points = ConfigParser(delimiters=("=",), interpolation=None)
            entry_points.optionxform = str
            entry_points.read_string(read_dist_info_member(archive, "entry_points.txt") or "")
        followed_requirements = build_followed_requirements(metadata.get_all("Requires-Dist") or [])
        wheels[canonicalize_name(metadata["Name"])] = BuiltWheel(
            path=wheel_path,
            version=Version(metadata["Version"]),
            followed_requirements=followed_requirements,
            requirements=tuple(canonicalize_name(requirement.name) for requirement in followed_requirements),
            entry_point_groups={group: set(entry_points[group]) for group in entry_points.sections()},
        )
    return wheels


def compute_release_set(wheels: dict[str, BuiltWheel], served: str) -> list[str]:
    """Return the served distribution and every built wheel its requirements reach.

    Args:
        wheels: Built wheels keyed by canonical name.
        served: Canonical name of the provider to start from.

    Returns:
        Canonical names in the release set, the served one first.
    """
    reached: list[str] = []
    pending = [served]
    while pending:
        name = pending.pop()
        if name in wheels and name not in reached:
            reached.append(name)
            pending.extend(wheels[name].requirements)
    return reached


def print_distribution_name(pyproject_path: str) -> None:
    """Print the `[project] name` of a pyproject.toml.

    Args:
        pyproject_path: Path to the provider's pyproject.toml.
    """
    with Path(pyproject_path).open("rb") as pyproject_file:
        print(tomllib.load(pyproject_file)["project"]["name"])


def print_release_set(wheel_directory: str, served_name: str) -> None:
    """Print the wheel paths of the distribution's release set.

    Args:
        wheel_directory: Directory of built wheels.
        served_name: Distribution name of the provider.
    """
    wheels = load_built_wheels(wheel_directory)
    for name in compute_release_set(wheels, canonicalize_name(served_name)):
        print(wheels[name].path)


def require_successful_report(served: str, report_path: str) -> None:
    """Exit with a failure unless the introspection report succeeded.

    Args:
        served: Distribution name, for the failure message.
        report_path: Path to the report pragma_sdk.introspection wrote.
    """
    report_text = Path(report_path).read_text()
    if json.loads(report_text)["outcome"] != "success":
        sys.exit(f"{PREFIX}: {served}: introspection failed:\n{report_text}")


def require_matching_embeds(wheel_directory: str, served_name: str) -> None:
    """Exit with a failure unless the pragma.embeds keys equal the providers in the release set.

    Args:
        wheel_directory: Directory of built wheels.
        served_name: Distribution name of the provider.
    """
    wheels = load_built_wheels(wheel_directory)
    served = canonicalize_name(served_name)
    declared = {canonicalize_name(name) for name in wheels[served].entry_point_groups.get("pragma.embeds", set())}
    expected = {
        name
        for name in compute_release_set(wheels, served)
        if name != served and "pragma.provider" in wheels[name].entry_point_groups
    }
    if served in declared or declared != expected:
        sys.exit(
            f"{PREFIX}: {served}: pragma.embeds declares {sorted(declared)}, "
            f"but the wheel reaches providers {sorted(expected)} and must not embed itself"
        )
    print(f"{PREFIX}: {served}: introspection and embeds pass")


def is_unreleased_floor(requirement: Requirement, version: Version) -> bool:
    """Tell whether a requirement refuses a version only through a floor above it.

    Args:
        requirement: The requirement on a sibling.
        version: The sibling's in-tree version.

    Returns:
        True when the version fails the requirement and every clause it fails is a `>=` or `>` floor.
    """
    failing = [clause for clause in requirement.specifier if not clause.contains(version, prereleases=True)]
    return bool(failing) and all(clause.operator in (">=", ">") for clause in failing)


def print_unreleased_floor_overrides(wheel_directory: str) -> None:
    """Print a uv override for every wheel some wheel in the directory requires above its version.

    Args:
        wheel_directory: Directory of built wheels.
    """
    wheels = load_built_wheels(wheel_directory)
    overridden: set[str] = set()
    for dependent, wheel in sorted(wheels.items()):
        for requirement in wheel.followed_requirements:
            name = canonicalize_name(requirement.name)
            if name not in wheels or not is_unreleased_floor(requirement, wheels[name].version):
                continue
            print(
                f"::warning::{dependent} requires {requirement}; in-tree {wheels[name].version} is not "
                "released yet, so its in-tree wheel stands in",
                file=sys.stderr,
            )
            overridden.add(name)
    for name in sorted(overridden):
        print(f"{name} @ {wheels[name].path.resolve().as_uri()}")


def read_git_output(workspace_directory: str, arguments: list[str]) -> str:
    """Run a git command in the workspace and return its standard output.

    Args:
        workspace_directory: Repository root.
        arguments: git arguments.

    Returns:
        The command's standard output.
    """
    return subprocess.run(
        ["git", "-C", workspace_directory, *arguments], check=True, capture_output=True, text=True
    ).stdout


def read_project_table(workspace_directory: str, provider: str, revision: str | None) -> dict:
    """Read a provider's `[project]` table.

    Args:
        workspace_directory: Repository root.
        provider: Provider package directory name, such as `kubernetes`.
        revision: Git revision to read the pyproject.toml at, or None for the working tree.

    Returns:
        The `[project]` table.
    """
    pyproject_path = f"packages/{provider}/pyproject.toml"
    if revision is None:
        text = (Path(workspace_directory) / pyproject_path).read_text()
    else:
        text = read_git_output(workspace_directory, ["show", f"{revision}:{pyproject_path}"])
    return tomllib.loads(text)["project"]


def load_release_versions(workspace_directory: str, provider: str) -> list[Version]:
    """Read the versions a provider released under its current distribution name, lowest first.

    Args:
        workspace_directory: Repository root.
        provider: Provider package directory name, such as `kubernetes`.

    Returns:
        The version of every `<provider>-v<version>` tag whose package carries the distribution name
        `pragmatiks-<provider>`, sorted ascending; tags from before a rename are left out.
    """
    distribution = f"{DISTRIBUTION_PREFIX}{provider}"
    tags = read_git_output(workspace_directory, ["tag", "--list", f"{provider}-v*"]).split()
    return sorted(
        Version(tag.removeprefix(f"{provider}-v"))
        for tag in tags
        if canonicalize_name(read_project_table(workspace_directory, provider, tag)["name"]) == distribution
    )


def is_workspace_sibling(workspace_directory: str, requirement_name: str) -> bool:
    """Tell whether a requirement names a provider under packages/.

    Args:
        workspace_directory: Repository root.
        requirement_name: The requirement's distribution name.

    Returns:
        True when the name is `pragmatiks-<provider>` and packages/<provider> exists.
    """
    name = canonicalize_name(requirement_name)
    if not name.startswith(DISTRIBUTION_PREFIX):
        return False
    return (Path(workspace_directory) / "packages" / name.removeprefix(DISTRIBUTION_PREFIX)).is_dir()


def read_sibling_requirements(workspace_directory: str, provider: str, revision: str | None) -> list[Requirement]:
    """Read a provider's requirements on workspace siblings.

    Args:
        workspace_directory: Repository root.
        provider: Provider package directory name.
        revision: Git revision to read the pyproject.toml at, or None for the working tree.

    Returns:
        The `[project] dependencies` naming a provider under packages/, marker rule applied.
    """
    project = read_project_table(workspace_directory, provider, revision)
    requirements = [Requirement(item) for item in project.get("dependencies", [])]
    return [
        requirement
        for requirement in requirements
        if is_workspace_sibling(workspace_directory, requirement.name)
        and (requirement.marker is None or requirement.marker.evaluate({"extra": ""}))
    ]


def load_lowest_release(workspace_directory: str, sibling: str, requirement: Requirement) -> Version | None:
    """Read the lowest release of a sibling that satisfies a requirement.

    Args:
        workspace_directory: Repository root.
        sibling: Sibling package directory name.
        requirement: The requirement on that sibling.

    Returns:
        The lowest satisfying release version, or None when no release satisfies it.
    """
    for version in load_release_versions(workspace_directory, sibling):
        if requirement.specifier.contains(version, prereleases=True):
            return version
    return None


def load_floor_versions(workspace_directory: str, provider: str) -> dict[str, Version | None]:
    """Resolve every sibling a provider reaches to its lowest released version allowed.

    Args:
        workspace_directory: Repository root.
        provider: Provider package directory name to start from, read in the working tree.

    Returns:
        Sibling package directory name to the version of its lowest satisfying release tag, or None when
        some requirement on it no release satisfies; None wins over any release, since only the in-tree
        package satisfies that requirement, and that in-tree package's own sibling requirements are
        followed in the working tree.
    """
    floors: dict[str, Version | None] = {}
    pending: list[tuple[str, str | None]] = [(provider, None)]
    while pending:
        dependent, revision = pending.pop()
        for requirement in read_sibling_requirements(workspace_directory, dependent, revision):
            sibling = canonicalize_name(requirement.name).removeprefix(DISTRIBUTION_PREFIX)
            lowest = load_lowest_release(workspace_directory, sibling, requirement)
            if lowest is None:
                print(
                    f"::warning::{DISTRIBUTION_PREFIX}{dependent} requires {requirement}; no {sibling} release "
                    "satisfies it yet, so the in-tree wheel stands in",
                    file=sys.stderr,
                )
                if sibling not in floors or floors[sibling] is not None:
                    pending.append((sibling, None))
                floors[sibling] = None
                continue
            if sibling in floors and (floors[sibling] is None or floors[sibling] >= lowest):
                continue
            floors[sibling] = lowest
            pending.append((sibling, f"{sibling}-v{lowest}"))
    return floors


def print_floor_tags(workspace_directory: str, served_name: str) -> None:
    """Print one record per sibling the distribution reaches at its lowest allowed version.

    Each line is `release <sibling> <tag>` for a release tag, or `tree <sibling>` when the in-tree
    package stands in.

    Args:
        workspace_directory: Repository root.
        served_name: Distribution name of the provider.
    """
    provider = canonicalize_name(served_name).removeprefix(DISTRIBUTION_PREFIX)
    for sibling, version in sorted(load_floor_versions(workspace_directory, provider).items()):
        print(f"tree {sibling}" if version is None else f"release {sibling} {sibling}-v{version}")


COMMANDS = {
    "name": (print_distribution_name, 1),
    "release-set": (print_release_set, 2),
    "report": (require_successful_report, 2),
    "embeds": (require_matching_embeds, 2),
    "floor-tags": (print_floor_tags, 2),
    "unreleased-floors": (print_unreleased_floor_overrides, 1),
}


def main(arguments: list[str]) -> None:
    """Run the command the arguments name.

    Args:
        arguments: Command name followed by its arguments.
    """
    handler, argument_count = COMMANDS.get(arguments[0], (None, -1)) if arguments else (None, -1)
    if handler is None or len(arguments) - 1 != argument_count:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    handler(*arguments[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
