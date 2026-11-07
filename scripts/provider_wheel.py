"""Checks a provider's publishable wheel against the workspace wheels built for one release.

Usage:
    python provider_wheel.py name <pyproject_path>
    python provider_wheel.py release-set <wheel_dir> <distribution>
    python provider_wheel.py report <distribution> <report_path>
    python provider_wheel.py embeds <wheel_dir> <distribution>

Commands:
    name: prints the distribution name a provider's pyproject.toml declares.
    release-set: prints, one per line, the paths of the distribution's wheel and of every wheel in wheel_dir
        that its Requires-Dist reaches, directly or through the chain.
    report: exits 1, printing the report, unless the pragma_sdk.introspection report at report_path succeeded.
    embeds: exits 1 unless the distribution's pragma.embeds keys equal the distributions in its release set,
        itself excluded, whose wheel declares a pragma.provider entry point; it never embeds itself.

A requirement is followed when it has no marker or its marker holds for this interpreter with no extra.
Needs packaging installed.

Exit codes:
    0 when the check passes, 1 when it fails, 2 on a usage error.
"""

import json
import sys
import tomllib
import zipfile
from configparser import ConfigParser
from dataclasses import dataclass
from email.parser import HeaderParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PREFIX = "check_provider_wheel"


@dataclass(frozen=True)
class BuiltWheel:
    """Metadata of one built wheel that the checks read.

    Attributes:
        path: Wheel file path.
        requirements: Canonical names of the requirements followed under the marker rule.
        entry_point_groups: Entry point group name to its entry names.
    """

    path: Path
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


def build_followed_requirements(metadata_requirements: list[str]) -> tuple[str, ...]:
    """Return the canonical names of the requirements the marker rule follows.

    Args:
        metadata_requirements: `Requires-Dist` values.

    Returns:
        Canonical names whose requirement has no marker or a marker that holds with no extra.
    """
    followed = []
    for text in metadata_requirements:
        requirement = Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            followed.append(canonicalize_name(requirement.name))
    return tuple(followed)


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
        wheels[canonicalize_name(metadata["Name"])] = BuiltWheel(
            path=wheel_path,
            requirements=build_followed_requirements(metadata.get_all("Requires-Dist") or []),
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


COMMANDS = {
    "name": (print_distribution_name, 1),
    "release-set": (print_release_set, 2),
    "report": (require_successful_report, 2),
    "embeds": (require_matching_embeds, 2),
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
