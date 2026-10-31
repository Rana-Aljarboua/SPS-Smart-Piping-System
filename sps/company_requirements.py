from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS_PATH = PROJECT_DIR / "data" / "company" / "requirements.json"


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    title: str
    description: str


@dataclass(frozen=True)
class RequirementGroup:
    group_id: str
    title: str
    requirements: tuple[Requirement, ...]


def _required_text(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty text field {key!r}")
    return value.strip()


def load_requirement_groups(
    path: str | Path = DEFAULT_REQUIREMENTS_PATH,
) -> tuple[RequirementGroup, ...] | None:
    """Load company requirement records; return None only when the file is absent."""
    source = Path(path)
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read requirements data at {source}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise ValueError("Requirements JSON must contain a 'groups' array")

    groups: list[RequirementGroup] = []
    seen_groups: set[str] = set()
    seen_requirements: set[str] = set()
    for group_index, raw_group in enumerate(payload["groups"]):
        context = f"groups[{group_index}]"
        if not isinstance(raw_group, dict):
            raise ValueError(f"{context} must be an object")
        group_id = _required_text(raw_group, "group_id", context)
        if group_id in seen_groups:
            raise ValueError(f"Duplicate requirement group ID: {group_id}")
        seen_groups.add(group_id)
        title = _required_text(raw_group, "title", context)
        raw_requirements = raw_group.get("requirements")
        if not isinstance(raw_requirements, list):
            raise ValueError(f"{context} requires a 'requirements' array")

        requirements: list[Requirement] = []
        for requirement_index, raw_requirement in enumerate(raw_requirements):
            requirement_context = (
                f"{context}.requirements[{requirement_index}]"
            )
            if not isinstance(raw_requirement, dict):
                raise ValueError(f"{requirement_context} must be an object")
            requirement_id = _required_text(
                raw_requirement, "requirement_id", requirement_context
            )
            if requirement_id in seen_requirements:
                raise ValueError(
                    f"Duplicate requirement ID: {requirement_id}"
                )
            seen_requirements.add(requirement_id)
            requirements.append(
                Requirement(
                    requirement_id=requirement_id,
                    title=_required_text(
                        raw_requirement, "title", requirement_context
                    ),
                    description=_required_text(
                        raw_requirement, "description", requirement_context
                    ),
                )
            )
        groups.append(
            RequirementGroup(
                group_id=group_id,
                title=title,
                requirements=tuple(requirements),
            )
        )
    return tuple(groups)
