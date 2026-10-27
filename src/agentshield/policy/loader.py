"""Load and validate policy files."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from agentshield.policy.models import Policy


class PolicyError(ValueError):
    """The policy file is unreadable or invalid. The message says what to fix."""


def parse_policy(text: str, source: str = "<string>") -> Policy:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"{source}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyError(f"{source}: top level must be a mapping, got {type(raw).__name__}")
    try:
        return Policy.model_validate(raw)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            where = ".".join(str(part) for part in err["loc"]) or "<root>"
            lines.append(f"  - {where}: {err['msg']}")
        raise PolicyError(f"{source}: invalid policy\n" + "\n".join(lines)) from exc


def load_policy(path: str | Path) -> Policy:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"cannot read policy file {p}: {exc}") from exc
    return parse_policy(text, source=str(p))
