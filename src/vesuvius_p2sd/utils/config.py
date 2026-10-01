"""YAML config loading with simple dotlist overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    base = data.pop("_base_", None)
    if base is None:
        return data
    base_path = resolve_base_path(path, base)
    merged = load_yaml(base_path)
    deep_update(merged, data)
    return merged


def resolve_base_path(path: Path, base: str | Path) -> Path:
    base_path = Path(base)
    if base_path.is_absolute():
        return base_path
    candidates = [
        path.parent / base_path,
        Path.cwd() / base_path,
    ]
    for parent in path.parents:
        candidates.append(parent / base_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve _base_={base!r} from {path}. "
        f"Tried: {[str(c) for c in candidates]}")


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    cfg = load_yaml(path)
    for override in overrides or []:
        apply_override(cfg, override)
    return cfg


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        # A child config occasionally needs to replace a nested mapping rather
        # than inherit keys that are invalid for a different implementation
        # choice (for example broadcast-only prompt-modulator settings).
        if isinstance(value, dict) and value.get("_replace_") is True:
            dst[key] = copy.deepcopy({
                child_key: child_value
                for child_key, child_value in value.items()
                if child_key != "_replace_"
            })
            continue
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def apply_override(cfg: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value, got {override!r}")
    key, raw_value = override.split("=", 1)
    if not key:
        raise ValueError(f"Override key is empty: {override!r}")
    value = parse_override_value(raw_value)
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if not part:
            raise ValueError(f"Invalid override key: {key!r}")
        nxt = cur.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"Cannot set nested key through non-mapping: {key!r}")
        cur = nxt
    cur[parts[-1]] = value


def parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def get_run_dir(cfg: dict[str, Any]) -> Path:
    run = cfg.get("run", {})
    output_root = Path(run.get("output_root", "runs"))
    run_id = run.get("run_id")
    if not run_id:
        raise ValueError("Config must define run.run_id")
    return output_root / str(run_id)
