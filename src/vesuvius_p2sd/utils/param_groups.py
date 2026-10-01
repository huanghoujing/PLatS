"""YAML-driven optimizer parameter grouping."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Iterable


@dataclass(frozen=True)
class ParamGroupSummary:
    name: str
    selector: str
    lr: float
    count: int
    requires_grad_count: int
    excluded_by_zero_lr: bool


def build_param_groups(
    named_parameters: Iterable[tuple[str, Any]],
    group_specs: list[dict[str, Any]],
    *,
    allow_unmatched: bool = False,
) -> tuple[list[dict[str, Any]], list[ParamGroupSummary]]:
    """Build optimizer parameter groups from ordered YAML specs.

    A group with ``lr: 0`` disables its parameters and is excluded from
    optimizer groups. A nonzero group retains the parameter's existing
    ``requires_grad`` state; it does not re-enable a model-frozen parameter.
    Non-default groups must match at least one parameter unless
    ``allow_unmatched`` is true.
    """

    params = list(named_parameters)
    if not group_specs:
        raise ValueError("At least one optimizer group spec is required")

    default_specs = [g for g in group_specs if g.get("name") == "default"]
    if len(default_specs) != 1:
        raise ValueError("Exactly one optimizer group named 'default' is required")
    default_spec = default_specs[0]

    assigned: set[str] = set()
    optimizer_groups: list[dict[str, Any]] = []
    summaries: list[ParamGroupSummary] = []

    for spec in group_specs:
        if spec.get("name") == "default":
            continue
        matched = _match_unassigned(params, spec.get("selector", ""), assigned)
        if not matched and not allow_unmatched:
            raise ValueError(
                f"Optimizer group {spec.get('name')!r} matched no parameters")
        _apply_group(spec, matched, optimizer_groups, summaries)
        assigned.update(name for name, _ in matched)

    default_matched = [(name, p) for name, p in params if name not in assigned]
    _apply_group(default_spec, default_matched, optimizer_groups, summaries)
    return optimizer_groups, summaries


def _match_unassigned(
    params: list[tuple[str, Any]],
    selector: str,
    assigned: set[str],
) -> list[tuple[str, Any]]:
    if not selector:
        raise ValueError("Optimizer group selector cannot be empty")
    out = []
    for name, param in params:
        if name in assigned:
            continue
        if fnmatch(name, selector):
            out.append((name, param))
    return out


def _apply_group(
    spec: dict[str, Any],
    matched: list[tuple[str, Any]],
    optimizer_groups: list[dict[str, Any]],
    summaries: list[ParamGroupSummary],
) -> None:
    name = str(spec.get("name", ""))
    selector = str(spec.get("selector", ""))
    lr = float(spec.get("lr", 0.0))
    excluded_by_zero_lr = lr == 0.0
    if excluded_by_zero_lr:
        for _, param in matched:
            _set_requires_grad(param, False)
    else:
        group = {
            "name": name,
            "params": [param for _, param in matched],
            "lr": lr,
        }
        if "weight_decay" in spec:
            group["weight_decay"] = float(spec["weight_decay"])
        optimizer_groups.append(group)
    summaries.append(
        ParamGroupSummary(
            name=name,
            selector=selector,
            lr=lr,
            count=len(matched),
            requires_grad_count=sum(
                bool(getattr(param, "requires_grad", False))
                for _, param in matched
            ),
            excluded_by_zero_lr=excluded_by_zero_lr,
        ))


def _set_requires_grad(param: Any, value: bool) -> None:
    if hasattr(param, "requires_grad_"):
        param.requires_grad_(value)
    elif hasattr(param, "requires_grad"):
        param.requires_grad = value
