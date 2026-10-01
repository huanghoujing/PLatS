"""Run status and resume-context helpers."""

from __future__ import annotations

import argparse
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_run_dir(run_dir: str | Path, *, command: list[str] | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "state.json", {
        "status": "initialized",
        "created_at": utc_now(),
    })
    write_json(run_dir / "heartbeat.json", {
        "time": utc_now(),
        "status": "initialized",
    })
    if command is not None:
        with (run_dir / "command.sh").open("w", encoding="utf-8") as f:
            f.write(" ".join(command) + "\n")
    return run_dir


def update_heartbeat(run_dir: str | Path, **fields: Any) -> None:
    payload = {"time": utc_now(), **fields}
    write_json(Path(run_dir) / "heartbeat.json", payload)


def write_experiment_provenance(
    run_dir: str | Path,
    *,
    argv: list[str],
    purpose: str | None,
    parents: dict[str, str] | None = None,
) -> None:
    """command.sh + lineage.json for RESEARCH experiment dirs.

    Trainer runs record their origin (command.sh, lineage.json with a
    ``reason``); research CLIs historically wrote neither, leaving pipeline
    experiment dirs with no in-dir record of what generated them or why
    (user-caught 2026-08-17). ``parents`` maps role -> input path; its first
    entry becomes lineage ``parent.run`` so ``run_index`` chains experiment
    dirs into the training tree. Never overwrites an existing lineage.json
    (re-touch tools like --backfill_nifti must not falsify origins).
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.sh").write_text(" ".join(argv) + "\n", encoding="utf-8")
    if (run_dir / "lineage.json").exists():
        return
    parents = {k: str(v) for k, v in (parents or {}).items() if v}
    primary = next(iter(parents.values()), None)
    git_commit = None
    try:
        import subprocess
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass
    write_json(run_dir / "lineage.json", {
        "created_at": utc_now(),
        "reason": purpose or None,
        "parent": {"run": Path(primary).name} if primary else None,
        "inputs": parents,
        "git_commit": git_commit,
    })


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state = read_json(run_dir / "state.json", {})
    heartbeat = read_json(run_dir / "heartbeat.json", {})
    best = read_json(run_dir / "best.json", {})
    pid = _read_pid(run_dir / "pid.txt")
    return {
        "run_dir": str(run_dir),
        "state": state,
        "heartbeat": heartbeat,
        "best": best,
        "pid": pid,
        "pid_alive": _pid_alive(pid) if pid is not None else False,
        "stop_requested": (run_dir / "stop_requested").exists(),
        "last_error": _read_text(run_dir / "last_error.txt"),
        "recent_metrics": _tail_jsonl(run_dir / "metrics.jsonl", n=5),
        "pngs": [str(p) for p in sorted((run_dir / "viz").glob("**/*.png"))[-8:]],
    }


def write_resume_context(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    summary = summarize_run(run_dir)
    path = run_dir / "resume_context.md"
    lines = [
        f"# Resume Context: {run_dir.name}",
        "",
        f"- status: {summary['state'].get('status', 'unknown')}",
        f"- pid: {summary['pid']}",
        f"- pid_alive: {summary['pid_alive']}",
        f"- stop_requested: {summary['stop_requested']}",
        "",
        "## Best",
        "",
        "```json",
        json.dumps(summary["best"], indent=2, sort_keys=True),
        "```",
        "",
        "## Heartbeat",
        "",
        "```json",
        json.dumps(summary["heartbeat"], indent=2, sort_keys=True),
        "```",
        "",
        "## Recent Metrics",
        "",
        "```json",
        json.dumps(summary["recent_metrics"], indent=2, sort_keys=True),
        "```",
        "",
        "## Recent PNGs",
        "",
    ]
    lines.extend(f"- {p}" for p in summary["pngs"])
    if summary["last_error"]:
        lines.extend(["", "## Last Error", "", "```text", summary["last_error"], "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    handoff = run_dir / "agent_handoff.md"
    if not handoff.exists():
        handoff.write_text(
            f"# Agent Handoff: {run_dir.name}\n\n"
            "## Current Hypothesis\n\n"
            "## Latest State\n\n"
            f"See `{path.name}`.\n\n"
            "## Recommended Next Action\n\n",
            encoding="utf-8",
        )
    return path


def request_stop(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "stop_requested"
    path.write_text(utc_now() + "\n", encoding="utf-8")
    return path


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return int(text) if text else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _tail_jsonl(path: Path, *, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"raw": line})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "resume_context", "request_stop"])
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    if args.action == "status":
        print(json.dumps(summarize_run(args.run_dir), indent=2, sort_keys=True))
    elif args.action == "resume_context":
        print(write_resume_context(args.run_dir))
    elif args.action == "request_stop":
        print(request_stop(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
