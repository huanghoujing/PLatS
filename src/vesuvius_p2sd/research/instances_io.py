"""Instance-volume storage: compressed ``instances_<case>.npz`` files.

A 320^3 int16 label volume is 65.5 MB raw but 0.3-1.5 MB as uint8 inside
``np.savez_compressed`` (labels are sparse and max out far below 255), and
126 stage dirs x 79 cases of raw npy had grown to 610 GB. Writers go through
:func:`save_instances`; readers go through :func:`load_instances` /
:func:`list_instance_files`, which accept BOTH the compressed form and the
legacy ``instances_<case>.npy`` so old dirs and mid-migration dirs keep
working. The npz stores a single array under key ``inst``.
"""
from pathlib import Path

import numpy as np

PREFIX = "instances_"
NPZ_KEY = "inst"


def case_of(path: Path) -> str:
    """``instances_<case>.npy|.npz`` -> ``<case>``."""
    return Path(path).stem.removeprefix(PREFIX)


def instances_file(directory: Path, case_id: str) -> Path | None:
    """The stored file for a case (compressed preferred), or None."""
    directory = Path(directory)
    for suffix in (".npz", ".npy"):
        p = directory / f"{PREFIX}{case_id}{suffix}"
        if p.exists():
            return p
    return None


def list_instance_files(directory: Path) -> list[Path]:
    """Sorted per-case files; when a case has both forms the npz wins."""
    directory = Path(directory)
    by_case: dict[str, Path] = {}
    for p in sorted(directory.glob(f"{PREFIX}*.npy")):
        by_case[case_of(p)] = p
    for p in sorted(directory.glob(f"{PREFIX}*.npz")):
        by_case[case_of(p)] = p
    return [by_case[c] for c in sorted(by_case)]


def load_instances(source: Path, case_id: str | None = None) -> np.ndarray:
    """Load one instance volume from a file path or (directory, case_id)."""
    path = Path(source) if case_id is None else instances_file(source, case_id)
    if path is None or not path.exists():
        raise FileNotFoundError(f"no instances file for {case_id!r} under {source}")
    if path.suffix == ".npz":
        with np.load(path) as data:
            return data[NPZ_KEY]
    return np.load(path)


def save_instances(directory: Path, case_id: str, arr: np.ndarray) -> Path:
    """Write ``instances_<case>.npz`` (uint8 when labels fit, else int16).

    Removes a stale legacy npy of the same case so a dir never holds both.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    dtype = np.uint8 if arr.size == 0 or arr.max() <= 255 else np.int16
    out = directory / f"{PREFIX}{case_id}.npz"
    np.savez_compressed(out, **{NPZ_KEY: np.ascontiguousarray(arr.astype(dtype))})
    legacy = directory / f"{PREFIX}{case_id}.npy"
    legacy.unlink(missing_ok=True)
    return out
