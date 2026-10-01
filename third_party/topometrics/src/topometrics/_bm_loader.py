import importlib
import sys
from pathlib import Path

def load_betti_matching():
    # Look for the compiled module in the submodule build dir.
    here = Path(__file__).resolve().parents[2]  # project root
    build_dir = here / "external" / "Betti-Matching-3D" / "build"
    if build_dir.exists():
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        try:
            return importlib.import_module("betti_matching")
        except Exception as e:
            raise ImportError(
                f"Found {build_dir} but could not import 'betti_matching'. "
                f"Make sure you've built the submodule (make build-betti). Original error: {e}"
            )
    raise ImportError(
        "Betti-Matching-3D build directory not found. "
        "Run: make build-betti"
    )
