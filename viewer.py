#!/usr/bin/env python3
"""Launch the PLatS inference viewer; open its URL through an SSH tunnel."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    from vesuvius_p2sd.interactive.server import main
    main(ROOT)
