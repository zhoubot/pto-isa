#!/usr/bin/env python3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "binding" / "python"))

from ptoas.python.st.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
