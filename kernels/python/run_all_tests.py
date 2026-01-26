#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

from kernels.python.run_regression import main


if __name__ == "__main__":
    if "ASCEND_HOME_PATH" not in os.environ:
        sys.stderr.write("error: ASCEND_HOME_PATH is required\n")
        raise SystemExit(2)
    raise SystemExit(main())

