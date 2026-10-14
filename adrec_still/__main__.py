"""``python -m adrec_still``: command-line entry point."""

from __future__ import annotations

import sys

from adrec_still.training.cli import main

if __name__ == "__main__":
    sys.exit(main())
