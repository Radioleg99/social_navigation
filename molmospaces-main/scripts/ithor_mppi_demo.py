#!/usr/bin/env python3
"""Backward-compatible wrapper for the social-nav MPPI experiment entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.social_nav.run_mppi import main


if __name__ == "__main__":
    main(default_method="mppi")
