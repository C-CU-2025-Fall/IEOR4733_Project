#!/usr/bin/env python3
"""Convenience entry for MACD baseline report."""
from __future__ import annotations

import run_frontier_macd as runner


if __name__ == "__main__":
    # Keep behavior centralized in run_frontier_macd.py (--baseline path)
    import sys

    if "--baseline" not in sys.argv:
        sys.argv.append("--baseline")
    runner.main()
