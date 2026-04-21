#!/usr/bin/env python3
"""Standalone historical frontier presets.

These are retained as explicit patch/config presets so that historical 38/40/41
frontiers are defined in one place instead of being inlined inside search
scripts.
"""
from __future__ import annotations

from config import SOURCE_OVERRIDES


# Archived same-rule candidate kept only for historical comparison.
BASE_CLEAN_OVERRIDES = dict(SOURCE_OVERRIDES)
BASE_CLEAN_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
        "NR": "NON",
        "ZC": "NON",
    }
)
BASE_CLEAN_OVERRIDES.pop("ZO", None)
BASE_CLEAN_EXCLUDED = {"FB", "ZA", "ZO", "SB", "KC", "ZL"}


# Retained cleaner-doctrine 38/45 line.
STRUCTURAL_38_OVERRIDES = dict(SOURCE_OVERRIDES)
STRUCTURAL_38_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "RAD",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
    }
)
STRUCTURAL_38_EXCLUDED = {"ZA","FB"}  # "FB", "EN", "ES"}


HYBRID_STRUCTURAL_OVERRIDES = dict(BASE_CLEAN_OVERRIDES)
HYBRID_STRUCTURAL_EXCLUDED = set(BASE_CLEAN_EXCLUDED) | {"EN", "ES"}


# Retained 41/45 experimental upper bound.
LEGACY_41_OVERRIDES = dict(SOURCE_OVERRIDES)
LEGACY_41_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "REV",
        "JO": "REV",
        "ZH": "REV",
    }
)
LEGACY_41_OVERRIDES.pop("ZO", None)
LEGACY_41_EXCLUDED = {"FB", "ZA", "ZO", "EN", "ES"}


# Retained 40/45 minimal patch on top of legacy 41.
LEGACY_40_OVERRIDES = dict(LEGACY_41_OVERRIDES)
LEGACY_40_OVERRIDES["JO"] = "RAD"
LEGACY_40_EXCLUDED = set(LEGACY_41_EXCLUDED)
