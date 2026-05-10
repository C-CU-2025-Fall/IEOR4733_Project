"""
Results I/O — infrastructure for saving experiment data per window.

Directory structure:
    results/<experiment>/<seed>/window_XX/

Each window directory contains:
    - model.pt               — DQNAgent checkpoint (via agent.save())
    - positions_<TICKER>.npy — per-contract position arrays
    - returns_<TICKER>.npz   — net/gross/TC breakdown (npz)
    - diagnostics.json       — per-cycle metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def init_results_dir(experiment_name: str, seed: int, window_idx: int) -> Path:
    """Create results/<experiment>/<seed>/window_XX/ directory."""
    path = Path("results") / experiment_name / str(seed) / f"window_{window_idx:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(agent, path: Path, metadata: dict | None = None) -> None:
    """Save DQNAgent checkpoint via agent.save()."""
    if agent is None:
        return
    agent.save(path / "model.pt", metadata=metadata, include_training_state=False)


def save_positions(positions_dict: dict[str, np.ndarray], path: Path) -> None:
    """Save positions_<TICKER>.npy per contract."""
    if positions_dict is None:
        return
    for ticker, arr in positions_dict.items():
        np.save(path / f"positions_{ticker}.npy", arr)


def save_returns(returns_dict: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    """Save returns_<TICKER>.npz with net/gross/TC arrays."""
    if returns_dict is None:
        return
    for ticker, breakdown in returns_dict.items():
        # breakdown keys: Rt, gross, tc, held, new, eff
        np.savez_compressed(path / f"returns_{ticker}.npz", **breakdown)


def save_diagnostics(diagnostics: dict, path: Path) -> None:
    """Save diagnostics.json with per-cycle metrics."""
    if diagnostics is None:
        return
    with open(path / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)


def load_window_results(path: Path) -> dict:
    """Load all saved data from a window directory."""
    path = Path(path)
    out: dict = {"positions": {}, "returns": {}, "diagnostics": None, "model_exists": False}

    # Positions
    for npy in path.glob("positions_*.npy"):
        ticker = npy.stem.replace("positions_", "")
        out["positions"][ticker] = np.load(npy)

    # Returns
    for npz in path.glob("returns_*.npz"):
        ticker = npz.stem.replace("returns_", "")
        data = np.load(npz)
        out["returns"][ticker] = {k: data[k] for k in data.files}

    # Diagnostics
    diag_path = path / "diagnostics.json"
    if diag_path.exists():
        with open(diag_path) as f:
            out["diagnostics"] = json.load(f)

    # Model
    out["model_exists"] = (path / "model.pt").exists()

    return out