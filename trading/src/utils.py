"""
utils.py — shared helpers: config loading, logging, path resolution.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Project root — two levels up from this file (src/utils.py → project/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load YAML config from *path*.  Defaults to <project_root>/config/config.yaml.
    Relative paths in the returned dict are resolved relative to PROJECT_ROOT
    so callers don't need to worry about the working directory.
    """
    if path is None:
        path = PROJECT_ROOT / "config" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Return an absolute path; relative paths are anchored to PROJECT_ROOT."""
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_dir(p: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return Path."""
    p = resolve_path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a simple console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
