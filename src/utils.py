"""Utility functions for logging, configuration, and helpers."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)
    console_level = getattr(logging, cfg.get("console_level", "WARNING").upper(), logging.WARNING)
    log_file = Path(cfg.get("file", "logs/backtest.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("backtest")
    logger.setLevel(level)
    logger.handlers.clear()
    
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    return logger

def get_trading_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """Get trading dates for a given frequency."""
    if freq.upper() == "D":
        return set(index)
    rule = freq.upper()
    dates = pd.Series(index, index=index)
    return set(dates.groupby(dates.index.to_period(rule)).last().tolist())

def safe_float(x) -> float:
    """Convert to float, handling NaN and inf."""
    if pd.isna(x) or not np.isfinite(x):
        return np.nan
    return float(x)

def progress_bar(iterable, desc: str = "Processing", **kwargs):
    """Wrapper for tqdm progress bar."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, **kwargs)
    except ImportError:
        return iterable

def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml_config(data: Dict[str, Any], path: str | Path):
    """Save configuration to YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def timer(func):
    """Decorator to time function execution."""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper