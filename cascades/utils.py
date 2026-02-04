import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class EmpiricalCDF:
    sorted_values: np.ndarray
    eps: float

    def cdf(self, x: np.ndarray) -> np.ndarray:
        ranks = np.searchsorted(self.sorted_values, x, side="right")
        n = len(self.sorted_values)
        u = (ranks - 0.5) / max(n, 1)
        return np.clip(u, self.eps, 1.0 - self.eps)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        n = len(self.sorted_values)
        grid = (np.arange(1, n + 1) - 0.5) / n
        return np.interp(u, grid, self.sorted_values)
