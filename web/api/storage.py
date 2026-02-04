from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import math


ROOT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT_DIR / "artifacts" / "runs"


def _index_path() -> Path:
    return BASE_DIR / "index.json"


def list_runs() -> List[Dict]:
    path = _index_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("runs", [])


def get_run_path(run_id: str) -> Path:
    runs = list_runs()
    for r in runs:
        if r.get("run_id") == run_id:
            path_val = r.get("path")
            if not path_val:
                return BASE_DIR / run_id
            path = Path(path_val)
            if not path.is_absolute():
                path = ROOT_DIR / path
            return path
    return BASE_DIR / run_id


def read_meta(run_id: str) -> Dict:
    path = get_run_path(run_id) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquet_slice(path: Path, offset: int, limit: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if offset < 0:
        offset = 0
    if limit <= 0:
        limit = len(df)
    return df.iloc[offset : offset + limit]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def read_events(run_id: str, offset: int, limit: int) -> List[Dict]:
    path = get_run_path(run_id) / "events.parquet"
    df = read_parquet_slice(path, offset, limit)
    records = df.to_dict(orient="records")
    return [_to_builtin(r) for r in records]


def read_metrics(run_id: str, offset: int, limit: int) -> List[Dict]:
    path = get_run_path(run_id) / "metrics.parquet"
    if not path.exists():
        return []
    df = read_parquet_slice(path, offset, limit)
    records = df.to_dict(orient="records")
    return [_to_builtin(r) for r in records]
