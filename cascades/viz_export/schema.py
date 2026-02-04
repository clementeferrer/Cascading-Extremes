from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Threshold(BaseModel):
    tau: float
    model: str


class RunMeta(BaseModel):
    run_id: str
    created_at: str
    source: Literal["simulated", "real", "generative"]
    assets: List[str]
    freq: str
    threshold: Threshold
    model_checkpoint: Optional[str]
    config_hash: str


class EventRecord(BaseModel):
    id: int
    t: float
    w: List[float]
    mag: float
    u_tau: float
    asset: str
    intensity: Optional[float] = None
    parent_id: Optional[int] = None
    is_real: bool


class MetricsRecord(BaseModel):
    t: float
    lambda_: Optional[float] = Field(default=None, alias="lambda")
    psi: Optional[float] = None
    mu: Optional[float] = None
    mean_mag: Optional[float] = None
    event_rate: Optional[float] = None
    per_asset_counts: Optional[Dict[str, int]] = None
    direction_density_bin: Optional[List[float]] = None
