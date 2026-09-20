from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_TARGET = "y_log1p_resolution_hours"
FINAL_STATUSES = {"closed", "completed", "complete", "resolved"}


def prepare_resolved_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the resolved-only analysis frame from harmonized Step-2 data.

    Completion conditioning is explicit. Deduplication uses request ID plus
    chronological order and does not fit any data-dependent parameter.
    """

    required = {"request_id", "created_date", "closed_date", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["request_id"] = out["request_id"].astype("string").str.strip()
    out = out[out["request_id"].notna() & out["request_id"].ne("")].copy()
    out["created_date"] = pd.to_datetime(
        out["created_date"], errors="coerce", utc=True, format="mixed"
    )
    out["closed_date"] = pd.to_datetime(
        out["closed_date"], errors="coerce", utc=True, format="mixed"
    )
    out = out[out["created_date"].notna() & out["closed_date"].notna()].copy()

    if "status" in out.columns:
        status = out["status"].astype(str).str.lower().str.strip()
        out = out[status.isin(FINAL_STATUSES)].copy()

    hours = (out["closed_date"] - out["created_date"]).dt.total_seconds() / 3600.0
    out = out[hours.ge(0)].copy()
    out = out.sort_values(["created_date", "request_id"], kind="stable")
    out = out.drop_duplicates(subset=["request_id"], keep="first").copy()

    hours = (out["closed_date"] - out["created_date"]).dt.total_seconds() / 3600.0
    out["resolution_hours"] = hours.astype(float)
    out[PRIMARY_TARGET] = np.log1p(out["resolution_hours"].to_numpy(dtype=float)).astype(np.float32)

    dt = out["created_date"]
    out["hour"] = dt.dt.hour.astype(np.float32)
    out["day_of_week"] = dt.dt.dayofweek.astype(np.float32)
    out["month"] = dt.dt.month.astype(np.float32)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32)
    out["category"] = out["category"].fillna("UNK").astype(str)
    if "descriptor" in out.columns:
        out["descriptor"] = out["descriptor"].fillna("").astype(str)
    return out.sort_values(["created_date", "request_id"], kind="stable").reset_index(drop=True)


def frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "unique_request_ids": int(frame["request_id"].nunique()) if len(frame) else 0,
        "created_date_min": frame["created_date"].min().isoformat() if len(frame) else None,
        "created_date_max": frame["created_date"].max().isoformat() if len(frame) else None,
        "resolution_hours_median": float(frame["resolution_hours"].median()) if len(frame) else None,
    }
