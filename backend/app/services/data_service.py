"""
services/data_service.py

Loads the featured dataset ONCE at import time and serves dataset
statistics / preview / column info from memory (no per-request CSV
reads). Never exposes patient_id or any leakage columns to the API.
"""

import math
import pandas as pd
from app.config import FEATURED_DATA_PATH

# Columns that must NEVER be exposed through the API — patient identifier
# and confirmed leakage/near-leakage columns from Phases 2/3.
EXCLUDED_FROM_API = {"patient_id", "outcome", "cause_of_death",
                      "recovery_days", "follow_up_required"}

_df = pd.read_csv(FEATURED_DATA_PATH)
_public_df = _df.drop(columns=[c for c in EXCLUDED_FROM_API if c in _df.columns])


def _clean_nan(value):
    """JSON has no NaN; convert to None for valid JSON output."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def get_dataset_overview() -> dict:
    return {
        "total_records": len(_public_df),
        "total_columns": _public_df.shape[1],
        "target_distribution": {
            str(k): int(v) for k, v in _df["death_flag"].value_counts().items()
        },
        "target_distribution_pct": {
            str(k): round(float(v) * 100, 2)
            for k, v in _df["death_flag"].value_counts(normalize=True).items()
        },
        "missing_values_total": int(_public_df.isna().sum().sum()),
        "date_range": {
            "earliest": str(_df["diagnosis_date"].min()) if "diagnosis_date" in _df else None,
            "latest": str(_df["diagnosis_date"].max()) if "diagnosis_date" in _df else None,
        },
    }


def get_columns_info() -> list:
    info = []
    for col in _public_df.columns:
        dtype = str(_public_df[col].dtype)
        entry = {
            "name": col,
            "dtype": dtype,
            "missing_count": int(_public_df[col].isna().sum()),
        }
        if dtype in ("int64", "float64"):
            entry["min"] = _clean_nan(float(_public_df[col].min()))
            entry["max"] = _clean_nan(float(_public_df[col].max()))
            entry["mean"] = _clean_nan(round(float(_public_df[col].mean()), 2))
        else:
            entry["unique_values"] = int(_public_df[col].nunique())
        info.append(entry)
    return info


def get_preview(limit: int = 20, offset: int = 0) -> dict:
    limit = min(max(limit, 1), 200)  # bound requests
    page = _public_df.iloc[offset:offset + limit]
    records = page.where(pd.notnull(page), None).to_dict(orient="records")
    return {
        "total_records": len(_public_df),
        "offset": offset,
        "limit": limit,
        "records": records,
    }


def search_dataset(column: str, value: str, limit: int = 20) -> dict:
    if column not in _public_df.columns:
        raise ValueError(f"Unknown column: '{column}'")
    mask = _public_df[column].astype(str).str.contains(value, case=False, na=False)
    matched = _public_df[mask].head(min(max(limit, 1), 200))
    records = matched.where(pd.notnull(matched), None).to_dict(orient="records")
    return {"matched_count": int(mask.sum()), "records": records}


def get_missing_value_summary() -> list:
    miss = _public_df.isna().sum()
    miss = miss[miss > 0]
    return [{"column": col, "missing_count": int(cnt),
             "missing_pct": round(float(cnt) / len(_public_df) * 100, 2)}
            for col, cnt in miss.items()]
