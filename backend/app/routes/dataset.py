"""
routes/dataset.py

GET /api/dataset/overview
GET /api/dataset/columns
GET /api/dataset/preview
GET /api/dataset/search
GET /api/dataset/missing-values
GET /api/dataset/valid-categories   (for frontend dropdowns)
GET /api/dataset/tracked-symptoms   (for frontend symptom checklist)
"""

import json
import sys
from fastapi import APIRouter, HTTPException, Query

from app.services import data_service
from app.config import VALID_CATEGORIES_PATH, PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/overview")
def dataset_overview():
    return data_service.get_dataset_overview()


@router.get("/columns")
def dataset_columns():
    return data_service.get_columns_info()


@router.get("/preview")
def dataset_preview(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    return data_service.get_preview(limit=limit, offset=offset)


@router.get("/search")
def dataset_search(column: str, value: str, limit: int = Query(20, ge=1, le=200)):
    try:
        return data_service.search_dataset(column, value, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/missing-values")
def dataset_missing_values():
    return data_service.get_missing_value_summary()


@router.get("/valid-categories")
def valid_categories():
    with open(VALID_CATEGORIES_PATH) as f:
        return json.load(f)


@router.get("/tracked-symptoms")
def tracked_symptoms():
    """
    Exposes the exact symptom list the trained models were built on
    (ml/prediction/predictor.py::TRACKED_SYMPTOMS), read-only, so the
    frontend never hardcodes its own copy of this list.
    """
    from ml.prediction.predictor import TRACKED_SYMPTOMS
    return {"symptoms": TRACKED_SYMPTOMS}
