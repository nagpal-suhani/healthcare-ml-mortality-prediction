"""
routes/predict.py

POST /api/predict/model-a
POST /api/predict/model-b
"""

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PatientInputA, PatientInputB, PredictionResponse
from app.services import predictor_service
from ml.prediction.predictor import ValidationError

router = APIRouter(prefix="/api/predict", tags=["Prediction"])


@router.post("/model-a", response_model=PredictionResponse)
def predict_model_a(patient: PatientInputA):
    try:
        result = predictor_service.predict("model_a", patient.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@router.post("/model-b", response_model=PredictionResponse)
def predict_model_b(patient: PatientInputB):
    try:
        result = predictor_service.predict("model_b", patient.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result
