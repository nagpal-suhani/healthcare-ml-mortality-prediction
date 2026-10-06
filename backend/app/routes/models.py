"""
routes/models.py

GET /api/models/summary            — headline final-model metrics for the dashboard
GET /api/models/baseline           — Phase 6 baseline results (all 8 models)
GET /api/models/decision-matrix    — Phase 8 XGBoost vs Logistic Regression comparison
GET /api/models/explainability     — Phase 8 SHAP / coefficient findings
GET /api/models/feature-config     — Model A / Model B feature list definitions
"""

from fastapi import APIRouter

from app.services import model_service

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/summary")
def model_summary():
    return model_service.get_model_summary()


@router.get("/baseline")
def baseline_results():
    return model_service.get_baseline_results()


@router.get("/decision-matrix")
def decision_matrix():
    return model_service.get_decision_matrix()


@router.get("/explainability")
def explainability_report():
    return model_service.get_explainability_report()


@router.get("/feature-config")
def feature_config():
    return model_service.get_feature_config()
