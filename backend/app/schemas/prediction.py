"""
schemas/prediction.py

Pydantic request/response models for the prediction endpoints.
Two separate request schemas because Model A requires 3 extra
post-admission fields (days_hospitalized, treatment_cost_inr,
treatment_type) that Model B deliberately excludes (Phase 5
temporal-availability audit).
"""

from typing import Optional
from pydantic import BaseModel, Field


class PatientInputB(BaseModel):
    """Raw patient input for Model B (early-risk / admission-time)."""
    age: int = Field(..., ge=0, le=120, examples=[65])
    gender: str = Field(..., examples=["Male"])
    state: str = Field(..., examples=["Maharashtra"])
    region: str = Field(..., examples=["West"])
    urban_rural: str = Field(..., examples=["Urban"])
    disease_name: str = Field(..., examples=["Heart Disease"])
    severity: str = Field(..., examples=["Severe"])
    comorbidity: str = Field(..., examples=["Diabetes"])
    smoking_status: str = Field(..., examples=["Former"])
    alcohol_use: str = Field(..., examples=["Occasional"])
    bmi: float = Field(..., ge=10, le=80, examples=[27.4])
    blood_group: str = Field(..., examples=["B+"])
    income_level: str = Field(..., examples=["Middle"])
    insurance_status: str = Field(..., examples=["Private"])
    hospital_type: str = Field(..., examples=["Private"])
    symptoms: str = Field(..., description="Comma-separated symptom list",
                           examples=["Chest pain, Fatigue, Shortness of breath"])


class PatientInputA(PatientInputB):
    """Raw patient input for Model A — adds post-admission fields."""
    days_hospitalized: int = Field(..., ge=0, le=365, examples=[12])
    treatment_cost_inr: float = Field(..., gt=0, examples=[125000])
    treatment_type: str = Field(..., examples=["Medication"])


class ContributingFactor(BaseModel):
    feature: str
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    model_used: str
    algorithm: str
    threshold: float
    probability_of_death: float
    prediction: int
    predicted_label: str
    risk_category: str
    top_contributing_factors: list[ContributingFactor]
    input_warnings: list[str]
    disclaimer: str
