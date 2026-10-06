"""
predictor.py

Phase 9 — Final Prediction System

Purpose:
    A single, reusable prediction class that takes RAW patient input
    (the same fields a clinician would enter — not the engineered
    feature set) and produces a mortality-risk prediction using the
    finalized Logistic Regression models from Phase 8.

    This is the ONE place feature engineering + preprocessing +
    prediction logic lives. The backend API (Phase 10) will import
    this class directly rather than re-implementing any of this —
    "never manually recreate preprocessing differently for the UI."

Models used:
    models/final_model_a.joblib  (Logistic Regression, threshold=0.40,
                                   full clinical-course features)
    models/final_model_b.joblib  (Logistic Regression, threshold=0.40,
                                   early-risk / admission-time features only)

Run directly for a demo:
    python ml/prediction/predictor.py
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

TRACKED_SYMPTOMS = [
    "Fatigue", "Fever", "Shortness of breath", "Chest pain",
    "Swelling", "Numbness", "Dizziness", "Blurred vision",
    "Unexplained weight loss",
]

DISCLAIMER = (
    "This is a statistical, model-generated risk estimate, NOT a medical "
    "diagnosis. It must not be used as a substitute for clinical judgement "
    "or professional healthcare advice. Always consult a qualified "
    "healthcare provider for diagnosis and treatment decisions."
)

# Required RAW input fields (before feature engineering), common to both models
COMMON_REQUIRED_FIELDS = [
    "age", "gender", "state", "region", "urban_rural", "disease_name",
    "severity", "comorbidity", "smoking_status", "alcohol_use", "bmi",
    "blood_group", "income_level", "insurance_status", "hospital_type",
    "symptoms",
]
# Additional fields required ONLY for Model A (post-admission info)
MODEL_A_EXTRA_FIELDS = ["days_hospitalized", "treatment_cost_inr", "treatment_type"]


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def age_risk_group(age):
    if age < 18:
        return "Pediatric"
    elif age < 60:
        return "Adult"
    elif age < 70:
        return "Senior"
    else:
        return "Elderly"


def length_of_stay_category(days):
    if days == 0:
        return "Same-day"
    elif days <= 7:
        return "Short"
    elif days <= 21:
        return "Medium"
    else:
        return "Long"


class ValidationError(Exception):
    """Raised when raw patient input fails validation."""
    pass


class HealthcareMortalityPredictor:
    """
    Loads one finalized model bundle (Model A or Model B) and exposes
    a single .predict(raw_input) entrypoint that does:
      1. input validation
      2. feature engineering (identical logic to Phase 4)
      3. preprocessing (the EXACT fitted pipeline from Phase 5)
      4. prediction + thresholding (the EXACT threshold from Phase 8)
      5. risk categorization
      6. per-feature contribution breakdown (logistic-regression-native,
         no separate explainability library needed at inference time)
    """

    RISK_BANDS = [
        (0.20, "Low Risk"),
        (0.40, "Moderate Risk"),
        (0.65, "High Risk"),
        (1.01, "Critical Risk"),
    ]

    def __init__(self, model_key: str):
        if model_key not in ("model_a", "model_b"):
            raise ValueError("model_key must be 'model_a' or 'model_b'")
        self.model_key = model_key
        bundle = joblib.load(MODELS_DIR / f"final_{model_key}.joblib")
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.numeric_features = bundle["numeric_features"]
        self.categorical_features = bundle["categorical_features"]
        self.threshold = bundle["threshold"]
        self.algorithm = bundle["algorithm"]
        self.all_features = self.numeric_features + self.categorical_features

        # Known categories the encoder was fit on (for soft validation warnings)
        onehot = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        self.known_categories = dict(zip(self.categorical_features, onehot.categories_))

        self.feature_names_out = self.preprocessor.get_feature_names_out()

    # ---------------------------------------------------------------- #
    def _required_fields(self):
        fields = list(COMMON_REQUIRED_FIELDS)
        if self.model_key == "model_a":
            fields += MODEL_A_EXTRA_FIELDS
        return fields

    def validate(self, raw_input: dict) -> list:
        """Returns a list of warning strings; raises ValidationError on hard failures."""
        errors = []
        warnings = []

        for field in self._required_fields():
            if field not in raw_input or raw_input[field] in (None, ""):
                errors.append(f"Missing required field: '{field}'")

        if errors:
            raise ValidationError("; ".join(errors))

        # Range checks (hard failures — physiologically impossible values)
        if not (0 <= raw_input["age"] <= 120):
            errors.append(f"age={raw_input['age']} is outside plausible range (0-120)")
        if not (10 <= raw_input["bmi"] <= 80):
            errors.append(f"bmi={raw_input['bmi']} is outside plausible range (10-80)")
        if self.model_key == "model_a":
            if raw_input["days_hospitalized"] < 0:
                errors.append("days_hospitalized cannot be negative")
            if raw_input["treatment_cost_inr"] <= 0:
                errors.append("treatment_cost_inr must be positive")

        if errors:
            raise ValidationError("; ".join(errors))

        # Soft warnings — category not seen during training (still handled
        # gracefully by handle_unknown='ignore', but worth flagging)
        for field in self.categorical_features:
            if field in ("bmi_category", "age_risk_group", "length_of_stay_category"):
                continue  # these are engineered, not raw input
            if field in raw_input and raw_input[field] not in self.known_categories.get(field, []):
                warnings.append(f"'{raw_input[field]}' is an unseen category for '{field}' — "
                                 f"model will treat it as unknown (zero contribution from this field)")

        return warnings

    # ---------------------------------------------------------------- #
    def engineer_features(self, raw_input: dict) -> dict:
        row = dict(raw_input)  # copy

        row["bmi_category"] = bmi_category(row["bmi"])
        row["age_risk_group"] = age_risk_group(row["age"])
        row["has_comorbidity"] = 0 if str(row["comorbidity"]).strip().lower() in (
            "no comorbidity", "none", "") else 1

        symptoms_str = row.get("symptoms", "")
        symptom_list = [s.strip() for s in symptoms_str.split(",") if s.strip()]
        row["symptom_count"] = len(symptom_list)
        for sym in TRACKED_SYMPTOMS:
            col = "symptom_" + sym.lower().replace(" ", "_")
            row[col] = int(any(sym.lower() == s.lower() for s in symptom_list))

        if self.model_key == "model_a":
            row["length_of_stay_category"] = length_of_stay_category(row["days_hospitalized"])
            row["treatment_cost_log"] = float(np.log1p(row["treatment_cost_inr"]))

        return row

    # ---------------------------------------------------------------- #
    def risk_category(self, probability: float) -> str:
        for upper_bound, label in self.RISK_BANDS:
            if probability < upper_bound:
                return label
        return "Critical Risk"

    def top_contributing_factors(self, X_transformed_row, top_n=6):
        """
        Logistic-regression-native local explanation: contribution of
        feature i to the logit = coefficient_i * transformed_value_i.
        This is exact for a linear model (no approximation needed),
        and is the same idea SHAP's LinearExplainer computes.
        """
        coefs = self.model.coef_[0]
        contributions = coefs * X_transformed_row
        order = np.argsort(-np.abs(contributions))[:top_n]
        results = []
        for idx in order:
            name = self.feature_names_out[idx].replace("num__", "").replace("cat__", "")
            contrib = float(contributions[idx])
            if X_transformed_row[idx] == 0 and name not in self.numeric_features:
                continue  # skip zero-value one-hot columns (not present for this patient)
            results.append({
                "feature": name,
                "contribution": round(contrib, 4),
                "direction": "increases risk" if contrib > 0 else "decreases risk",
            })
        return results

    # ---------------------------------------------------------------- #
    def predict(self, raw_input: dict) -> dict:
        warnings = self.validate(raw_input)
        engineered = self.engineer_features(raw_input)

        X_df = pd.DataFrame([{f: engineered.get(f) for f in self.all_features}])
        X_transformed = self.preprocessor.transform(X_df)

        probability = float(self.model.predict_proba(X_transformed)[0, 1])
        prediction = int(probability >= self.threshold)
        predicted_label = "Deceased (high mortality risk)" if prediction == 1 else "Survived / Ongoing care (lower mortality risk)"

        contributions = self.top_contributing_factors(X_transformed[0], top_n=6)

        return {
            "model_used": self.model_key,
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "probability_of_death": round(probability, 4),
            "prediction": prediction,
            "predicted_label": predicted_label,
            "risk_category": self.risk_category(probability),
            "top_contributing_factors": contributions,
            "input_warnings": warnings,
            "disclaimer": DISCLAIMER,
        }


# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== DEMO: Model A (full clinical-course) ===\n")
    predictor_a = HealthcareMortalityPredictor("model_a")

    high_risk_patient_a = {
        "age": 78, "gender": "Male", "state": "Maharashtra", "region": "West",
        "urban_rural": "Urban", "disease_name": "Heart Disease", "severity": "Critical",
        "comorbidity": "Multiple", "smoking_status": "Current", "alcohol_use": "Regular",
        "bmi": 31.2, "blood_group": "O+", "income_level": "Lower",
        "insurance_status": "Uninsured", "hospital_type": "Government",
        "symptoms": "Chest pain, Shortness of breath, Fatigue, Dizziness",
        "days_hospitalized": 25, "treatment_cost_inr": 850000, "treatment_type": "ICU + Ventilator",
    }
    result = predictor_a.predict(high_risk_patient_a)
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n=== DEMO: Model B (early-risk, admission-time only) ===\n")
    predictor_b = HealthcareMortalityPredictor("model_b")
    low_risk_patient_b = {
        "age": 24, "gender": "Female", "state": "Kerala", "region": "South",
        "urban_rural": "Urban", "disease_name": "Asthma", "severity": "Mild",
        "comorbidity": "No Comorbidity", "smoking_status": "Never", "alcohol_use": "Never",
        "bmi": 21.5, "blood_group": "A+", "income_level": "Middle",
        "insurance_status": "Private", "hospital_type": "Private",
        "symptoms": "Fatigue, Fever",
    }
    result_b = predictor_b.predict(low_risk_patient_b)
    for k, v in result_b.items():
        print(f"{k}: {v}")
