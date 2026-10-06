"""
03_feature_engineering.py

Phase 4 — Feature Engineering

Purpose:
    Build new, clinically-motivated features on top of the cleaned
    dataset, and define the two confirmed parallel feature sets:

    - MODEL A (full-featured):   includes `treatment_type`
    - MODEL B (early-risk-only): excludes `treatment_type`
                                  (a clinical-decision-during-care
                                  variable, per Phase 3 decision)

    `follow_up_required` was already confirmed as leakage in Phase 3
    and is excluded from BOTH feature sets.

    This script only CREATES features and defines feature lists.
    Encoding, scaling, and train/test splitting happen in Phase 5/6,
    fitted on the training data only, to avoid leakage.

Input:
    data/processed/cleaned_dataset.csv

Output:
    data/processed/featured_dataset.csv
    ml/preprocessing/feature_config.json   (feature-set definitions,
                                             reused by Phase 5 and the
                                             backend prediction API)

Run:
    python ml/preprocessing/03_feature_engineering.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/cleaned_dataset.csv")
OUT_PATH = Path("data/processed/featured_dataset.csv")
CONFIG_PATH = Path("ml/preprocessing/feature_config.json")

TARGET = "death_flag"

# Top symptoms selected from the Phase 3 EDA frequency + mortality-rate
# analysis (see reports/EDA_report.md, section 5). Picking a fixed,
# clinically-relevant top set instead of one-hotting all 11,245 unique
# symptom combinations, which would be unusable dimensionality.
TRACKED_SYMPTOMS = [
    "Fatigue", "Fever", "Shortness of breath", "Chest pain",
    "Swelling", "Numbness", "Dizziness", "Blurred vision",
    "Unexplained weight loss",
]


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
    """
    Custom clinical age bands, deliberately different from the
    uniform-decade `age_group` column. Phase 3 EDA found mortality is
    roughly flat (~3-10%) up to age 60, then jumps sharply at 70+
    (~20%). These bands are built around that observed inflection
    point rather than arbitrary 10-year bins, so the boundary itself
    carries signal instead of just being a smoother version of `age`.
    """
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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["bmi_category"] = df["bmi"].apply(bmi_category)
    print("Added bmi_category (Underweight/Normal/Overweight/Obese) — "
          "standard clinical BMI bucketing; lets tree models split on a "
          "clinically meaningful threshold instead of only a raw number.")

    df["age_risk_group"] = df["age"].apply(age_risk_group)
    print("Added age_risk_group (Pediatric/Adult/Senior/Elderly) — built "
          "around the sharp mortality inflection at age 70 found in Phase 3, "
          "not arbitrary decades.")

    df["has_comorbidity"] = (df["comorbidity"] != "No Comorbidity").astype(int)
    print("Added has_comorbidity (0/1) — simple binary summary of the "
          "existing comorbidity column, useful as a quick linear signal "
          "alongside the more detailed comorbidity category.")

    df["length_of_stay_category"] = df["days_hospitalized"].apply(length_of_stay_category)
    print("Added length_of_stay_category (Same-day/Short/Medium/Long) — "
          "`days_hospitalized` was the strongest numerical correlate of "
          "death_flag (r=0.35); bucketing it lets models capture "
          "non-linear risk jumps (e.g. long ICU-level stays) explicitly.")

    df["treatment_cost_log"] = np.log1p(df["treatment_cost_inr"])
    print("Added treatment_cost_log (log1p of treatment_cost_inr) — "
          "Phase 3 found this column heavily right-skewed (12 to "
          "15.3M INR). Log transform is a deterministic, no-fitting-"
          "required transform (safe to do before train/test split, "
          "unlike scaling), so it's done here rather than Phase 5.")

    df["symptom_count"] = df["symptoms"].str.split(",").apply(len)
    print("Added symptom_count — number of symptoms reported per patient; "
          "a simple severity proxy independent of the `severity` label itself.")

    for sym in TRACKED_SYMPTOMS:
        col_name = "symptom_" + sym.lower().replace(" ", "_")
        df[col_name] = df["symptoms"].str.contains(sym, regex=False).astype(int)
    print(f"Added {len(TRACKED_SYMPTOMS)} binary symptom flags for the "
          "most frequent / most mortality-relevant symptoms found in "
          "Phase 3 EDA: " + ", ".join(TRACKED_SYMPTOMS))

    return df


def define_feature_sets():
    """
    Defines the columns that go into each model's feature matrix.
    Excluded everywhere: patient_id, outcome, diagnosis_date, year,
    month, season (no meaningful temporal trend per Phase 3), city
    (132-category high cardinality, redundant with state/region),
    age_group (redundant with age + new age_risk_group),
    disease_category (redundant — fully determined by disease_name,
    which is more granular and already shown to have a strong spread),
    symptoms (raw text — replaced by symptom_count + symptom flags),
    follow_up_required (confirmed leakage, Phase 3).

    TEMPORAL AVAILABILITY AUDIT (Phase 5, pre-preprocessing):
    Model B is meant to predict risk at/near diagnosis/admission —
    before the course of treatment has played out. Reviewed for that:

    - days_hospitalized: NOT available at admission — this is only
      known once a stay is complete/ongoing. EXCLUDED from Model B.
    - length_of_stay_category: derived directly from
      days_hospitalized, same problem. EXCLUDED from Model B.
    - treatment_cost_inr / treatment_cost_log: cost accumulates over
      the course of treatment, not known at diagnosis. EXCLUDED from
      Model B.
    - treatment_type: already excluded from Model B per the Phase 3
      decision (clinical decision made during care).
    - severity, comorbidity, hospital_type, insurance_status,
      income_level, disease_name, symptoms, demographics: all
      genuinely available at/near diagnosis time. KEPT in Model B.

    Model A is not restricted this way — it is explicitly the
    "full clinical picture" model and may use information that
    accrues during the treatment course (days_hospitalized,
    treatment_cost, treatment_type), documented here so that's
    transparent rather than hidden.
    """
    numeric_common = [
        "age", "bmi", "symptom_count", "has_comorbidity",
    ] + ["symptom_" + s.lower().replace(" ", "_") for s in TRACKED_SYMPTOMS]

    categorical_common = [
        "gender", "state", "region", "urban_rural", "disease_name",
        "severity", "comorbidity", "smoking_status", "alcohol_use",
        "blood_group", "income_level", "insurance_status", "hospital_type",
        "bmi_category", "age_risk_group",
    ]

    # Post-admission / accumulates-during-treatment features —
    # legitimate for Model A, excluded from Model B (see audit above).
    post_admission_numeric = ["days_hospitalized", "treatment_cost_log"]
    post_admission_categorical = ["length_of_stay_category"]

    model_a = {
        "name": "Model A - Full Featured",
        "description": "Includes treatment_type and post-admission "
                        "information (length of stay, treatment cost). "
                        "Higher expected accuracy, but reflects the full "
                        "clinical course rather than admission-time risk only.",
        "numeric_features": numeric_common + post_admission_numeric,
        "categorical_features": categorical_common + post_admission_categorical + ["treatment_type"],
        "target": TARGET,
    }
    model_b = {
        "name": "Model B - Early Risk Only",
        "description": "Excludes treatment_type AND all post-admission "
                        "information (days_hospitalized, "
                        "length_of_stay_category, treatment_cost_log). "
                        "Uses only features realistically available at/near "
                        "diagnosis/admission, for a genuine early-warning "
                        "risk tool.",
        "numeric_features": numeric_common,
        "categorical_features": categorical_common,
        "target": TARGET,
    }
    return {"model_a": model_a, "model_b": model_b}


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

    df = add_engineered_features(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved featured dataset -> {OUT_PATH} "
          f"({df.shape[0]} rows, {df.shape[1]} columns)")

    feature_sets = define_feature_sets()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(feature_sets, f, indent=2)
    print(f"Saved feature-set config -> {CONFIG_PATH}")

    for key, cfg in feature_sets.items():
        n_num = len(cfg["numeric_features"])
        n_cat = len(cfg["categorical_features"])
        print(f"\n{cfg['name']}: {n_num} numeric + {n_cat} categorical "
              f"= {n_num + n_cat} raw feature columns (before one-hot encoding)")


if __name__ == "__main__":
    main()
