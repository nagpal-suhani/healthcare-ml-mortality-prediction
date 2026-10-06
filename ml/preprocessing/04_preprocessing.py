"""
04_preprocessing.py

Phase 5 — Preprocessing

Purpose:
    1. Split the featured dataset into train/test BEFORE any fitting.
    2. Build a separate sklearn preprocessing Pipeline for Model A and
       Model B (numeric imputation+scaling, categorical
       imputation+one-hot), fit ONLY on the training data.
    3. Run leakage/integrity checks.
    4. Save the fitted pipelines + the raw train/test splits so
       Phase 6 (modeling) can load them directly without repeating
       any of this work or risking re-fitting on test data.

Does NOT train any ML model yet.

Input:
    data/processed/featured_dataset.csv
    ml/preprocessing/feature_config.json

Output:
    data/processed/train_raw.csv, data/processed/test_raw.csv
    ml/preprocessing/pipeline_model_a.joblib
    ml/preprocessing/pipeline_model_b.joblib
    ml/preprocessing/split_report.json

Run:
    python ml/preprocessing/04_preprocessing.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

DATA_PATH = Path("data/processed/featured_dataset.csv")
CONFIG_PATH = Path("ml/preprocessing/feature_config.json")
OUT_DIR = Path("ml/preprocessing")
DATA_OUT_DIR = Path("data/processed")

TARGET = "death_flag"
ID_COL = "patient_id"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipeline(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


def run_leakage_checks(train_df, test_df, model_a_cfg, model_b_cfg):
    checks = {}

    checks["train_rows"] = len(train_df)
    checks["test_rows"] = len(test_df)

    checks["train_target_dist"] = train_df[TARGET].value_counts(normalize=True).round(4).to_dict()
    checks["test_target_dist"] = test_df[TARGET].value_counts(normalize=True).round(4).to_dict()

    overlap = set(train_df[ID_COL]) & set(test_df[ID_COL])
    checks["patient_id_overlap_count"] = len(overlap)

    leakage_cols = ["outcome", "cause_of_death", "recovery_days", "follow_up_required"]
    present_leaks_a = [c for c in leakage_cols
                        if c in model_a_cfg["numeric_features"] + model_a_cfg["categorical_features"]]
    present_leaks_b = [c for c in leakage_cols
                        if c in model_b_cfg["numeric_features"] + model_b_cfg["categorical_features"]]
    checks["leakage_columns_in_model_a"] = present_leaks_a
    checks["leakage_columns_in_model_b"] = present_leaks_b

    post_admission_cols = ["days_hospitalized", "treatment_cost_log", "length_of_stay_category", "treatment_type"]
    present_post_b = [c for c in post_admission_cols
                       if c in model_b_cfg["numeric_features"] + model_b_cfg["categorical_features"]]
    checks["post_admission_columns_in_model_b"] = present_post_b

    print("=== LEAKAGE / INTEGRITY CHECKS ===")
    print(f"Train rows: {checks['train_rows']}, Test rows: {checks['test_rows']}")
    print(f"Train target dist: {checks['train_target_dist']}")
    print(f"Test target dist: {checks['test_target_dist']}")
    print(f"Patient ID overlap between train/test: {checks['patient_id_overlap_count']} (must be 0)")
    print(f"Leakage columns found in Model A features: {present_leaks_a} (must be empty)")
    print(f"Leakage columns found in Model B features: {present_leaks_b} (must be empty)")
    print(f"Post-admission columns found in Model B features: {present_post_b} (must be empty)")

    assert checks["patient_id_overlap_count"] == 0, "FAIL: train/test patient overlap detected"
    assert not present_leaks_a, "FAIL: leakage column present in Model A"
    assert not present_leaks_b, "FAIL: leakage column present in Model B"
    assert not present_post_b, "FAIL: post-admission column present in Model B"
    print("\nAll checks PASSED.\n")

    return checks


def fit_and_verify_no_test_leakage(preprocessor, X_train, X_test):
    """
    Fits the preprocessor on TRAIN ONLY, then transforms both sets.
    Explicitly demonstrates: .fit() is called with X_train alone;
    X_test only ever sees .transform().
    """
    preprocessor.fit(X_train)  # <-- fit on train only
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)  # <-- transform only, no fit
    return preprocessor, X_train_t, X_test_t


def main():
    df = pd.read_csv(DATA_PATH)
    with open(CONFIG_PATH) as f:
        feature_config = json.load(f)
    model_a_cfg = feature_config["model_a"]
    model_b_cfg = feature_config["model_b"]

    print(f"Loaded featured dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # ---- 1. Stratified train/test split (before ANY fitting) ----
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    print(f"Split: {len(train_df)} train / {len(test_df)} test "
          f"(stratified on {TARGET}, random_state={RANDOM_STATE})\n")

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_OUT_DIR / "train_raw.csv", index=False)
    test_df.to_csv(DATA_OUT_DIR / "test_raw.csv", index=False)
    print("Saved data/processed/train_raw.csv and data/processed/test_raw.csv "
          "(raw, unencoded — pipelines below are fit and applied on top of these)\n")

    # ---- 2. Leakage / integrity checks ----
    checks = run_leakage_checks(train_df, test_df, model_a_cfg, model_b_cfg)

    # ---- 3. Build + fit pipelines, train-only ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, cfg in [("model_a", model_a_cfg), ("model_b", model_b_cfg)]:
        num_feats = cfg["numeric_features"]
        cat_feats = cfg["categorical_features"]
        X_train = train_df[num_feats + cat_feats]
        X_test = test_df[num_feats + cat_feats]

        preprocessor = build_pipeline(num_feats, cat_feats)
        preprocessor, X_train_t, X_test_t = fit_and_verify_no_test_leakage(
            preprocessor, X_train, X_test
        )

        out_path = OUT_DIR / f"pipeline_{key}.joblib"
        joblib.dump({
            "preprocessor": preprocessor,
            "numeric_features": num_feats,
            "categorical_features": cat_feats,
            "target": TARGET,
        }, out_path)

        print(f"{cfg['name']}: fit on {X_train.shape[0]} training rows, "
              f"{X_train.shape[1]} raw columns -> "
              f"{X_train_t.shape[1]} columns after encoding")
        print(f"Saved fitted pipeline -> {out_path}\n")

    # ---- 4. Save split/leakage report ----
    report_path = OUT_DIR / "split_report.json"
    with open(report_path, "w") as f:
        json.dump(checks, f, indent=2)
    print(f"Saved split/leakage report -> {report_path}")


if __name__ == "__main__":
    main()
