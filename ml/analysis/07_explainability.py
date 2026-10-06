"""
07_explainability.py

Phase 8 — Model Explainability

Purpose:
    For BOTH Model A and Model B, analyze BOTH the tuned XGBoost model
    (Phase 7 winner by PR-AUC) and the tuned Logistic Regression model
    (Phase 7 runner-up, far more interpretable):

    1. SHAP global + local explanations for XGBoost.
    2. Coefficient analysis for Logistic Regression.
    3. A fair, threshold-optimized test-set evaluation of Logistic
       Regression (it didn't get this in Phase 7, since only the
       PR-AUC winner was threshold-tuned and test-evaluated there).
    4. A final side-by-side decision matrix — XGBoost vs Logistic
       Regression, per feature set — to inform (not make) the final
       model choice.

    Does NOT choose the final model. Does NOT start Phase 9.

Input:
    data/processed/train_raw.csv, data/processed/test_raw.csv
    ml/preprocessing/pipeline_model_a.joblib, pipeline_model_b.joblib
    models/tuned_model_a_xgboost.joblib, tuned_model_a_logistic_regression.joblib
    models/tuned_model_b_xgboost.joblib, tuned_model_b_logistic_regression.joblib
    reports/model_results/final_selection.json   (Phase 7 XGBoost test metrics)

Output:
    reports/figures/22-29_*.png  (SHAP + coefficient plots, 2 feature sets x ~4 charts)
    reports/model_results/explainability_report.json
    reports/model_results/decision_matrix.json
    reports/EXPLAINABILITY_report.md

Run:
    python ml/analysis/07_explainability.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

TRAIN_PATH = Path("data/processed/train_raw.csv")
TEST_PATH = Path("data/processed/test_raw.csv")
PIPE_PATHS = {"model_a": Path("ml/preprocessing/pipeline_model_a.joblib"),
              "model_b": Path("ml/preprocessing/pipeline_model_b.joblib")}
MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/model_results")

TARGET = "death_flag"
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
XGB_THRESHOLDS = {"model_a": 0.60, "model_b": 0.65}

POST_ADMISSION_KEYWORDS = ["days_hospitalized", "treatment_cost", "length_of_stay", "treatment_type"]

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


def get_lr_threshold_and_test_metrics(model_key, lr_model, preprocessor, feats, train_df, test_df):
    """
    Logistic Regression didn't get threshold-tuned + test-evaluated in
    Phase 7 (only the PR-AUC winner did). Doing that here, same method:
    OOF (cross_val_predict) F2-maximizing threshold on TRAIN ONLY, then
    ONE evaluation on the untouched test set.
    """
    X_train = preprocessor.transform(train_df[feats])
    y_train = train_df[TARGET].values
    X_test = preprocessor.transform(test_df[feats])
    y_test = test_df[TARGET].values

    oof_proba = cross_val_predict(lr_model, X_train, y_train, cv=CV,
                                   method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_t, best_f2 = 0.5, -1
    for t in thresholds:
        pred = (oof_proba >= t).astype(int)
        f2 = fbeta_score(y_train, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t

    proba_test = lr_model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= best_t).astype(int)
    cm = confusion_matrix(y_test, pred_test)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "threshold": round(float(best_t), 2),
        "accuracy": round(accuracy_score(y_test, pred_test), 4),
        "precision": round(precision_score(y_test, pred_test, zero_division=0), 4),
        "recall_sensitivity": round(recall_score(y_test, pred_test), 4),
        "specificity": round(tn / (tn + fp), 4),
        "f1": round(f1_score(y_test, pred_test, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba_test), 4),
        "pr_auc": round(average_precision_score(y_test, proba_test), 4),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    return metrics


def analyze_xgboost(model_key, xgb_model, preprocessor, feats, train_df, test_df, title):
    X_train = preprocessor.transform(train_df[feats])
    feature_names = preprocessor.get_feature_names_out()

    # Sample for speed (SHAP TreeExplainer is fast, but keep runtime bounded)
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        X_train.shape[0], size=min(2000, X_train.shape[0]), replace=False)
    X_sample = X_train[sample_idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)
    shap_values.feature_names = list(feature_names)

    # Global importance: mean |SHAP value| per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Beeswarm (global summary)
    plt.figure(figsize=(9, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Summary — XGBoost — {title}")
    plt.tight_layout()
    fname_beeswarm = FIG_DIR / f"{'22' if model_key == 'model_a' else '24'}_shap_summary_{model_key}.png"
    plt.savefig(fname_beeswarm, dpi=130, bbox_inches="tight")
    plt.close()

    # Bar chart of top 15 mean |SHAP|
    fig, ax = plt.subplots(figsize=(8, 7))
    top15 = importance_df.head(15).sort_values("mean_abs_shap")
    ax.barh(top15["feature"], top15["mean_abs_shap"], color="#2E5EAA")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top 15 Features (SHAP) — XGBoost — {title}")
    plt.tight_layout()
    fname_bar = FIG_DIR / f"{'23' if model_key == 'model_a' else '25'}_shap_bar_{model_key}.png"
    plt.savefig(fname_bar, dpi=130)
    plt.close()

    # Local explanation: one correctly-flagged high-risk patient, one low-risk
    X_test_t = preprocessor.transform(test_df[feats])
    proba_test = xgb_model.predict_proba(X_test_t)[:, 1]
    y_test = test_df[TARGET].values
    high_risk_idx = np.argsort(-proba_test)[0]     # most confident "deceased"
    low_risk_idx = np.argsort(proba_test)[0]        # most confident "survived"

    for label, idx in [("high_risk", high_risk_idx), ("low_risk", low_risk_idx)]:
        sv_local = explainer(X_test_t[idx:idx+1])
        sv_local.feature_names = list(feature_names)
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(sv_local[0], max_display=12, show=False)
        plt.title(f"Local Explanation ({label}, true={y_test[idx]}, "
                  f"predicted_proba={proba_test[idx]:.2f}) — {title}")
        plt.tight_layout()
        fname_local = FIG_DIR / f"{'23b' if model_key=='model_a' else '25b'}_shap_waterfall_{model_key}_{label}.png"
        plt.savefig(fname_local, dpi=130, bbox_inches="tight")
        plt.close()

    return importance_df


def analyze_logistic_regression(model_key, lr_model, preprocessor, title):
    feature_names = preprocessor.get_feature_names_out()
    coefs = lr_model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    top20 = coef_df.head(20).sort_values("coefficient")
    colors = ["#C0392B" if c > 0 else "#2E5EAA" for c in top20["coefficient"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top20["feature"], top20["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardized/one-hot scale) — red=increases risk, blue=decreases risk")
    ax.set_title(f"Top 20 Coefficients — Logistic Regression — {title}")
    plt.tight_layout()
    fname = FIG_DIR / f"{'26' if model_key == 'model_a' else '27'}_lr_coefficients_{model_key}.png"
    plt.savefig(fname, dpi=130)
    plt.close()

    return coef_df


def audit_early_availability(top_features, model_key):
    if model_key != "model_b":
        return []
    flagged = []
    for feat in top_features:
        for kw in POST_ADMISSION_KEYWORDS:
            if kw in feat:
                flagged.append(feat)
    return flagged


def clean_feature_name(name):
    return name.replace("num__", "").replace("cat__", "")


def main():
    np.random.seed(RANDOM_STATE)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    with open(RESULTS_DIR / "final_selection.json") as f:
        xgb_final = json.load(f)

    log("# Explainability Report — Phase 8\n")

    explainability_report = {}
    decision_matrix = {}

    titles = {"model_a": "Model A (Full Clinical-Course)", "model_b": "Model B (Early-Risk)"}

    for model_key, pipe_path in PIPE_PATHS.items():
        title = titles[model_key]
        log(f"\n## {title}\n")
        bundle = joblib.load(pipe_path)
        preprocessor = bundle["preprocessor"]
        feats = bundle["numeric_features"] + bundle["categorical_features"]

        xgb_model = joblib.load(MODELS_DIR / f"tuned_{model_key}_xgboost.joblib")
        lr_model = joblib.load(MODELS_DIR / f"tuned_{model_key}_logistic_regression.joblib")

        print(f"  Running SHAP analysis for {model_key} XGBoost...")
        shap_importance = analyze_xgboost(model_key, xgb_model, preprocessor, feats,
                                           train_df, test_df, title)
        top10_xgb = [clean_feature_name(f) for f in shap_importance["feature"].head(10)]
        log(f"**Top 10 features (XGBoost, SHAP mean |value|):** {top10_xgb}")

        print(f"  Running coefficient analysis for {model_key} Logistic Regression...")
        coef_df = analyze_logistic_regression(model_key, lr_model, preprocessor, title)
        top10_lr_pos = [clean_feature_name(f) for f in
                         coef_df[coef_df.coefficient > 0].head(10)["feature"]]
        top10_lr_neg = [clean_feature_name(f) for f in
                         coef_df[coef_df.coefficient < 0].head(10)["feature"]]
        log(f"**Top risk-increasing features (LR, positive coefficient):** {top10_lr_pos}")
        log(f"**Top risk-decreasing features (LR, negative coefficient):** {top10_lr_neg}")

        # Overlap between XGBoost's top 10 (by SHAP) and LR's top 10 (by |coef|)
        top10_lr_abs = [clean_feature_name(f) for f in coef_df.head(10)["feature"]]
        overlap = set(top10_xgb) & set(top10_lr_abs)
        log(f"\n**Feature agreement:** {len(overlap)}/10 features appear in both models' "
            f"top 10 — {sorted(overlap)}")

        # Early-availability audit for Model B
        flagged = audit_early_availability(top10_xgb + top10_lr_abs, model_key)
        if model_key == "model_b":
            if flagged:
                log(f"\n**⚠️ EARLY-AVAILABILITY AUDIT FAILED:** post-admission-looking "
                    f"features found in top features: {flagged}")
            else:
                log(f"\n**✅ Early-availability audit passed:** none of the top "
                    f"important features reference post-admission information "
                    f"(days_hospitalized, treatment_cost, length_of_stay, treatment_type). "
                    f"This confirms the Phase 5 exclusion held through to the final "
                    f"important-feature set — no leakage found.")

        # Fair threshold-optimized test evaluation for LR (XGBoost already has this from Phase 7)
        print(f"  Threshold-tuning + test-evaluating {model_key} Logistic Regression...")
        lr_metrics = get_lr_threshold_and_test_metrics(
            model_key, lr_model, preprocessor, feats, train_df, test_df)
        xgb_metrics = xgb_final[model_key]["test_metrics"]["threshold_optimized"]

        decision_matrix[model_key] = {
            "xgboost": {
                "threshold": XGB_THRESHOLDS[model_key],
                **xgb_metrics,
                "interpretability": "Low (tree ensemble, needs SHAP to explain)",
            },
            "logistic_regression": {
                **lr_metrics,
                "interpretability": "High (linear coefficients, directly explainable)",
            },
        }

        log(f"\n**Test comparison — {title}:**\n")
        log("| Metric | XGBoost | Logistic Regression |")
        log("|---|---|---|")
        for m, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                          ("recall_sensitivity", "Recall"), ("specificity", "Specificity"),
                          ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                          ("false_negative", "False Negatives"), ("false_positive", "False Positives")]:
            xgb_v = xgb_metrics.get(m, "-")
            lr_v = lr_metrics.get(m, "-")
            log(f"| {label} | {xgb_v} | {lr_v} |")

        explainability_report[model_key] = {
            "top10_xgboost_shap": top10_xgb,
            "top10_lr_positive": top10_lr_pos,
            "top10_lr_negative": top10_lr_neg,
            "feature_agreement_count": len(overlap),
            "feature_agreement_list": sorted(overlap),
            "early_availability_flagged": flagged,
        }

    with open(RESULTS_DIR / "explainability_report.json", "w") as f:
        json.dump(explainability_report, f, indent=2)
    with open(RESULTS_DIR / "decision_matrix.json", "w") as f:
        json.dump(decision_matrix, f, indent=2)

    report_path = Path("reports/EXPLAINABILITY_report.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved: {RESULTS_DIR / 'explainability_report.json'}")
    print(f"Saved: {RESULTS_DIR / 'decision_matrix.json'}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
