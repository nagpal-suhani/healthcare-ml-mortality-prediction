"""
06_tune_and_select.py

Phase 7 — Model Comparison / Hyperparameter Tuning

Purpose:
    1. Tune the 4 baseline algorithms (Logistic Regression, Random Forest,
       XGBoost, HistGradientBoosting) via RandomizedSearchCV + 5-fold
       stratified cross-validation, ON TRAINING DATA ONLY.
    2. For the best tuned model per feature set, run a decision-threshold
       analysis using out-of-fold (cross_val_predict) probabilities on
       TRAINING DATA ONLY — the test set is never used for threshold
       selection.
    3. Select one final model + one final threshold for Model A and one
       for Model B.
    4. Evaluate each final model EXACTLY ONCE on the untouched test set.

Input:
    data/processed/train_raw.csv, data/processed/test_raw.csv
    ml/preprocessing/pipeline_model_a.joblib
    ml/preprocessing/pipeline_model_b.joblib
    reports/model_results/baseline_results.json   (Phase 6, for comparison)

Output:
    models/tuned_<model_key>_<algo>.joblib   (best estimator per algo/feature set)
    reports/model_results/tuning_results.json
    reports/model_results/threshold_analysis.json
    reports/model_results/final_selection.json
    reports/figures/18_cv_comparison.png
    reports/figures/19_threshold_analysis_model_a.png
    reports/figures/20_threshold_analysis_model_b.png
    reports/figures/21_baseline_vs_tuned.png

Run:
    python ml/training/06_tune_and_select.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_validate, cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    make_scorer, average_precision_score, roc_auc_score, recall_score,
    precision_score, f1_score, accuracy_score, confusion_matrix, fbeta_score,
)

sns.set_theme(style="whitegrid", font_scale=0.9)

TRAIN_PATH = Path("data/processed/train_raw.csv")
TEST_PATH = Path("data/processed/test_raw.csv")
PIPE_PATHS = {"model_a": Path("ml/preprocessing/pipeline_model_a.joblib"),
              "model_b": Path("ml/preprocessing/pipeline_model_b.joblib")}
BASELINE_RESULTS_PATH = Path("reports/model_results/baseline_results.json")
MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/model_results")

RANDOM_STATE = 42
TARGET = "death_flag"
N_ITER = 12
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def param_distributions(pos_weight_ratio):
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, solver="liblinear"),
            {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "class_weight": ["balanced", {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 15}],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300, 400],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", 0.5],
                "class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        "xgboost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "scale_pos_weight": [pos_weight_ratio * 0.5, pos_weight_ratio,
                                      pos_weight_ratio * 1.5, pos_weight_ratio * 2],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_iter": [100, 200, 300],
                "max_leaf_nodes": [15, 31, 63],
                "max_depth": [None, 5, 10],
                "class_weight": ["balanced", None],
                "l2_regularization": [0, 0.1, 1.0],
            },
        ),
    }


def tune_one(algo_name, estimator, params, X_train, y_train):
    search = RandomizedSearchCV(
        estimator, params, n_iter=N_ITER, cv=CV,
        scoring="average_precision", random_state=RANDOM_STATE,
        n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)

    # Full CV metric report for the winning params (supplementary,
    # beyond the single average_precision score search used internally)
    scoring = {
        "recall": "recall", "precision": "precision", "f1": "f1",
        "roc_auc": "roc_auc", "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
    }
    cv_report = cross_validate(search.best_estimator_, X_train, y_train,
                                cv=CV, scoring=scoring, n_jobs=-1)

    summary = {
        "best_params": {k: (v if not isinstance(v, dict) else v) for k, v in search.best_params_.items()},
        "best_cv_pr_auc": round(search.best_score_, 4),
        "cv_recall_mean": round(cv_report["test_recall"].mean(), 4),
        "cv_precision_mean": round(cv_report["test_precision"].mean(), 4),
        "cv_f1_mean": round(cv_report["test_f1"].mean(), 4),
        "cv_roc_auc_mean": round(cv_report["test_roc_auc"].mean(), 4),
        "cv_pr_auc_mean": round(cv_report["test_pr_auc"].mean(), 4),
    }
    print(f"    [{algo_name}] best CV PR-AUC={summary['best_cv_pr_auc']}, "
          f"CV recall={summary['cv_recall_mean']}, CV F1={summary['cv_f1_mean']}, "
          f"CV ROC-AUC={summary['cv_roc_auc_mean']}")
    return search.best_estimator_, summary


def threshold_sweep(estimator, X_train, y_train):
    """
    Out-of-fold probabilities via cross_val_predict — every prediction
    comes from a fold where that row was held out, so this is a fair
    estimate WITHOUT touching the test set at all.
    """
    oof_proba = cross_val_predict(estimator, X_train, y_train, cv=CV,
                                   method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        pred = (oof_proba >= t).astype(int)
        cm = confusion_matrix(y_train, pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(precision_score(y_train, pred, zero_division=0), 4),
            "recall": round(recall_score(y_train, pred), 4),
            "f1": round(f1_score(y_train, pred, zero_division=0), 4),
            "f2": round(fbeta_score(y_train, pred, beta=2, zero_division=0), 4),
            "false_negatives": int(fn),
            "false_positives": int(fp),
        })
    sweep_df = pd.DataFrame(rows)
    best_row = sweep_df.loc[sweep_df["f2"].idxmax()]
    return sweep_df, float(best_row["threshold"])


def final_test_evaluation(estimator, X_test, y_test, threshold):
    proba = estimator.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)
    pred_opt = (proba >= threshold).astype(int)

    def metrics_for(pred):
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall_sensitivity": round(recall_score(y_test, pred), 4),
            "specificity": round(tn / (tn + fp), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        }

    return {"threshold_0.5": metrics_for(pred_default),
            "threshold_optimized": metrics_for(pred_opt)}


def plot_cv_comparison(tuning_results):
    rows = []
    for model_key, algos in tuning_results.items():
        for algo, res in algos.items():
            rows.append({"feature_set": model_key, "algorithm": algo,
                         "cv_recall": res["cv_recall_mean"], "cv_f1": res["cv_f1_mean"],
                         "cv_roc_auc": res["cv_roc_auc_mean"], "cv_pr_auc": res["cv_pr_auc_mean"]})
    df = pd.DataFrame(rows)
    df["label"] = df["feature_set"] + " - " + df["algorithm"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for i, m in enumerate(["cv_recall", "cv_f1", "cv_roc_auc", "cv_pr_auc"]):
        d = df.sort_values(m)
        colors = ["#2E5EAA" if fs == "model_a" else "#C0392B" for fs in d["feature_set"]]
        axes[i].barh(d["label"], d[m], color=colors)
        axes[i].set_title(m)
        axes[i].set_xlim(0, 1)
    plt.suptitle("5-Fold CV Comparison After Tuning — blue=Model A, red=Model B")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "18_cv_comparison.png", dpi=130)
    plt.close()


def plot_threshold_sweep(sweep_df, best_threshold, model_key, algo_name):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", marker="o")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", marker="o")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", marker="o")
    ax.plot(sweep_df["threshold"], sweep_df["f2"], label="F2 (recall-weighted)", marker="o")
    ax.axvline(best_threshold, color="red", linestyle="--", alpha=0.6,
               label=f"Selected threshold={best_threshold}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis (OOF, train-only) — {model_key} / {algo_name}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    suffix = "model_a" if model_key == "model_a" else "model_b"
    fname = FIG_DIR / f"{'19' if model_key == 'model_a' else '20'}_threshold_analysis_{suffix}.png"
    plt.savefig(fname, dpi=130)
    plt.close()


def plot_baseline_vs_tuned(baseline_results, final_selection):
    rows = []
    for model_key in ["model_a", "model_b"]:
        chosen_algo = final_selection[model_key]["algorithm"]
        base = baseline_results[model_key][chosen_algo]
        tuned = final_selection[model_key]["test_metrics"]["threshold_optimized"]
        for metric in ["recall", "f1", "roc_auc", "pr_auc"]:
            base_val = base[metric] if metric != "recall" else base["recall"]
            tuned_val = tuned[metric] if metric != "recall" else tuned["recall_sensitivity"]
            rows.append({"feature_set": model_key, "metric": metric,
                         "baseline": base_val, "tuned_optimized_threshold": tuned_val})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, model_key in enumerate(["model_a", "model_b"]):
        d = df[df.feature_set == model_key].set_index("metric")[["baseline", "tuned_optimized_threshold"]]
        d.plot(kind="bar", ax=axes[i], color=["#95A5A6", "#2E5EAA" if model_key == "model_a" else "#C0392B"])
        axes[i].set_title(f"{model_key}: baseline vs tuned+optimized-threshold")
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "21_baseline_vs_tuned.png", dpi=130)
    plt.close()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    with open(BASELINE_RESULTS_PATH) as f:
        baseline_results = json.load(f)

    tuning_results = {}
    threshold_results = {}
    fitted_best_per_set = {}

    for model_key, pipe_path in PIPE_PATHS.items():
        print(f"\n{'='*70}\nTuning {model_key}\n{'='*70}")
        bundle = joblib.load(pipe_path)
        preprocessor = bundle["preprocessor"]
        feats = bundle["numeric_features"] + bundle["categorical_features"]

        X_train = preprocessor.transform(train_df[feats])
        y_train = train_df[TARGET].values
        X_test = preprocessor.transform(test_df[feats])
        y_test = test_df[TARGET].values

        pos_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        algos = param_distributions(pos_weight_ratio)

        tuning_results[model_key] = {}
        fitted_estimators = {}
        for algo_name, (estimator, params) in algos.items():
            print(f"  Tuning {algo_name} (n_iter={N_ITER}, 5-fold CV, scoring=PR-AUC)...")
            best_est, summary = tune_one(algo_name, estimator, params, X_train, y_train)
            tuning_results[model_key][algo_name] = summary
            fitted_estimators[algo_name] = best_est
            joblib.dump(best_est, MODELS_DIR / f"tuned_{model_key}_{algo_name}.joblib")

        # Pick best algorithm for this feature set by CV PR-AUC
        best_algo = max(tuning_results[model_key],
                         key=lambda a: tuning_results[model_key][a]["best_cv_pr_auc"])
        print(f"\n  --> Best algorithm for {model_key} by CV PR-AUC: {best_algo}")

        best_estimator = fitted_estimators[best_algo]
        sweep_df, best_threshold = threshold_sweep(best_estimator, X_train, y_train)
        plot_threshold_sweep(sweep_df, best_threshold, model_key, best_algo)
        threshold_results[model_key] = {
            "algorithm": best_algo,
            "selected_threshold": best_threshold,
            "sweep_table": sweep_df.to_dict(orient="records"),
        }
        print(f"  Selected threshold (max F2, OOF/train-only) = {best_threshold}")

        test_metrics = final_test_evaluation(best_estimator, X_test, y_test, best_threshold)
        fitted_best_per_set[model_key] = {
            "algorithm": best_algo,
            "threshold": best_threshold,
            "test_metrics": test_metrics,
            "cv_summary": tuning_results[model_key][best_algo],
        }
        print(f"  FINAL TEST (touched once) — threshold {best_threshold}: "
              f"recall={test_metrics['threshold_optimized']['recall_sensitivity']}, "
              f"precision={test_metrics['threshold_optimized']['precision']}, "
              f"f1={test_metrics['threshold_optimized']['f1']}, "
              f"FN={test_metrics['threshold_optimized']['false_negative']}, "
              f"FP={test_metrics['threshold_optimized']['false_positive']}")

    plot_cv_comparison(tuning_results)
    plot_baseline_vs_tuned(baseline_results, fitted_best_per_set)

    with open(RESULTS_DIR / "tuning_results.json", "w") as f:
        json.dump(tuning_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "threshold_analysis.json", "w") as f:
        json.dump(threshold_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "final_selection.json", "w") as f:
        json.dump(fitted_best_per_set, f, indent=2, default=str)

    print("\nSaved: tuning_results.json, threshold_analysis.json, final_selection.json")
    print("Saved charts: 18_cv_comparison.png, 19/20_threshold_analysis_*.png, 21_baseline_vs_tuned.png")


if __name__ == "__main__":
    main()
