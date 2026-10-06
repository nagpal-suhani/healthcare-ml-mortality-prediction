"""
05_train_baseline_models.py

Phase 6 — Model Training (baseline, no tuning yet)

Purpose:
    Train 4 classification algorithms x 2 feature sets (Model A / Model B)
    = 8 baseline models total, all on the exact same Phase 5 train/test
    split, all using the exact same fitted Phase 5 preprocessing
    pipelines (no refitting on test data), all evaluated with the same
    imbalance-aware metrics.

    NO hyperparameter tuning, NO SHAP/explainability, NO web system yet.
    This phase establishes an honest baseline to tune against in Phase 7.

Algorithms:
    1. Logistic Regression   (class_weight='balanced')
    2. Random Forest         (class_weight='balanced')
    3. XGBoost               (scale_pos_weight = imbalance ratio)
    4. HistGradientBoosting  (class_weight='balanced')

Input:
    data/processed/train_raw.csv, data/processed/test_raw.csv
    ml/preprocessing/pipeline_model_a.joblib
    ml/preprocessing/pipeline_model_b.joblib

Output:
    models/<model_key>_<algo>.joblib          (8 files)
    reports/model_results/baseline_results.json
    reports/figures/11_roc_curves_model_a.png, 12_..._model_b.png
    reports/figures/13_pr_curves_model_a.png,  14_..._model_b.png
    reports/figures/15_confusion_matrices_model_a.png, 16_..._model_b.png
    reports/figures/17_model_comparison.png

Run:
    python ml/training/05_train_baseline_models.py
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

sns.set_theme(style="whitegrid", font_scale=0.9)

TRAIN_PATH = Path("data/processed/train_raw.csv")
TEST_PATH = Path("data/processed/test_raw.csv")
PIPE_A_PATH = Path("ml/preprocessing/pipeline_model_a.joblib")
PIPE_B_PATH = Path("ml/preprocessing/pipeline_model_b.joblib")
MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures")
RESULTS_PATH = Path("reports/model_results/baseline_results.json")

RANDOM_STATE = 42
TARGET = "death_flag"


def get_algorithms(pos_weight_ratio):
    """Fresh, unfitted estimator instances for one feature set."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, scale_pos_weight=pos_weight_ratio,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]), "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]), "true_positive": int(cm[1, 1]),
        },
    }


def train_and_evaluate_one_set(model_key, pipe_bundle, train_df, test_df):
    preprocessor = pipe_bundle["preprocessor"]
    num_feats = pipe_bundle["numeric_features"]
    cat_feats = pipe_bundle["categorical_features"]
    all_feats = num_feats + cat_feats

    X_train_raw = train_df[all_feats]
    X_test_raw = test_df[all_feats]
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    # Preprocessor was already FIT in Phase 5 on train only — reuse it,
    # never refit here. .transform() only.
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    pos_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"[{model_key}] scale_pos_weight (for XGBoost) = {pos_weight_ratio:.2f} "
          f"(class imbalance ratio in TRAIN set)")

    algorithms = get_algorithms(pos_weight_ratio)
    results = {}
    fitted_models = {}
    roc_data = {}
    pr_data = {}

    for algo_name, model in algorithms.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[algo_name] = evaluate(y_test, y_pred, y_proba)
        fitted_models[algo_name] = model

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[algo_name] = (fpr, tpr, results[algo_name]["roc_auc"])
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_data[algo_name] = (rec, prec, results[algo_name]["pr_auc"])

        out_path = MODELS_DIR / f"{model_key}_{algo_name}.joblib"
        joblib.dump(model, out_path)

        print(f"  {algo_name}: acc={results[algo_name]['accuracy']}, "
              f"recall={results[algo_name]['recall']}, "
              f"f1={results[algo_name]['f1']}, "
              f"roc_auc={results[algo_name]['roc_auc']}, "
              f"pr_auc={results[algo_name]['pr_auc']}")

    return results, fitted_models, roc_data, pr_data


def plot_roc_curves(roc_data, model_key, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    for algo, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{algo} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {title}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fname = FIG_DIR / f"{'11' if model_key == 'model_a' else '12'}_roc_curves_{model_key}.png"
    plt.savefig(fname, dpi=130)
    plt.close()
    return fname


def plot_pr_curves(pr_data, model_key, title, baseline_rate):
    fig, ax = plt.subplots(figsize=(7, 6))
    for algo, (rec, prec, ap) in pr_data.items():
        ax.plot(rec, prec, label=f"{algo} (AP={ap:.3f})")
    ax.axhline(baseline_rate, color="k", linestyle="--", alpha=0.4,
               label=f"Random (baseline={baseline_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {title}")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fname = FIG_DIR / f"{'13' if model_key == 'model_a' else '14'}_pr_curves_{model_key}.png"
    plt.savefig(fname, dpi=130)
    plt.close()
    return fname


def plot_confusion_matrices(results, model_key, title):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (algo, res) in enumerate(results.items()):
        cm = np.array([
            [res["confusion_matrix"]["true_negative"], res["confusion_matrix"]["false_positive"]],
            [res["confusion_matrix"]["false_negative"], res["confusion_matrix"]["true_positive"]],
        ])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Pred: Survived", "Pred: Deceased"],
                    yticklabels=["True: Survived", "True: Deceased"], cbar=False)
        axes[i].set_title(algo)
    plt.suptitle(f"Confusion Matrices — {title}")
    plt.tight_layout()
    fname = FIG_DIR / f"{'15' if model_key == 'model_a' else '16'}_confusion_matrices_{model_key}.png"
    plt.savefig(fname, dpi=130)
    plt.close()
    return fname


def plot_model_comparison(all_results):
    rows = []
    for model_key, algos in all_results.items():
        for algo, res in algos.items():
            rows.append({"feature_set": model_key, "algorithm": algo,
                         "recall": res["recall"], "f1": res["f1"],
                         "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"]})
    comp_df = pd.DataFrame(rows)
    comp_df["label"] = comp_df["feature_set"] + " - " + comp_df["algorithm"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    metrics = ["recall", "f1", "roc_auc", "pr_auc"]
    for i, m in enumerate(metrics):
        d = comp_df.sort_values(m, ascending=True)
        colors = ["#2E5EAA" if fs == "model_a" else "#C0392B" for fs in d["feature_set"]]
        axes[i].barh(d["label"], d[m], color=colors)
        axes[i].set_title(m)
        axes[i].set_xlim(0, 1)
    plt.suptitle("Baseline Model Comparison — blue=Model A (full), red=Model B (early-risk)")
    plt.tight_layout()
    fname = FIG_DIR / "17_model_comparison.png"
    plt.savefig(fname, dpi=130)
    plt.close()
    return fname, comp_df


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    baseline_rate = test_df[TARGET].mean()
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows | "
          f"Test positive rate: {baseline_rate:.4f}\n")

    all_results = {}
    for model_key, pipe_path, title in [
        ("model_a", PIPE_A_PATH, "Model A (Full Clinical-Course)"),
        ("model_b", PIPE_B_PATH, "Model B (Early-Risk / Admission-Time)"),
    ]:
        print(f"\n{'='*70}\nTraining {title}\n{'='*70}")
        pipe_bundle = joblib.load(pipe_path)
        results, fitted_models, roc_data, pr_data = train_and_evaluate_one_set(
            model_key, pipe_bundle, train_df, test_df
        )
        all_results[model_key] = results

        plot_roc_curves(roc_data, model_key, title)
        plot_pr_curves(pr_data, model_key, title, baseline_rate)
        plot_confusion_matrices(results, model_key, title)
        print(f"Saved ROC / PR / confusion-matrix charts for {model_key}")

    fname, comp_df = plot_model_comparison(all_results)
    print(f"\nSaved model comparison chart -> {fname}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved machine-readable results -> {RESULTS_PATH}")

    print("\n=== BASELINE COMPARISON TABLE ===")
    print(comp_df[["feature_set", "algorithm", "recall", "f1", "roc_auc", "pr_auc"]]
          .sort_values(["feature_set", "roc_auc"], ascending=[True, False])
          .to_string(index=False))


if __name__ == "__main__":
    main()
