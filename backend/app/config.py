"""
config.py

Centralizes every file path the backend needs, all relative to the
project root, so nothing is hardcoded per-route. If the project folder
is ever moved, only this file needs to change.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_PATH = DATA_DIR / "processed" / "cleaned_dataset.csv"
FEATURED_DATA_PATH = DATA_DIR / "processed" / "featured_dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_RESULTS_DIR = REPORTS_DIR / "model_results"

BASELINE_RESULTS_PATH = MODEL_RESULTS_DIR / "baseline_results.json"
TUNING_RESULTS_PATH = MODEL_RESULTS_DIR / "tuning_results.json"
DECISION_MATRIX_PATH = MODEL_RESULTS_DIR / "decision_matrix.json"
EXPLAINABILITY_REPORT_PATH = MODEL_RESULTS_DIR / "explainability_report.json"
FINAL_MODEL_CHOICE_PATH = MODEL_RESULTS_DIR / "final_model_choice.json"

VALID_CATEGORIES_PATH = PROJECT_ROOT / "ml" / "prediction" / "valid_categories.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "ml" / "preprocessing" / "feature_config.json"
