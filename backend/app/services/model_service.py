"""
services/model_service.py

Serves the already-computed model results (baseline metrics, tuning
results, explainability, final decision) straight from the JSON files
Phases 6-9 produced. No model is retrained or re-evaluated here — this
is purely reporting what was already established, so the API can never
show a number that disagrees with the saved reports.
"""

import json
from app.config import (
    BASELINE_RESULTS_PATH, DECISION_MATRIX_PATH,
    EXPLAINABILITY_REPORT_PATH, FINAL_MODEL_CHOICE_PATH, FEATURE_CONFIG_PATH,
)


def _load(path):
    with open(path) as f:
        return json.load(f)


def get_baseline_results() -> dict:
    return _load(BASELINE_RESULTS_PATH)


def get_decision_matrix() -> dict:
    return _load(DECISION_MATRIX_PATH)


def get_explainability_report() -> dict:
    return _load(EXPLAINABILITY_REPORT_PATH)


def get_final_model_choice() -> dict:
    return _load(FINAL_MODEL_CHOICE_PATH)


def get_feature_config() -> dict:
    return _load(FEATURE_CONFIG_PATH)


def get_model_summary() -> dict:
    """
    A single consolidated view for the dashboard: which model was
    finally chosen for A and B, with headline metrics.
    """
    final_choice = get_final_model_choice()
    return {
        "model_a": {
            "algorithm": final_choice["model_a"]["algorithm"],
            "threshold": final_choice["model_a"]["threshold"],
            "metrics": final_choice["model_a"]["test_metrics"],
            "description": "Full clinical-course model — includes treatment "
                            "type, hospital stay length, and treatment cost.",
        },
        "model_b": {
            "algorithm": final_choice["model_b"]["algorithm"],
            "threshold": final_choice["model_b"]["threshold"],
            "metrics": final_choice["model_b"]["test_metrics"],
            "description": "Early-risk model — uses only information "
                            "realistically available at/near diagnosis.",
        },
    }
