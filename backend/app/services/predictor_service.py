"""
services/predictor_service.py

Loads BOTH finalized predictors (Model A, Model B) ONCE at import time
(module-level singletons), so every request reuses the same fitted
pipeline instead of reloading joblib files from disk per-request.

This is the ONLY place the backend touches the ML prediction logic —
it imports the exact same class used in Phase 9's standalone demo, so
there is no risk of the API silently drifting from what was trained
and tested.
"""

import sys
from app.config import PROJECT_ROOT

# The `ml` package lives at the project root (a sibling of `backend/`),
# not inside backend/. Add it to sys.path so `from ml.prediction...`
# resolves regardless of whether uvicorn was launched from backend/
# or from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.prediction.predictor import HealthcareMortalityPredictor, ValidationError

_predictors: dict[str, HealthcareMortalityPredictor] = {}


def get_predictor(model_key: str) -> HealthcareMortalityPredictor:
    if model_key not in _predictors:
        _predictors[model_key] = HealthcareMortalityPredictor(model_key)
    return _predictors[model_key]


def warm_up():
    """Called on FastAPI startup so the first real request isn't slow."""
    get_predictor("model_a")
    get_predictor("model_b")


def predict(model_key: str, raw_input: dict) -> dict:
    predictor = get_predictor(model_key)
    return predictor.predict(raw_input)
