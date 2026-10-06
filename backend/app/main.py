"""
main.py

FastAPI application entrypoint.

Run from the `backend/` directory:
    uvicorn app.main:app --reload

Then visit:
    http://127.0.0.1:8000/docs   — interactive Swagger API docs
    http://127.0.0.1:8000/api/... — the actual endpoints
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import FIGURES_DIR
from app.routes import predict, dataset, models, eda
from app.services import predictor_service

load_dotenv()  # reads backend/.env if present (see .env.example)

CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*")
_origins = (
    ["*"] if CORS_ALLOWED_ORIGINS.strip() == "*"
    else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
)

app = FastAPI(
    title="Healthcare Mortality Risk API",
    description=(
        "Backend for the healthcare mortality-risk prediction system. "
        "Serves dataset statistics, EDA results, model performance, "
        "explainability findings, and live predictions from the "
        "finalized Logistic Regression models (Model A: full "
        "clinical-course, Model B: early-risk/admission-time). "
        "Predictions are statistical risk estimates, NOT medical "
        "diagnoses — see the disclaimer field on every prediction response."
    ),
    version="1.0.0",
)

# Allow the local React dev server (and any origin during development) to
# call this API. Set CORS_ALLOWED_ORIGINS in backend/.env to restrict this
# to real frontend domain(s) before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(dataset.router)
app.include_router(models.router)
app.include_router(eda.router)

if FIGURES_DIR.exists():
    app.mount("/static/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


@app.on_event("startup")
def startup_event():
    # Load both prediction pipelines into memory once, at server startup,
    # instead of on the first prediction request.
    predictor_service.warm_up()


@app.get("/")
def root():
    return {
        "message": "Healthcare Mortality Risk API is running.",
        "docs": "/docs",
        "disclaimer": "All predictions are statistical risk estimates, not medical diagnoses.",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    # Alternative to `uvicorn app.main:app --reload` — lets HOST/PORT
    # be set via backend/.env instead of command-line flags.
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
