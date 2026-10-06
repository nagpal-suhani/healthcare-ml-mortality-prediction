"""
routes/eda.py

GET /api/eda/report    — full Phase 3 EDA markdown report (text)
GET /api/eda/figures   — list of available chart filenames (served
                          statically at /static/figures/<filename>)
"""

from fastapi import APIRouter, HTTPException

from app.config import REPORTS_DIR, FIGURES_DIR

router = APIRouter(prefix="/api/eda", tags=["EDA"])


@router.get("/report")
def eda_report():
    path = REPORTS_DIR / "EDA_report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="EDA report not found")
    return {"content": path.read_text(encoding="utf-8")}


@router.get("/figures")
def eda_figures():
    if not FIGURES_DIR.exists():
        return {"figures": []}
    files = sorted(p.name for p in FIGURES_DIR.glob("*.png"))
    return {"figures": files, "base_url": "/static/figures/"}
