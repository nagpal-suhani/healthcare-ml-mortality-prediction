# Frontend — Healthcare Mortality Risk Dashboard

React + Vite + Tailwind CSS v4 dashboard for the healthcare mortality-risk
analytics project. See the [project root README](../README.md) for full
setup instructions, API reference, and project background.

## Quick start

```bash
npm install
copy .env.example .env    # Windows; use `cp` on macOS/Linux
npm run dev
```

Requires the backend (`../backend`) to be running — see the root README.

## Structure

- `src/api/client.js` — single API service layer, all backend calls
- `src/hooks/useApi.js` — shared loading/error/data-fetching hook
- `src/components/` — reusable UI (Layout, StatCard, MetricsTable, FigureCard, RiskBadge, StateBlock, Section)
- `src/pages/` — the 6 dashboard pages (Dashboard, DataAnalysis, ModelPerformance, Explainability, Prediction, Dataset)
