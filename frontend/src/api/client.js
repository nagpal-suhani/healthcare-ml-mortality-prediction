import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
})

/**
 * Normalizes axios errors into a single shape the UI can render
 * consistently, whether it's a network failure, a 422 validation
 * error from FastAPI/Pydantic, or a 4xx/5xx from our own code.
 */
function normalizeError(error) {
  if (error.response) {
    const { status, data } = error.response
    let message = `Request failed (${status})`
    if (data?.detail) {
      message = Array.isArray(data.detail)
        ? data.detail.map((d) => `${d.loc?.at(-1) ?? 'field'}: ${d.msg}`).join('; ')
        : String(data.detail)
    }
    return { status, message, raw: data }
  }
  if (error.request) {
    return {
      status: 0,
      message: `Could not reach the backend at ${API_BASE_URL}. Is it running?`,
      raw: null,
    }
  }
  return { status: -1, message: error.message || 'Unknown error', raw: null }
}

async function get(path, params) {
  try {
    const res = await client.get(path, { params })
    return res.data
  } catch (err) {
    throw normalizeError(err)
  }
}

async function post(path, body) {
  try {
    const res = await client.post(path, body)
    return res.data
  } catch (err) {
    throw normalizeError(err)
  }
}

export const api = {
  baseUrl: API_BASE_URL,

  // Dataset
  getDatasetOverview: () => get('/api/dataset/overview'),
  getDatasetColumns: () => get('/api/dataset/columns'),
  getDatasetPreview: (limit = 20, offset = 0) =>
    get('/api/dataset/preview', { limit, offset }),
  searchDataset: (column, value, limit = 20) =>
    get('/api/dataset/search', { column, value, limit }),
  getMissingValues: () => get('/api/dataset/missing-values'),
  getValidCategories: () => get('/api/dataset/valid-categories'),
  getTrackedSymptoms: () => get('/api/dataset/tracked-symptoms'),

  // Models
  getModelSummary: () => get('/api/models/summary'),
  getBaselineResults: () => get('/api/models/baseline'),
  getDecisionMatrix: () => get('/api/models/decision-matrix'),
  getExplainability: () => get('/api/models/explainability'),
  getFeatureConfig: () => get('/api/models/feature-config'),

  // EDA
  getEdaReport: () => get('/api/eda/report'),
  getEdaFigures: () => get('/api/eda/figures'),
  figureUrl: (filename) => `${API_BASE_URL}/static/figures/${filename}`,

  // Prediction
  predictModelA: (payload) => post('/api/predict/model-a', payload),
  predictModelB: (payload) => post('/api/predict/model-b', payload),
}
