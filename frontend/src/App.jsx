import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import Dashboard from './pages/Dashboard'
import DataAnalysis from './pages/DataAnalysis'
import ModelPerformance from './pages/ModelPerformance'
import Explainability from './pages/Explainability'
import Prediction from './pages/Prediction'
import Dataset from './pages/Dataset'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data-analysis" element={<DataAnalysis />} />
          <Route path="/model-performance" element={<ModelPerformance />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/dataset" element={<Dataset />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
