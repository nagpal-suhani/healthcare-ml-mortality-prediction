import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Section } from '../components/Section'
import { StateBlock } from '../components/StateBlock'
import { MetricsTable } from '../components/MetricsTable'
import { FigureCard } from '../components/FigureCard'

const FINAL_METRIC_KEYS = [
  'accuracy', 'precision', 'recall_sensitivity', 'specificity', 'f1', 'roc_auc', 'pr_auc',
  'false_negative', 'false_positive',
]
const BASELINE_METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

export default function ModelPerformance() {
  const decision = useApi(api.getDecisionMatrix, [])
  const baseline = useApi(api.getBaselineResults, [])
  const figures = useApi(api.getEdaFigures, [])

  const perfFigures = (figures.data?.figures || []).filter((f) => /^(1[1-9]|2[01])_/.test(f))

  return (
    <div>
      <Section
        title="Final model: Logistic Regression, Model A vs Model B"
        description="Both finalized models use Logistic Regression at threshold 0.40. This compares them directly on the untouched test set."
      >
        <StateBlock loading={decision.loading} error={decision.error} onRetry={decision.reload}>
          {decision.data && (
            <MetricsTable
              metricKeys={FINAL_METRIC_KEYS}
              columns={[
                { key: 'a', label: 'Model A (full clinical-course)' },
                { key: 'b', label: 'Model B (early-risk)' },
              ]}
              rowsByColumn={{
                a: decision.data.model_a.logistic_regression,
                b: decision.data.model_b.logistic_regression,
              }}
            />
          )}
        </StateBlock>
      </Section>

      <Section
        title="Logistic Regression vs XGBoost"
        description="XGBoost was tuned as a competing candidate (Phase 7-8) but Logistic Regression matched or beat it on nearly every metric while remaining fully interpretable — this was the basis for the final model choice."
      >
        <StateBlock loading={decision.loading} error={decision.error}>
          {decision.data && (
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm font-medium text-ink mb-2">Model A</p>
                <MetricsTable
                  metricKeys={FINAL_METRIC_KEYS}
                  columns={[
                    { key: 'lr', label: 'Logistic Regression' },
                    { key: 'xgb', label: 'XGBoost' },
                  ]}
                  rowsByColumn={{
                    lr: decision.data.model_a.logistic_regression,
                    xgb: decision.data.model_a.xgboost,
                  }}
                />
              </div>
              <div>
                <p className="text-sm font-medium text-ink mb-2">Model B</p>
                <MetricsTable
                  metricKeys={FINAL_METRIC_KEYS}
                  columns={[
                    { key: 'lr', label: 'Logistic Regression' },
                    { key: 'xgb', label: 'XGBoost' },
                  ]}
                  rowsByColumn={{
                    lr: decision.data.model_b.logistic_regression,
                    xgb: decision.data.model_b.xgboost,
                  }}
                />
              </div>
            </div>
          )}
        </StateBlock>
      </Section>

      <Section
        title="Baseline results (Phase 6)"
        description="Untuned baseline performance for all four algorithms, before hyperparameter tuning and threshold optimization. Shown for context on how much tuning improved results."
      >
        <StateBlock loading={baseline.loading} error={baseline.error} onRetry={baseline.reload}>
          {baseline.data && (
            <div className="grid md:grid-cols-2 gap-6">
              {['model_a', 'model_b'].map((key) => (
                <div key={key}>
                  <p className="text-sm font-medium text-ink mb-2">
                    {key === 'model_a' ? 'Model A' : 'Model B'} — baseline algorithms
                  </p>
                  <MetricsTable
                    metricKeys={BASELINE_METRIC_KEYS}
                    columns={Object.keys(baseline.data[key]).map((algo) => ({
                      key: algo,
                      label: algo.replaceAll('_', ' '),
                    }))}
                    rowsByColumn={baseline.data[key]}
                  />
                </div>
              ))}
            </div>
          )}
        </StateBlock>
      </Section>

      <Section
        title="ROC, precision-recall, and confusion matrices"
        description="Evaluation charts from baseline training and threshold tuning (Phases 6-7)."
      >
        <StateBlock loading={figures.loading} error={figures.error} onRetry={figures.reload}>
          <div className="grid md:grid-cols-2 gap-5">
            {perfFigures.map((f) => (
              <FigureCard key={f} filename={f} />
            ))}
          </div>
        </StateBlock>
      </Section>
    </div>
  )
}
