import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'
import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Section } from '../components/Section'
import { StatCard } from '../components/StatCard'
import { StateBlock } from '../components/StateBlock'

const TARGET_COLORS = ['#1d4e6d', '#b4472b']

export default function Dashboard() {
  const overview = useApi(api.getDatasetOverview, [])
  const summary = useApi(api.getModelSummary, [])

  const targetData = overview.data
    ? [
        { name: 'Survived / ongoing care', value: overview.data.target_distribution['0'] },
        { name: 'Deceased', value: overview.data.target_distribution['1'] },
      ]
    : []

  return (
    <div>
      <Section
        title="Dataset overview"
        description="Summary statistics for the underlying patient dataset used to train both models."
      >
        <StateBlock loading={overview.loading} error={overview.error} onRetry={overview.reload}>
          {overview.data && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                label="Total patient records"
                value={overview.data.total_records.toLocaleString()}
                accent="brand"
              />
              <StatCard label="Columns retained" value={overview.data.total_columns} />
              <StatCard
                label="Deceased in dataset"
                value={`${overview.data.target_distribution_pct['1']}%`}
                sublabel={`${overview.data.target_distribution['1'].toLocaleString()} of ${overview.data.total_records.toLocaleString()} records`}
                accent="risk"
              />
              <StatCard
                label="Missing values"
                value={overview.data.missing_values_total}
                sublabel="after cleaning"
                accent="safe"
              />
            </div>
          )}
        </StateBlock>
      </Section>

      <Section
        title="Target distribution"
        description="death_flag is heavily imbalanced — about 9 in 100 recorded patients are deceased. This is the reason both models are evaluated on recall and PR-AUC, not accuracy alone."
      >
        <StateBlock loading={overview.loading} error={overview.error} onRetry={overview.reload}>
          {overview.data && (
            <div className="border border-line bg-white p-4 max-w-md">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={targetData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                    {targetData.map((_, i) => (
                      <Cell key={i} fill={TARGET_COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v) => v.toLocaleString()} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </StateBlock>
      </Section>

      <Section
        title="Final model summary"
        description="Both models use Logistic Regression at a decision threshold of 0.40, chosen to catch more at-risk patients (higher recall) at a deliberate cost to precision — appropriate for a mortality screening tool."
      >
        <StateBlock loading={summary.loading} error={summary.error} onRetry={summary.reload}>
          {summary.data && (
            <div className="grid md:grid-cols-2 gap-4">
              <ModelSummaryCard modelKey="A" data={summary.data.model_a} />
              <ModelSummaryCard modelKey="B" data={summary.data.model_b} />
            </div>
          )}
        </StateBlock>
      </Section>

      <Section title="Key insights">
        <ul className="space-y-2 text-sm text-ink-soft max-w-2xl list-disc pl-5">
          <li>Severity and age are the strongest, most consistent predictors of mortality risk across both models.</li>
          <li>Model A (full clinical-course) outperforms Model B (early-risk) on every metric, quantifying the value of treatment-course information — but Model B remains useful as a genuine admission-time risk tool.</li>
          <li>Logistic Regression was selected over XGBoost for both models: nearly identical or better performance, with full coefficient-level interpretability appropriate for a healthcare setting.</li>
        </ul>
      </Section>
    </div>
  )
}

function ModelSummaryCard({ modelKey, data }) {
  const m = data.metrics
  return (
    <div className="border border-line bg-white p-5">
      <p className="text-sm font-medium text-ink">Model {modelKey}</p>
      <p className="mt-1 text-sm text-ink-soft">{data.description}</p>
      <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div>
          <dt className="text-ink-soft">Recall</dt>
          <dd className="font-semibold text-ink">{(m.recall_sensitivity * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-ink-soft">ROC-AUC</dt>
          <dd className="font-semibold text-ink">{(m.roc_auc * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-ink-soft">Threshold</dt>
          <dd className="font-semibold text-ink">{data.threshold}</dd>
        </div>
      </dl>
    </div>
  )
}
