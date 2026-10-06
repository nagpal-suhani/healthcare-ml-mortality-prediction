const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall (sensitivity)',
  recall_sensitivity: 'Recall (sensitivity)',
  specificity: 'Specificity',
  f1: 'F1 score',
  roc_auc: 'ROC-AUC',
  pr_auc: 'PR-AUC',
  false_negative: 'False negatives',
  false_positive: 'False positives',
  true_positive: 'True positives',
  true_negative: 'True negatives',
}

const PERCENT_METRICS = new Set([
  'accuracy', 'precision', 'recall', 'recall_sensitivity', 'specificity', 'f1', 'roc_auc', 'pr_auc',
])

function formatValue(key, value) {
  if (value === undefined || value === null) return '—'
  if (PERCENT_METRICS.has(key)) return `${(value * 100).toFixed(1)}%`
  return value
}

/**
 * columns: [{ key, label }]
 * rows: [{ key, ...values per column key }]
 */
export function MetricsTable({ metricKeys, columns, rowsByColumn }) {
  return (
    <div className="overflow-x-auto border border-line">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-line bg-panel text-left">
            <th className="px-4 py-2.5 font-medium text-ink-soft">Metric</th>
            {columns.map((col) => (
              <th key={col.key} className="px-4 py-2.5 font-medium text-ink-soft">
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metricKeys.map((metricKey, i) => (
            <tr key={metricKey} className={i !== metricKeys.length - 1 ? 'border-b border-line' : ''}>
              <td className="px-4 py-2.5 text-ink-soft">{METRIC_LABELS[metricKey] || metricKey}</td>
              {columns.map((col) => (
                <td key={col.key} className="px-4 py-2.5 font-medium text-ink">
                  {formatValue(metricKey, rowsByColumn[col.key]?.[metricKey])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
