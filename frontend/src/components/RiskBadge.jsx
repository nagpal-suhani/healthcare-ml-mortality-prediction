const STYLES = {
  'Low Risk': 'bg-safe-soft text-safe',
  'Moderate Risk': 'bg-amber-soft text-amber',
  'High Risk': 'bg-risk-soft text-risk',
  'Critical Risk': 'bg-risk text-white',
}

export function RiskBadge({ category }) {
  const cls = STYLES[category] || 'bg-panel text-ink-soft'
  return (
    <span className={`inline-flex items-center rounded px-2.5 py-1 text-sm font-medium ${cls}`}>
      {category}
    </span>
  )
}
