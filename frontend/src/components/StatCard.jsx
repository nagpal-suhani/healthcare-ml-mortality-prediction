/**
 * A single key-metric card with a left accent bar (not a uniform
 * shadow card) — accent color communicates meaning (neutral/brand/
 * risk/safe), not decoration.
 */
const ACCENTS = {
  brand: 'border-brand',
  risk: 'border-risk',
  safe: 'border-safe',
  neutral: 'border-line',
}

export function StatCard({ label, value, sublabel, accent = 'neutral' }) {
  return (
    <div className={`border-l-4 ${ACCENTS[accent]} bg-white pl-4 py-3 pr-3`}>
      <p className="text-sm text-ink-soft">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-ink">{value}</p>
      {sublabel && <p className="mt-0.5 text-xs text-ink-soft">{sublabel}</p>}
    </div>
  )
}
