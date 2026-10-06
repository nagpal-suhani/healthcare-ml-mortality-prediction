import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Section } from '../components/Section'
import { StateBlock } from '../components/StateBlock'
import { FigureCard } from '../components/FigureCard'

// Curated order + grouping of the 10 EDA figures (Phase 3), by filename
// prefix, so related charts sit together rather than in raw glob order.
const EDA_GROUPS = [
  {
    title: 'Distributions',
    description: 'How individual numerical and categorical variables are distributed across the dataset.',
    match: (f) => /^0[12]_/.test(f),
  },
  {
    title: 'Target and mortality patterns',
    description: 'How the death_flag target is distributed, and how mortality rate varies across patient groups.',
    match: (f) => /^0[3-6]_/.test(f),
  },
  {
    title: 'Interactions and correlation',
    description: 'How severity and disease category compound, and how numerical features correlate with each other.',
    match: (f) => /^0[78]_/.test(f),
  },
  {
    title: 'Symptoms and time',
    description: 'Which symptoms carry the most mortality signal, and whether mortality varies by year or season.',
    match: (f) => /^(09|10)_/.test(f),
  },
]

export default function DataAnalysis() {
  const overview = useApi(api.getDatasetOverview, [])
  const columns = useApi(api.getDatasetColumns, [])
  const figures = useApi(api.getEdaFigures, [])

  const edaFigures = (figures.data?.figures || []).filter((f) => /^\d{2}_/.test(f))

  return (
    <div>
      <Section
        title="Dataset statistics"
        description="Column-level summary of the featured dataset (post-cleaning, post-feature-engineering)."
      >
        <StateBlock loading={overview.loading || columns.loading} error={overview.error || columns.error}>
          {overview.data && columns.data && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <StatBox label="Records" value={overview.data.total_records.toLocaleString()} />
              <StatBox label="Columns" value={columns.data.length} />
              <StatBox label="Earliest diagnosis" value={overview.data.date_range.earliest} />
              <StatBox label="Latest diagnosis" value={overview.data.date_range.latest} />
            </div>
          )}
        </StateBlock>
      </Section>

      {EDA_GROUPS.map((group) => {
        const groupFigures = edaFigures.filter(group.match)
        if (groupFigures.length === 0) return null
        return (
          <Section key={group.title} title={group.title} description={group.description}>
            <StateBlock loading={figures.loading} error={figures.error} onRetry={figures.reload}>
              <div className="grid md:grid-cols-2 gap-5">
                {groupFigures.map((f) => (
                  <FigureCard key={f} filename={f} />
                ))}
              </div>
            </StateBlock>
          </Section>
        )
      })}
    </div>
  )
}

function StatBox({ label, value }) {
  return (
    <div className="border border-line bg-white px-4 py-3">
      <p className="text-sm text-ink-soft">{label}</p>
      <p className="mt-1 text-lg font-semibold text-ink">{value}</p>
    </div>
  )
}
