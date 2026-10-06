import { useState } from 'react'
import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Section } from '../components/Section'
import { StateBlock } from '../components/StateBlock'

const PAGE_SIZE = 15

export default function Dataset() {
  const columns = useApi(api.getDatasetColumns, [])
  const missing = useApi(api.getMissingValues, [])

  const [page, setPage] = useState(0)
  const [searchColumn, setSearchColumn] = useState('')
  const [searchValue, setSearchValue] = useState('')
  const [searchResult, setSearchResult] = useState(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState(null)

  const preview = useApi(api.getDatasetPreview.bind(null, PAGE_SIZE, page * PAGE_SIZE), [page])

  const activeRecords = searchResult ? searchResult.records : preview.data?.records
  const columnNames = columns.data?.map((c) => c.name) || []

  async function handleSearch(e) {
    e.preventDefault()
    if (!searchColumn || !searchValue) {
      setSearchResult(null)
      return
    }
    setSearchLoading(true)
    setSearchError(null)
    try {
      const result = await api.searchDataset(searchColumn, searchValue, 50)
      setSearchResult(result)
    } catch (err) {
      setSearchError(err)
    } finally {
      setSearchLoading(false)
    }
  }

  function clearSearch() {
    setSearchColumn('')
    setSearchValue('')
    setSearchResult(null)
    setSearchError(null)
  }

  return (
    <div>
      <Section
        title="Columns"
        description="Every field below is safe for display — patient IDs and outcome-leakage columns (outcome, cause_of_death, recovery_days, follow_up_required) are excluded by the backend and never reach this page."
      >
        <StateBlock loading={columns.loading} error={columns.error} onRetry={columns.reload}>
          {columns.data && (
            <div className="overflow-x-auto border border-line max-h-80 overflow-y-auto">
              <table className="w-full text-sm border-collapse">
                <thead className="sticky top-0 bg-panel">
                  <tr className="border-b border-line text-left">
                    <th className="px-4 py-2 font-medium text-ink-soft">Column</th>
                    <th className="px-4 py-2 font-medium text-ink-soft">Type</th>
                    <th className="px-4 py-2 font-medium text-ink-soft">Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.data.map((col, i) => (
                    <tr key={col.name} className={i !== columns.data.length - 1 ? 'border-b border-line' : ''}>
                      <td className="px-4 py-2 text-ink font-medium">{col.name}</td>
                      <td className="px-4 py-2 text-ink-soft">{col.dtype}</td>
                      <td className="px-4 py-2 text-ink-soft">
                        {col.unique_values !== undefined
                          ? `${col.unique_values} unique values`
                          : `range ${col.min} – ${col.max}`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </StateBlock>
      </Section>

      <Section title="Missing values" description="Should be empty — the dataset was fully cleaned in Phase 2.">
        <StateBlock loading={missing.loading} error={missing.error} onRetry={missing.reload}>
          {missing.data && missing.data.length === 0 && (
            <p className="text-sm text-safe">No missing values in any displayed column.</p>
          )}
          {missing.data && missing.data.length > 0 && (
            <ul className="text-sm text-ink-soft list-disc pl-5">
              {missing.data.map((m) => (
                <li key={m.column}>
                  {m.column}: {m.missing_count} ({m.missing_pct}%)
                </li>
              ))}
            </ul>
          )}
        </StateBlock>
      </Section>

      <Section title="Search records" description="Search for a value within a specific column.">
        <form onSubmit={handleSearch} className="flex flex-wrap gap-3 mb-4">
          <select
            value={searchColumn}
            onChange={(e) => setSearchColumn(e.target.value)}
            className="border border-line px-3 py-2 text-sm bg-white"
          >
            <option value="">Choose column…</option>
            {columnNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Value to search for"
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            className="border border-line px-3 py-2 text-sm flex-1 min-w-[200px]"
          />
          <button
            type="submit"
            className="bg-brand text-white px-4 py-2 text-sm font-medium hover:bg-brand-dark"
          >
            Search
          </button>
          {searchResult && (
            <button
              type="button"
              onClick={clearSearch}
              className="border border-line px-4 py-2 text-sm text-ink-soft hover:bg-panel"
            >
              Clear
            </button>
          )}
        </form>
        {searchResult && (
          <p className="text-sm text-ink-soft mb-3">
            {searchResult.matched_count} record(s) matched — showing up to 50.
          </p>
        )}
        {searchError && <p className="text-sm text-risk mb-3">{searchError.message}</p>}
      </Section>

      <Section title={searchResult ? 'Search results' : 'Dataset preview'}>
        <StateBlock
          loading={!searchResult && preview.loading}
          error={!searchResult && preview.error}
          onRetry={preview.reload}
          empty={activeRecords && activeRecords.length === 0}
          emptyMessage="No records found."
        >
          {activeRecords && activeRecords.length > 0 && (
            <>
              <div className="overflow-x-auto border border-line">
                <table className="w-full text-sm border-collapse whitespace-nowrap">
                  <thead>
                    <tr className="border-b border-line bg-panel text-left">
                      {Object.keys(activeRecords[0]).map((key) => (
                        <th key={key} className="px-3 py-2 font-medium text-ink-soft">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {activeRecords.map((record, i) => (
                      <tr key={i} className="border-b border-line last:border-b-0">
                        {Object.values(record).map((val, j) => (
                          <td key={j} className="px-3 py-2 text-ink-soft">
                            {val === null ? '—' : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {!searchResult && preview.data && (
                <div className="flex items-center justify-between mt-4 text-sm">
                  <span className="text-ink-soft">
                    Records {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, preview.data.total_records)} of{' '}
                    {preview.data.total_records.toLocaleString()}
                  </span>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      disabled={page === 0}
                      onClick={() => setPage((p) => Math.max(0, p - 1))}
                      className="border border-line px-3 py-1.5 text-ink-soft disabled:opacity-40 hover:bg-panel"
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      disabled={(page + 1) * PAGE_SIZE >= preview.data.total_records}
                      onClick={() => setPage((p) => p + 1)}
                      className="border border-line px-3 py-1.5 text-ink-soft disabled:opacity-40 hover:bg-panel"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </StateBlock>
        {searchLoading && <p className="text-sm text-ink-soft mt-3">Searching…</p>}
      </Section>
    </div>
  )
}
