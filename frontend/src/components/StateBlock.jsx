/**
 * Renders one of: loading, error, empty, or the actual content —
 * so every page handles these states identically instead of each
 * page inventing its own spinner/error copy.
 */
export function StateBlock({ loading, error, empty, emptyMessage, onRetry, children }) {
  if (loading) {
    return (
      <div className="flex items-center gap-3 py-16 justify-center text-ink-soft">
        <span
          className="h-4 w-4 rounded-full border-2 border-line border-t-brand animate-spin"
          aria-hidden="true"
        />
        <span className="text-sm">Loading…</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-md border border-risk/30 bg-risk-soft px-5 py-4">
        <p className="text-sm font-medium text-risk">Couldn't load this data</p>
        <p className="mt-1 text-sm text-ink-soft">{error.message}</p>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-3 text-sm font-medium text-brand hover:text-brand-dark underline underline-offset-2"
          >
            Try again
          </button>
        )}
      </div>
    )
  }

  if (empty) {
    return (
      <div className="rounded-md border border-dashed border-line px-5 py-10 text-center">
        <p className="text-sm text-ink-soft">{emptyMessage || 'Nothing to show yet.'}</p>
      </div>
    )
  }

  return children
}
