import { useEffect, useState, useCallback } from 'react'

/**
 * Wraps an async fetcher with standard loading/error/data state, so
 * every page handles the three states (loading, error, empty) the
 * same way instead of re-implementing it per page.
 *
 * @param {() => Promise<any>} fetcher
 * @param {any[]} deps
 */
export function useApi(fetcher, deps = []) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetcher()
      .then((result) => {
        if (!cancelled) {
          setData(result)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err)
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => load(), [load])

  return { data, error, loading, reload: load }
}
