import { api } from '../api/client'

/**
 * Displays one chart image served by the backend's /static/figures/
 * mount, with a human-readable caption derived from the filename.
 */
function captionFromFilename(filename) {
  return filename
    .replace(/\.png$/, '')
    .replace(/^\d+b?_/, '')
    .replace(/_model_(a|b)/, (m, x) => ` — Model ${x.toUpperCase()}`)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export function FigureCard({ filename }) {
  return (
    <figure className="border border-line bg-white">
      <img
        src={api.figureUrl(filename)}
        alt={captionFromFilename(filename)}
        loading="lazy"
        className="w-full h-auto"
      />
      <figcaption className="border-t border-line px-3 py-2 text-sm text-ink-soft">
        {captionFromFilename(filename)}
      </figcaption>
    </figure>
  )
}
