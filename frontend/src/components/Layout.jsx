import { NavLink, useLocation } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/data-analysis', label: 'Data analysis' },
  { to: '/model-performance', label: 'Model performance' },
  { to: '/explainability', label: 'Explainability' },
  { to: '/prediction', label: 'Prediction' },
  { to: '/dataset', label: 'Dataset' },
]

const PAGE_TITLES = {
  '/': 'Dashboard',
  '/data-analysis': 'Data analysis',
  '/model-performance': 'Model performance',
  '/explainability': 'Explainability',
  '/prediction': 'Prediction',
  '/dataset': 'Dataset',
}

export function Layout({ children }) {
  const { pathname } = useLocation()
  const title = PAGE_TITLES[pathname] || 'Healthcare Mortality Risk'

  return (
    <div className="min-h-screen bg-white flex">
      <aside className="w-60 shrink-0 border-r border-line bg-panel flex flex-col">
        <div className="px-5 py-5 border-b border-line">
          <p className="text-base font-semibold text-ink leading-tight">Mortality Risk</p>
          <p className="text-sm text-ink-soft leading-tight">Analytics &amp; prediction</p>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `block rounded px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? 'bg-brand text-white font-medium'
                    : 'text-ink-soft hover:bg-brand-soft hover:text-brand-dark'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-5 py-4 border-t border-line">
          <p className="text-xs text-ink-soft leading-relaxed">
            Model outputs are statistical estimates, not medical diagnoses.
          </p>
        </div>
      </aside>

      <div className="flex-1 min-w-0">
        <header className="border-b border-line px-8 py-4">
          <h1 className="text-lg font-semibold text-ink">{title}</h1>
        </header>
        <main className="px-8 py-6 max-w-6xl">{children}</main>
      </div>
    </div>
  )
}
