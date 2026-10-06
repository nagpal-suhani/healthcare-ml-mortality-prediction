export function Section({ title, description, children, className = '' }) {
  return (
    <section className={`mb-10 ${className}`}>
      {(title || description) && (
        <div className="mb-4">
          {title && <h2 className="text-lg font-semibold text-ink">{title}</h2>}
          {description && <p className="mt-1 text-sm text-ink-soft max-w-2xl">{description}</p>}
        </div>
      )}
      {children}
    </section>
  )
}
