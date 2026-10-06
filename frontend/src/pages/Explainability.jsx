import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Section } from '../components/Section'
import { StateBlock } from '../components/StateBlock'
import { FigureCard } from '../components/FigureCard'

export default function Explainability() {
  const explain = useApi(api.getExplainability, [])
  const figures = useApi(api.getEdaFigures, [])

  const shapFigures = (figures.data?.figures || []).filter((f) => /^2[2-7]/.test(f))

  return (
    <div>
      <Section
        title="What drives each prediction"
        description="For XGBoost, feature importance comes from SHAP values. For Logistic Regression (the final model for both A and B), it comes directly from the model's coefficients — no separate explainability step is needed, since the coefficients ARE the explanation."
      >
        <StateBlock loading={explain.loading} error={explain.error} onRetry={explain.reload}>
          {explain.data && (
            <div className="grid md:grid-cols-2 gap-6">
              {['model_a', 'model_b'].map((key) => (
                <FindingsCard key={key} label={key === 'model_a' ? 'Model A' : 'Model B'} data={explain.data[key]} />
              ))}
            </div>
          )}
        </StateBlock>
      </Section>

      <Section
        title="In simple terms"
        description="How to read the charts below."
      >
        <ul className="space-y-2 text-sm text-ink-soft max-w-2xl list-disc pl-5">
          <li><strong className="text-ink font-medium">SHAP summary plots</strong> show, for each feature, whether a high or low value of that feature pushes a prediction toward higher or lower risk — dots further from center matter more.</li>
          <li><strong className="text-ink font-medium">Coefficient charts</strong> show the same idea for Logistic Regression: bars pointing right (red) increase predicted risk, bars pointing left (blue) decrease it.</li>
          <li>Across both approaches, <strong className="text-ink font-medium">severity and age</strong> are consistently the strongest signals — this agreement across two very different modeling approaches is a good sign the pattern is real, not an artifact.</li>
        </ul>
      </Section>

      <Section title="SHAP and coefficient charts">
        <StateBlock loading={figures.loading} error={figures.error} onRetry={figures.reload}>
          <div className="grid md:grid-cols-2 gap-5">
            {shapFigures.map((f) => (
              <FigureCard key={f} filename={f} />
            ))}
          </div>
        </StateBlock>
      </Section>
    </div>
  )
}

function FindingsCard({ label, data }) {
  return (
    <div className="border border-line bg-white p-5">
      <p className="text-sm font-medium text-ink mb-3">{label}</p>

      <p className="text-sm text-ink-soft mb-1">Top features (XGBoost, SHAP)</p>
      <TagList items={data.top10_xgboost_shap} tone="neutral" />

      <p className="text-sm text-ink-soft mt-4 mb-1">Increases risk (Logistic Regression)</p>
      <TagList items={data.top10_lr_positive} tone="risk" />

      <p className="text-sm text-ink-soft mt-4 mb-1">Decreases risk (Logistic Regression)</p>
      <TagList items={data.top10_lr_negative} tone="safe" />

      {data.early_availability_flagged?.length > 0 ? (
        <p className="mt-4 text-sm text-risk">
          ⚠ Post-admission features found among top importances: {data.early_availability_flagged.join(', ')}
        </p>
      ) : (
        label === 'Model B' && (
          <p className="mt-4 text-sm text-safe">
            ✓ No post-admission information found among top important features — confirmed early-risk safe.
          </p>
        )
      )}
    </div>
  )
}

const TAG_TONES = {
  neutral: 'bg-panel text-ink-soft',
  risk: 'bg-risk-soft text-risk',
  safe: 'bg-safe-soft text-safe',
}

function TagList({ items, tone }) {
  if (!items || items.length === 0) return <p className="text-sm text-ink-soft">None</p>
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.slice(0, 6).map((item) => (
        <span key={item} className={`rounded px-2 py-0.5 text-xs ${TAG_TONES[tone]}`}>
          {item}
        </span>
      ))}
    </div>
  )
}
