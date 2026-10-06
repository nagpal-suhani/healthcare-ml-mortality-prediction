import { useState, useMemo } from 'react'
import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Section } from '../components/Section'
import { StateBlock } from '../components/StateBlock'
import { RiskBadge } from '../components/RiskBadge'

const NUMERIC_FIELDS = {
  age: { label: 'Age (years)', min: 0, max: 120 },
  bmi: { label: 'BMI', min: 10, max: 80, step: 0.1 },
}
const MODEL_A_NUMERIC_FIELDS = {
  days_hospitalized: { label: 'Days hospitalized', min: 0, max: 365 },
  treatment_cost_inr: { label: 'Treatment cost (INR)', min: 1 },
}

// Categorical fields, in display order, mapped to the key returned by
// /api/dataset/valid-categories. Grouped into sections for readability.
const FIELD_GROUPS = [
  {
    title: 'Demographics',
    fields: ['gender', 'state', 'region', 'urban_rural'],
  },
  {
    title: 'Clinical profile',
    fields: ['disease_name', 'severity', 'comorbidity', 'smoking_status', 'alcohol_use', 'blood_group'],
  },
  {
    title: 'Access and care setting',
    fields: ['income_level', 'insurance_status', 'hospital_type'],
  },
]
const MODEL_A_CATEGORICAL_GROUP = { title: 'Treatment course', fields: ['treatment_type'] }

const FIELD_LABELS = {
  gender: 'Gender', state: 'State', region: 'Region', urban_rural: 'Urban / rural',
  disease_name: 'Diagnosed disease', severity: 'Severity', comorbidity: 'Comorbidity',
  smoking_status: 'Smoking status', alcohol_use: 'Alcohol use', blood_group: 'Blood group',
  income_level: 'Income level', insurance_status: 'Insurance status', hospital_type: 'Hospital type',
  treatment_type: 'Treatment type',
}

function emptyFormState(withModelA) {
  const state = { age: '', bmi: '' }
  for (const g of FIELD_GROUPS) for (const f of g.fields) state[f] = ''
  if (withModelA) {
    state.days_hospitalized = ''
    state.treatment_cost_inr = ''
    state.treatment_type = ''
  }
  return state
}

export default function Prediction() {
  const [modelKey, setModelKey] = useState('model_b')
  const isModelA = modelKey === 'model_a'

  const categories = useApi(api.getValidCategories, [])
  const symptomsList = useApi(api.getTrackedSymptoms, [])

  const [form, setForm] = useState(() => emptyFormState(false))
  const [selectedSymptoms, setSelectedSymptoms] = useState([])
  const [clientErrors, setClientErrors] = useState([])
  const [result, setResult] = useState(null)
  const [submitError, setSubmitError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const requiredFields = useMemo(() => {
    const fields = ['age', 'bmi', ...FIELD_GROUPS.flatMap((g) => g.fields)]
    if (isModelA) fields.push('days_hospitalized', 'treatment_cost_inr', 'treatment_type')
    return fields
  }, [isModelA])

  function switchModel(nextKey) {
    setModelKey(nextKey)
    setForm(emptyFormState(nextKey === 'model_a'))
    setSelectedSymptoms([])
    setResult(null)
    setSubmitError(null)
    setClientErrors([])
  }

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  function toggleSymptom(symptom) {
    setSelectedSymptoms((prev) =>
      prev.includes(symptom) ? prev.filter((s) => s !== symptom) : [...prev, symptom]
    )
  }

  function validateClientSide() {
    const errors = []
    for (const field of requiredFields) {
      if (form[field] === '' || form[field] === null || form[field] === undefined) {
        errors.push(`${FIELD_LABELS[field] || field} is required.`)
      }
    }
    const age = Number(form.age)
    if (form.age !== '' && (age < 0 || age > 120)) errors.push('Age must be between 0 and 120.')
    const bmi = Number(form.bmi)
    if (form.bmi !== '' && (bmi < 10 || bmi > 80)) errors.push('BMI must be between 10 and 80.')
    if (isModelA) {
      const cost = Number(form.treatment_cost_inr)
      if (form.treatment_cost_inr !== '' && cost <= 0) errors.push('Treatment cost must be positive.')
    }
    return errors
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setResult(null)
    setSubmitError(null)

    const errors = validateClientSide()
    setClientErrors(errors)
    if (errors.length > 0) return

    const payload = {
      ...form,
      age: Number(form.age),
      bmi: Number(form.bmi),
      symptoms: selectedSymptoms.join(', '),
    }
    if (isModelA) {
      payload.days_hospitalized = Number(form.days_hospitalized)
      payload.treatment_cost_inr = Number(form.treatment_cost_inr)
    }

    setSubmitting(true)
    try {
      const response = isModelA ? await api.predictModelA(payload) : await api.predictModelB(payload)
      setResult(response)
    } catch (err) {
      setSubmitError(err)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div>
      <Section
        title="Predict mortality risk"
        description="Enter patient information to get a statistical risk estimate. Model A uses full treatment-course information (more accurate, but requires data only available after treatment begins); Model B uses only information available at or near diagnosis."
      >
        <div className="inline-flex border border-line mb-6">
          {[
            { key: 'model_b', label: 'Model B — Early risk' },
            { key: 'model_a', label: 'Model A — Full clinical-course' },
          ].map((opt) => (
            <button
              key={opt.key}
              type="button"
              onClick={() => switchModel(opt.key)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                modelKey === opt.key ? 'bg-brand text-white' : 'bg-white text-ink-soft hover:bg-panel'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <StateBlock
          loading={categories.loading || symptomsList.loading}
          error={categories.error || symptomsList.error}
          onRetry={() => {
            categories.reload()
            symptomsList.reload()
          }}
        >
          {categories.data && symptomsList.data && (
            <form onSubmit={handleSubmit} className="grid lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 space-y-8">
                <FieldGroup title="Basic measurements">
                  <NumberInput
                    field="age"
                    config={NUMERIC_FIELDS.age}
                    value={form.age}
                    onChange={updateField}
                  />
                  <NumberInput
                    field="bmi"
                    config={NUMERIC_FIELDS.bmi}
                    value={form.bmi}
                    onChange={updateField}
                  />
                </FieldGroup>

                {FIELD_GROUPS.map((group) => (
                  <FieldGroup key={group.title} title={group.title}>
                    {group.fields.map((field) => (
                      <SelectInput
                        key={field}
                        field={field}
                        options={categories.data[field] || []}
                        value={form[field]}
                        onChange={updateField}
                      />
                    ))}
                  </FieldGroup>
                ))}

                {isModelA && (
                  <>
                    <FieldGroup title="Admission and cost">
                      <NumberInput
                        field="days_hospitalized"
                        config={MODEL_A_NUMERIC_FIELDS.days_hospitalized}
                        value={form.days_hospitalized}
                        onChange={updateField}
                      />
                      <NumberInput
                        field="treatment_cost_inr"
                        config={MODEL_A_NUMERIC_FIELDS.treatment_cost_inr}
                        value={form.treatment_cost_inr}
                        onChange={updateField}
                      />
                    </FieldGroup>
                    <FieldGroup title={MODEL_A_CATEGORICAL_GROUP.title}>
                      {MODEL_A_CATEGORICAL_GROUP.fields.map((field) => (
                        <SelectInput
                          key={field}
                          field={field}
                          options={categories.data[field] || []}
                          value={form[field]}
                          onChange={updateField}
                        />
                      ))}
                    </FieldGroup>
                  </>
                )}

                <FieldGroup title="Symptoms" description="Select all that apply.">
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 col-span-2">
                    {symptomsList.data.symptoms.map((symptom) => (
                      <label
                        key={symptom}
                        className="flex items-center gap-2 text-sm text-ink-soft border border-line px-2.5 py-1.5 cursor-pointer has-checked:border-brand has-checked:bg-brand-soft has-checked:text-brand-dark"
                      >
                        <input
                          type="checkbox"
                          checked={selectedSymptoms.includes(symptom)}
                          onChange={() => toggleSymptom(symptom)}
                          className="accent-brand"
                        />
                        {symptom}
                      </label>
                    ))}
                  </div>
                </FieldGroup>

                {clientErrors.length > 0 && (
                  <div className="border border-risk/30 bg-risk-soft px-4 py-3">
                    <p className="text-sm font-medium text-risk mb-1">Please fix the following:</p>
                    <ul className="text-sm text-risk list-disc pl-5 space-y-0.5">
                      {clientErrors.map((e) => (
                        <li key={e}>{e}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={submitting}
                  className="bg-brand text-white px-5 py-2.5 text-sm font-medium hover:bg-brand-dark disabled:opacity-60 transition-colors"
                >
                  {submitting ? 'Calculating…' : 'Calculate risk'}
                </button>
              </div>

              <div className="lg:col-span-1">
                <ResultPanel result={result} error={submitError} />
              </div>
            </form>
          )}
        </StateBlock>
      </Section>
    </div>
  )
}

function FieldGroup({ title, description, children }) {
  return (
    <fieldset>
      <legend className="text-sm font-medium text-ink mb-1">{title}</legend>
      {description && <p className="text-sm text-ink-soft mb-2">{description}</p>}
      <div className="grid sm:grid-cols-2 gap-3 mt-2">{children}</div>
    </fieldset>
  )
}

function NumberInput({ field, config, value, onChange }) {
  return (
    <label className="text-sm text-ink-soft block">
      {config.label}
      <input
        type="number"
        min={config.min}
        max={config.max}
        step={config.step || 1}
        value={value}
        onChange={(e) => onChange(field, e.target.value)}
        className="mt-1 w-full border border-line px-3 py-2 text-sm text-ink focus:border-brand"
      />
    </label>
  )
}

function SelectInput({ field, options, value, onChange }) {
  return (
    <label className="text-sm text-ink-soft block">
      {FIELD_LABELS[field] || field}
      <select
        value={value}
        onChange={(e) => onChange(field, e.target.value)}
        className="mt-1 w-full border border-line px-3 py-2 text-sm text-ink bg-white focus:border-brand"
      >
        <option value="">Select…</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  )
}

function ResultPanel({ result, error }) {
  if (error) {
    return (
      <div className="border border-risk/30 bg-risk-soft px-5 py-4 sticky top-6">
        <p className="text-sm font-medium text-risk">Prediction failed</p>
        <p className="mt-1 text-sm text-ink-soft">{error.message}</p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="border border-dashed border-line px-5 py-10 text-center sticky top-6">
        <p className="text-sm text-ink-soft">Fill in the form and calculate risk to see results here.</p>
      </div>
    )
  }

  return (
    <div className="border border-line bg-white p-5 sticky top-6 space-y-4">
      <div>
        <p className="text-sm text-ink-soft">Predicted mortality probability</p>
        <p className="text-3xl font-semibold text-ink mt-1">
          {(result.probability_of_death * 100).toFixed(1)}%
        </p>
      </div>

      <RiskBadge category={result.risk_category} />

      <div className="text-sm text-ink-soft space-y-1 border-t border-line pt-3">
        <p>Prediction: <span className="text-ink font-medium">{result.predicted_label}</span></p>
        <p>Decision threshold: <span className="text-ink font-medium">{result.threshold}</span></p>
        <p>Model: <span className="text-ink font-medium">{result.algorithm.replaceAll('_', ' ')}</span></p>
      </div>

      {result.top_contributing_factors?.length > 0 && (
        <div className="border-t border-line pt-3">
          <p className="text-sm font-medium text-ink mb-2">Top contributing factors</p>
          <ul className="space-y-1.5">
            {result.top_contributing_factors.map((f) => (
              <li key={f.feature} className="flex items-center justify-between text-sm">
                <span className="text-ink-soft">{f.feature.replaceAll('_', ' ')}</span>
                <span className={f.direction === 'increases risk' ? 'text-risk' : 'text-safe'}>
                  {f.direction === 'increases risk' ? '↑' : '↓'}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.input_warnings?.length > 0 && (
        <div className="border-t border-line pt-3">
          <p className="text-sm text-amber">{result.input_warnings.join(' ')}</p>
        </div>
      )}

      <div className="border-t border-line pt-3">
        <p className="text-xs text-ink-soft leading-relaxed">{result.disclaimer}</p>
      </div>
    </div>
  )
}
