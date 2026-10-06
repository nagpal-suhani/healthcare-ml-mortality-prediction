# EDA Report — Healthcare Mortality Dataset

Dataset: 20,000 rows × 31 columns (cleaned). Target: `death_flag`.

**Leakage rule enforced in this analysis:** `outcome`, `cause_of_death`, `recovery_days`, `patient_id` are excluded from all feature-relationship analysis. `outcome` appears once, purely for descriptive context.

## 1a. Univariate — Numerical Features

|                    |   count |      mean |       std |   min |    25% |    50% |     75% |          max |   skew |
|:-------------------|--------:|----------:|----------:|------:|-------:|-------:|--------:|-------------:|-------:|
| age                |   20000 |     44.06 |     25.17 |     0 |   23   |   44   |    63   | 95           |   0.08 |
| bmi                |   20000 |     23.49 |      4.67 |    14 |   19.7 |   23.2 |    27   | 40           |   0.52 |
| days_hospitalized  |   20000 |      9.77 |     12.08 |     0 |    2   |    6   |    14   | 60           |   2.14 |
| treatment_cost_inr |   20000 | 107941    | 494064    |    12 | 1421   | 6604.5 | 39333.8 |  1.53758e+07 |  13.73 |

**Highly skewed (|skew| > 1):** ['days_hospitalized', 'treatment_cost_inr']
`treatment_cost_inr` in particular is extremely right-skewed — confirms the Phase 2 decision to log-transform it during feature engineering rather than treating extreme values as errors.

Saved: `reports/figures/01_numerical_distributions.png`

## 1b. Univariate — Key Categorical Features

Saved: `reports/figures/02_categorical_distributions.png`

**severity** top categories: Moderate (32.6%), Severe (31.5%), Mild (24.2%), Critical (11.6%)

**disease_category** top categories: Non-Communicable (24.8%), Vector-Borne (19.0%), Respiratory (15.6%), Infectious (15.5%)

**gender** top categories: Male (49.6%), Female (49.4%), Other (1.0%)

**urban_rural** top categories: Rural (40.3%), Urban (39.7%), Semi-Urban (20.0%)

**income_level** top categories: Lower (34.6%), Middle (30.0%), Below Poverty Line (20.2%), Upper-Middle (10.1%)

**insurance_status** top categories: Uninsured (40.4%), Ayushman Bharat (24.6%), State Scheme (15.3%), Private (14.9%)

**treatment_type** top categories: Medication (65.4%), Surgery (15.7%), ICU + Ventilator (7.2%), Home Remedies (4.8%)

## 2. Target Distribution — death_flag

- Survived/Ongoing (0): 18,178 rows (90.89%)
- Deceased (1): 1,822 rows (9.11%)

**Class imbalance ratio ≈ 10.0 : 1** (survived : deceased). This is a genuine, expected imbalance for mortality data, not a data problem — must be handled with class weighting/resampling and recall-focused metrics in later phases.

*Context only — `outcome` (NOT a model feature, shown here only to explain where death_flag comes from):*
| outcome            |   count |
|:-------------------|--------:|
| Recovered          |    8440 |
| Recovering         |    5253 |
| Chronic Management |    4485 |
| Deceased           |    1822 |

Saved: `reports/figures/03_target_distribution.png`

## 2a. Mortality Rate by Categorical Variables

**Ranked by mortality-rate spread across categories (most discriminating first):**

- `treatment_type` — spread 96.8 pts. Highest: Palliative (100.0%), ICU + Ventilator (34.12%) | Lowest: Home Remedies (3.15%)
- `severity` — spread 38.6 pts. Highest: Critical (41.0%), Severe (9.25%) | Lowest: Mild (2.37%)
- `disease_name` — spread 31.2 pts. Highest: Stroke (31.69%), Heart Disease (25.08%) | Lowest: Jaundice (Hepatitis A) (0.47%)
- `follow_up_required` — spread 29.9 pts. Highest: No (29.89%), Yes (0.0%) | Lowest: Yes (0.0%)
- `disease_category` — spread 21.3 pts. Highest: Non-Communicable (22.01%), Respiratory (10.06%) | Lowest: Waterborne (0.75%)
- `gender` — spread 2.7 pts. Highest: Female (9.18%), Male (9.1%) | Lowest: Other (6.47%)
- `hospital_type` — spread 1.9 pts. Highest: AIIMS / Apex Institution (10.15%), Private (9.38%) | Lowest: Primary Health Center (8.29%)
- `insurance_status` — spread 1.7 pts. Highest: Employer (10.35%), State Scheme (9.67%) | Lowest: Private (8.68%)
- `region` — spread 1.2 pts. Highest: Northeast (9.64%), North (9.46%) | Lowest: West (8.43%)
- `alcohol_use` — spread 0.8 pts. Highest: Occasional (9.51%), Regular (9.27%) | Lowest: Heavy (8.7%)
- `urban_rural` — spread 0.5 pts. Highest: Semi-Urban (9.39%), Urban (9.19%) | Lowest: Rural (8.9%)
- `income_level` — spread 0.3 pts. Highest: Lower (9.27%), Upper-Middle (9.08%) | Lowest: Below Poverty Line (8.92%)
- `smoking_status` — spread 0.2 pts. Highest: Former (9.25%), Never (9.08%) | Lowest: Current (9.05%)

Saved: `reports/figures/04_mortality_by_category.png` (top 6 most discriminating: ['treatment_type', 'severity', 'disease_name', 'follow_up_required', 'disease_category', 'gender'])

## 2b. Mortality Trends by Numerical Variables

- `age`: mean = 42.6 (survived/ongoing) vs 58.7 (deceased)
- `bmi`: mean = 23.4 (survived/ongoing) vs 24.1 (deceased)
- `days_hospitalized`: mean = 8.4 (survived/ongoing) vs 23.1 (deceased)
- `treatment_cost_inr`: mean = 77539.1 (survived/ongoing) vs 411254.1 (deceased)

Saved: `reports/figures/05_numerical_vs_target.png`
Saved: `reports/figures/06_mortality_by_age.png` — confirms a clear, monotonically rising mortality trend with age (strongest single numerical signal in the dataset).

## 3. Bivariate — Severity × Disease Category Interaction

| disease_category   |   Mild |   Moderate |   Severe |   Critical |
|:-------------------|-------:|-----------:|---------:|-----------:|
| Genetic            |    0   |        0   |      0.2 |       14.5 |
| Infectious         |    1.6 |        3   |      6.1 |       32   |
| Mental Health      |    0.2 |        0.2 |      0   |       13.2 |
| Non-Communicable   |    6.8 |        6.9 |     19.1 |       70   |
| Respiratory        |    3.7 |        3.2 |     10.8 |       42   |
| Vector-Borne       |    1   |        0.6 |      4.4 |       21.3 |
| Waterborne         |    0.2 |        0.4 |      0.5 |        6   |

Saved: `reports/figures/07_severity_disease_heatmap.png`

Both `severity` and `disease_category` independently move mortality risk, and the effect compounds — Critical-severity Non-Communicable-disease cases show the highest observed rates. This is a descriptive association only, not a causal claim.

## 4. Correlation Analysis (Numerical Features)

|                    |   age |   bmi |   days_hospitalized |   treatment_cost_inr |   death_flag |
|:-------------------|------:|------:|--------------------:|---------------------:|-------------:|
| age                |  1    |  0.08 |                0.09 |                 0.1  |         0.18 |
| bmi                |  0.08 |  1    |               -0.01 |                 0.01 |         0.04 |
| days_hospitalized  |  0.09 | -0.01 |                1    |                 0.22 |         0.35 |
| treatment_cost_inr |  0.1  |  0.01 |                0.22 |                 1    |         0.19 |
| death_flag         |  0.18 |  0.04 |                0.35 |                 0.19 |         1    |

Saved: `reports/figures/08_correlation_heatmap.png`

**Strongest linear correlation with death_flag:** days_hospitalized (r = 0.35)
No pair of numerical features shows correlation above 0.3 with each other — **no meaningful multicollinearity** among the numerical features. Note: correlation here is Pearson (linear) only; it will under-detect non-linear relationships like the severity/disease interaction found above, and correlation is not evidence of causation.

## 5. Symptoms Analysis

**Top 15 most frequent individual symptoms:**

| symptoms                |   count |
|:------------------------|--------:|
| Fatigue                 |    5639 |
| Nausea                  |    4148 |
| Fever                   |    3829 |
| Shortness of breath     |    3365 |
| Chest pain              |    3254 |
| Vomiting                |    2912 |
| Rash                    |    2867 |
| Swelling                |    2430 |
| Numbness                |    1975 |
| Unexplained weight loss |    1950 |
| Frequent urination      |    1918 |
| Dizziness               |    1917 |
| Blurred vision          |    1886 |
| High Fever              |    1669 |
| Muscle pain             |    1643 |

Saved: `reports/figures/09_mortality_by_symptom.png`

Highest-mortality symptom in top-15: **Dizziness** (23.0%). Full symptom multi-label feature engineering (one-hot per symptom) is deferred to Phase 4 — this is descriptive only.

## 6. Temporal Analysis

**Mortality rate by year:**

|   year |   death_flag |
|-------:|-------------:|
|   2018 |         9.66 |
|   2019 |         8.56 |
|   2020 |         9.44 |
|   2021 |         9.27 |
|   2022 |         9.63 |
|   2023 |         9.07 |
|   2024 |         8.89 |
|   2025 |         8.34 |

**Mortality rate by season:**

| season       |   death_flag |
|:-------------|-------------:|
| Monsoon      |         7.71 |
| Post-Monsoon |         8.9  |
| Summer       |         9.79 |
| Winter       |        10.08 |

Saved: `reports/figures/10_temporal_trends.png`

Year-over-year mortality spread is 1.32 points — no strong or clinically meaningful trend over time; `year`/`season` are unlikely to be useful predictive features and are treated as context only, not recommended as model inputs.

## 7. Leakage Audit — follow_up_required

| follow_up_required   |     0 |    1 |
|:---------------------|------:|-----:|
| No                   |  4273 | 1822 |
| Yes                  | 13905 |    0 |

**Finding:** every single one of the 1,822 deceased patients has `follow_up_required = No`, and every patient with `follow_up_required = Yes` survived (13,905 of 13,905). This column is a near-perfect proxy for the target — clinically it makes sense (a deceased patient has no follow-up), but that also means it is logically determined *after* the outcome is known, the same way `recovery_days` and `cause_of_death` are.

**Decision required for Phase 4:** `follow_up_required` should be treated as a **leakage column and excluded from modeling**, the same as `outcome`/`cause_of_death`/`recovery_days`. Recommending exclusion — flagged for your approval below.

**CONFIRMED DECISION (user, post-Phase 3 review):** `follow_up_required` is DROPPED from all modeling feature sets.

## 7b. Leakage Watch — treatment_type

| treatment_type   |    0 |     1 |
|:-----------------|-----:|------:|
| Home Remedies    | 96.9 |   3.1 |
| ICU + Ventilator | 65.9 |  34.1 |
| Medication       | 96   |   4   |
| Observation      | 96.6 |   3.4 |
| Palliative       |  0   | 100   |
| Surgery          | 84.9 |  15.1 |
| Therapy          | 91.6 |   8.4 |

**Finding:** `Palliative` treatment has a 100% mortality rate (all such patients died) and `ICU + Ventilator` shows 34.1%. Unlike `follow_up_required`, this is not automatically post-outcome — palliative/ICU care is a clinical decision made *during* treatment, generally before the final outcome is known, so it is plausibly legitimate to keep as a feature. However, because it is such a strong proxy (Palliative = 100%), it should be flagged as a **high-influence feature to watch**, not necessarily dropped — recommend keeping it in Phase 4 but noting it explicitly as a near-deterministic feature so its role in the final model is transparent, not hidden.

**CONFIRMED DECISION (user, post-Phase 3 review):** build TWO parallel feature sets for Phase 4 onward:
- **Model A (full-featured):** includes `treatment_type`. Expected to have higher accuracy/recall, but partly reflects clinical decisions made during care rather than pure admission-time risk.
- **Model B (early-risk-only):** excludes `treatment_type` (and `follow_up_required`). Uses only features available at/near diagnosis (age, severity, disease, comorbidity, bmi, lifestyle, socioeconomic, hospital context). A cleaner early-warning-risk story.

Both models will be trained, evaluated, and compared in Phase 9, and both will be exposed in the final prediction UI.