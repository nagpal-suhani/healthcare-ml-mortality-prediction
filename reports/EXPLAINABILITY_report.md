# Explainability Report — Phase 8


## Model A (Full Clinical-Course)

**Top 10 features (XGBoost, SHAP mean |value|):** ['severity_Critical', 'age', 'severity_Severe', 'treatment_cost_log', 'days_hospitalized', 'treatment_type_Medication', 'treatment_type_Palliative', 'symptom_chest_pain', 'symptom_shortness_of_breath', 'disease_name_Depression']
**Top risk-increasing features (LR, positive coefficient):** ['treatment_type_Palliative', 'severity_Critical', 'disease_name_Hypertension', 'disease_name_Heart Disease', 'disease_name_Stroke', 'disease_name_Diabetes (Type 2)', 'disease_name_Breast Cancer', 'disease_name_Oral Cancer', 'disease_name_Chronic Kidney Disease', 'age_risk_group_Elderly']
**Top risk-decreasing features (LR, negative coefficient):** ['disease_name_Jaundice (Hepatitis A)', 'disease_name_Depression', 'disease_name_Sickle Cell Anemia', 'disease_name_Thalassemia', 'disease_name_Anxiety Disorder', 'disease_name_Cholera', 'state_Mizoram', 'disease_name_Gastroenteritis', 'disease_name_Diarrhea', 'severity_Mild']

**Feature agreement:** 2/10 features appear in both models' top 10 — ['disease_name_Depression', 'treatment_type_Palliative']

**Test comparison — Model A (Full Clinical-Course):**

| Metric | XGBoost | Logistic Regression |
|---|---|---|
| Accuracy | 0.8047 | 0.8367 |
| Precision | 0.2892 | 0.3253 |
| Recall | 0.7857 | 0.739 |
| Specificity | 0.8067 | 0.8465 |
| F1 | 0.4228 | 0.4517 |
| ROC-AUC | 0.8811 | 0.8861 |
| PR-AUC | 0.6134 | 0.6143 |
| False Negatives | 78 | 95 |
| False Positives | 703 | 558 |

## Model B (Early-Risk)

**Top 10 features (XGBoost, SHAP mean |value|):** ['severity_Critical', 'severity_Severe', 'age', 'symptom_shortness_of_breath', 'symptom_chest_pain', 'symptom_numbness', 'severity_Mild', 'symptom_count', 'disease_name_Anxiety Disorder', 'symptom_dizziness']
**Top risk-increasing features (LR, positive coefficient):** ['severity_Critical', 'disease_name_Hypertension', 'disease_name_Heart Disease', 'disease_name_Diabetes (Type 2)', 'disease_name_Stroke', 'disease_name_Breast Cancer', 'disease_name_Oral Cancer', 'disease_name_Chronic Kidney Disease', 'disease_name_Lung Cancer', 'age_risk_group_Elderly']
**Top risk-decreasing features (LR, negative coefficient):** ['state_Mizoram', 'disease_name_Jaundice (Hepatitis A)', 'disease_name_Cholera', 'disease_name_Gastroenteritis', 'disease_name_Anxiety Disorder', 'disease_name_Depression', 'disease_name_Sickle Cell Anemia', 'disease_name_Diarrhea', 'disease_name_Thalassemia', 'disease_name_Filariasis']

**Feature agreement:** 2/10 features appear in both models' top 10 — ['disease_name_Anxiety Disorder', 'severity_Critical']

**✅ Early-availability audit passed:** none of the top important features reference post-admission information (days_hospitalized, treatment_cost, length_of_stay, treatment_type). This confirms the Phase 5 exclusion held through to the final important-feature set — no leakage found.

**Test comparison — Model B (Early-Risk):**

| Metric | XGBoost | Logistic Regression |
|---|---|---|
| Accuracy | 0.8293 | 0.8273 |
| Precision | 0.3076 | 0.3083 |
| Recall | 0.7005 | 0.7225 |
| Specificity | 0.8421 | 0.8377 |
| F1 | 0.4275 | 0.4322 |
| ROC-AUC | 0.8684 | 0.8727 |
| PR-AUC | 0.5409 | 0.5324 |
| False Negatives | 109 | 101 |
| False Positives | 574 | 590 |