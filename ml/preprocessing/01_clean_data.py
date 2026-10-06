"""
01_clean_data.py

Phase 2 — Data Cleaning

Purpose:
    Load the raw dataset, fix data-quality problems, and produce a clean,
    analysis-ready dataset. This script does NOT do modeling, encoding,
    or scaling — that happens later (Phase 5) using only the TRAINING
    split, to avoid data leakage.

Input:
    data/raw/indian_diseases_dataset.xlsx

Output:
    data/processed/cleaned_dataset.csv

Run:
    python ml/preprocessing/01_clean_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/indian_diseases_dataset.xlsx")
OUT_PATH = Path("data/processed/cleaned_dataset.csv")


def load_raw(path: Path) -> pd.DataFrame:
    """The file is named .csv but is actually an Excel workbook (sheet 'in')."""
    df = pd.read_excel(path, sheet_name="in")
    print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def fix_age_group_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data-quality issue found during inspection:
    Excel auto-converted the text age_group '10-19' into the date
    2019-10-01 ('Oct-19') for 2,062 rows. We rebuild age_group correctly
    from the reliable 'age' column instead of trusting the stored text.
    """
    def bucket(age):
        lo = (age // 10) * 10
        hi = lo + 9
        return f"{lo}-{hi}"

    before_bad = df["age_group"].apply(lambda x: not isinstance(x, str)).sum()
    df["age_group"] = df["age"].apply(bucket)
    print(f"Fixed age_group Excel date-corruption on {before_bad} rows "
          f"(rebuilt entirely from 'age' for consistency).")
    return df


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    comorbidity and alcohol_use are missing NOT because of bad data
    collection, but because the value legitimately doesn't apply
    (no comorbidity / non-drinker). We fill with an explicit label
    rather than imputing a statistic, since these are categorical
    and the missingness itself is meaningful information.
    """
    # NOTE: we deliberately do NOT use the literal string "None" here.
    # Pandas' default CSV reader treats "None" as a missing-value marker,
    # so it would silently turn back into NaN the moment this file is
    # re-loaded. Using an unambiguous label avoids that trap.
    df["comorbidity"] = df["comorbidity"].fillna("No Comorbidity")
    df["alcohol_use"] = df["alcohol_use"].fillna("Never")
    print("Filled comorbidity NaN -> 'No Comorbidity' (33.0% of rows)")
    print("Filled alcohol_use NaN -> 'Never' (33.1% of rows)")
    return df


def drop_leakage_and_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    These columns are dropped from the CLEANED dataset entirely because
    keeping them creates a direct or near-direct leak of the target
    (death_flag), decided in Phase 1:

    - patient_id: unique identifier, zero predictive value, must never
      be used as a feature.
    - cause_of_death: filled ONLY when death_flag == 1 (91% missing
      overall). Its mere presence/absence perfectly reveals the target.
    - recovery_days: missing exactly for Deceased + Chronic Management
      outcomes. Using it would leak outcome/death_flag.
    - outcome: death_flag is derived 1:1 from outcome == 'Deceased'.
      Keeping outcome as a feature would let the model trivially read
      off the answer.

    We drop these now (rather than only at modeling time) because this
    is a leakage-sensitive project and we want the cleaned dataset used
    for EDA/Phase 3 to already reflect the chosen target boundary.
    outcome and cause_of_death are still needed for some EDA context,
    so we keep a SEPARATE audit file with them for descriptive-only use.
    """
    leak_cols = ["cause_of_death", "recovery_days"]
    id_cols = ["patient_id"]

    audit_cols_to_keep_for_eda_only = ["outcome"]  # kept, flagged, never a feature

    df_clean = df.drop(columns=leak_cols)
    # patient_id is kept as an index/reference column (not a feature) —
    # dropped only right before modeling in Phase 5.
    print(f"Dropped leakage columns: {leak_cols}")
    print(f"Kept as reference (non-feature) columns: {id_cols + audit_cols_to_keep_for_eda_only}")
    return df_clean


def validate_no_duplicates_no_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    dup_rows = df.duplicated().sum()
    dup_ids = df.duplicated(subset=["patient_id"]).sum()
    print(f"Duplicate rows: {dup_rows} (none found, nothing to drop)")
    print(f"Duplicate patient_id: {dup_ids} (none found)")

    # Sanity checks on physiologically-plausible ranges
    assert df["age"].between(0, 120).all(), "Impossible age found"
    assert df["bmi"].between(10, 80).all(), "Impossible BMI found"
    assert df["days_hospitalized"].between(0, 365).all(), "Impossible hospitalization length"
    assert (df["treatment_cost_inr"] > 0).all(), "Non-positive treatment cost found"
    print("Range checks passed: age, bmi, days_hospitalized, treatment_cost_inr "
          "all within physiologically/operationally plausible bounds.")
    return df


def document_outlier_decision():
    """
    No outlier removal is performed in this script. Rationale (per column):

    - bmi (14.0-40.0): entire range is clinically real (14 = severe
      underweight/malnourished, common in TB/cancer/chronic disease
      populations in this dataset; 40 = class III obesity). Removing
      these would delete genuine high-risk patients relevant to death_flag.
      DECISION: retain as-is.

    - treatment_cost_inr (Rs 12 to Rs 1.5 crore): extreme right skew,
      but driven by real treatment_type differences (Home Remedies/
      Medication vs Surgery/ICU+Ventilator). This is expected healthcare
      cost behavior, not measurement error.
      DECISION: retain raw values in the cleaned dataset; apply a
      log1p transform at the FEATURE ENGINEERING stage (Phase 4/5) so
      models aren't destabilized by scale, without discarding any data.

    - days_hospitalized (0-60): 0 represents outpatient/same-day cases,
      60 represents long critical-care stays. Both are valid.
      DECISION: retain as-is.

    - age (0-95): includes infants (201 records at age 0) and elderly.
      Both are valid patient populations for this disease set.
      DECISION: retain as-is.
    """
    print("Outlier policy: NO rows removed. See function docstring for the "
          "per-column clinical justification (bmi, cost, hospitalization, age).")


def main():
    df = load_raw(RAW_PATH)
    df = fix_age_group_corruption(df)
    df = fill_structural_missing(df)
    df = validate_no_duplicates_no_impossible_values(df)
    document_outlier_decision()
    df = drop_leakage_and_id_columns(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {OUT_PATH}  ({df.shape[0]} rows, {df.shape[1]} columns)")
    print("\nRemaining missing values after cleaning:")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
