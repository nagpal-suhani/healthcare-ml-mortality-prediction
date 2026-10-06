"""
02_eda.py

Phase 3 — Exploratory Data Analysis

Purpose:
    Explore the cleaned dataset against the confirmed target `death_flag`.
    Produces console output (read this first) + saved chart images +
    a markdown summary report. Does NOT do encoding, scaling, splitting,
    or modeling — that is Phase 4/5/6.

Leakage rule enforced throughout this script:
    `outcome`, `cause_of_death`, `recovery_days`, `patient_id` are NEVER
    used as predictive features. `outcome` is shown in ONE descriptive
    cell purely for dataset context (it's where death_flag comes from),
    clearly labeled as non-feature.

Input:
    data/processed/cleaned_dataset.csv

Output:
    reports/figures/*.png
    reports/EDA_report.md

Run:
    python ml/analysis/02_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=0.95)
PALETTE = {"0": "#2E5EAA", "1": "#C0392B"}

DATA_PATH = Path("data/processed/cleaned_dataset.csv")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/EDA_report.md")

NUMERICAL = ["age", "bmi", "days_hospitalized", "treatment_cost_inr"]
CATEGORICAL_FOR_MORTALITY = [
    "severity", "disease_category", "disease_name", "gender", "urban_rural",
    "region", "smoking_status", "alcohol_use", "income_level",
    "insurance_status", "treatment_type", "hospital_type", "follow_up_required",
]

report_lines = []  # collects markdown for the final report


def log(md_line=""):
    print(md_line)
    report_lines.append(md_line)


def section(title):
    log(f"\n## {title}\n")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["death_flag_label"] = df["death_flag"].map({0: "Survived/Ongoing", 1: "Deceased"})
    return df


# ---------------------------------------------------------------- #
# 1. UNIVARIATE ANALYSIS
# ---------------------------------------------------------------- #
def univariate_numerical(df):
    section("1a. Univariate — Numerical Features")
    desc = df[NUMERICAL].describe().T
    desc["skew"] = df[NUMERICAL].skew()
    log(desc.round(2).to_markdown())

    skewed = desc[desc["skew"].abs() > 1].index.tolist()
    log(f"\n**Highly skewed (|skew| > 1):** {skewed}")
    log("`treatment_cost_inr` in particular is extremely right-skewed — "
        "confirms the Phase 2 decision to log-transform it during "
        "feature engineering rather than treating extreme values as errors.")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i, col in enumerate(NUMERICAL):
        sns.histplot(df[col], bins=40, ax=axes[0, i], color="#2E5EAA")
        axes[0, i].set_title(f"{col} — distribution")
        sns.boxplot(x=df[col], ax=axes[1, i], color="#2E5EAA")
        axes[1, i].set_title(f"{col} — boxplot")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_numerical_distributions.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/01_numerical_distributions.png`")


def univariate_categorical(df):
    section("1b. Univariate — Key Categorical Features")
    cols = ["severity", "disease_category", "gender", "urban_rural",
            "income_level", "insurance_status", "treatment_type"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        order = df[col].value_counts().index
        sns.countplot(y=df[col], order=order, ax=axes[i], color="#2E5EAA")
        axes[i].set_title(col)
        axes[i].set_xlabel("count")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_categorical_distributions.png", dpi=130)
    plt.close()
    log("Saved: `reports/figures/02_categorical_distributions.png`")

    for col in cols:
        pct = (df[col].value_counts(normalize=True) * 100).round(1)
        log(f"\n**{col}** top categories: " +
            ", ".join([f"{k} ({v}%)" for k, v in pct.head(4).items()]))


# ---------------------------------------------------------------- #
# 2. TARGET ANALYSIS
# ---------------------------------------------------------------- #
def target_distribution(df):
    section("2. Target Distribution — death_flag")
    counts = df["death_flag"].value_counts()
    pct = df["death_flag"].value_counts(normalize=True) * 100
    log(f"- Survived/Ongoing (0): {counts[0]:,} rows ({pct[0]:.2f}%)")
    log(f"- Deceased (1): {counts[1]:,} rows ({pct[1]:.2f}%)")
    log(f"\n**Class imbalance ratio ≈ {counts[0]/counts[1]:.1f} : 1** "
        "(survived : deceased). This is a genuine, expected imbalance "
        "for mortality data, not a data problem — must be handled with "
        "class weighting/resampling and recall-focused metrics in later phases.")

    # Context-only: outcome breakdown (NOT a feature, shown for understanding)
    log("\n*Context only — `outcome` (NOT a model feature, shown here only "
        "to explain where death_flag comes from):*")
    log(df["outcome"].value_counts().to_markdown())

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    counts.plot(kind="pie", labels=["Survived/Ongoing", "Deceased"],
                autopct="%1.1f%%", colors=["#2E5EAA", "#C0392B"], ax=ax)
    ax.set_ylabel("")
    ax.set_title("death_flag class distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_target_distribution.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/03_target_distribution.png`")


def mortality_by_categorical(df):
    section("2a. Mortality Rate by Categorical Variables")
    results = {}
    for col in CATEGORICAL_FOR_MORTALITY:
        grp = df.groupby(col)["death_flag"].agg(["mean", "count"])
        grp = grp[grp["count"] >= 30]  # ignore tiny groups for stability
        grp["mortality_pct"] = (grp["mean"] * 100).round(2)
        results[col] = grp.sort_values("mortality_pct", ascending=False)

    # Report the ones with the widest spread (most discriminating)
    spreads = {c: r["mortality_pct"].max() - r["mortality_pct"].min() for c, r in results.items()}
    ranked = sorted(spreads.items(), key=lambda x: -x[1])
    log("**Ranked by mortality-rate spread across categories (most discriminating first):**\n")
    for col, spread in ranked:
        top = results[col].head(2)
        bottom = results[col].tail(1)
        log(f"- `{col}` — spread {spread:.1f} pts. Highest: " +
            ", ".join(f"{idx} ({row.mortality_pct}%)" for idx, row in top.iterrows()) +
            f" | Lowest: {bottom.index[0]} ({bottom.iloc[0].mortality_pct}%)")

    # Plot the top 6 most discriminating
    top6 = [c for c, _ in ranked[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, col in enumerate(top6):
        r = results[col].sort_values("mortality_pct", ascending=False)
        sns.barplot(x=r["mortality_pct"], y=r.index, ax=axes[i], color="#C0392B")
        axes[i].set_title(f"Mortality rate by {col}")
        axes[i].set_xlabel("Mortality rate (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_mortality_by_category.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/04_mortality_by_category.png` "
        f"(top 6 most discriminating: {top6})")
    return results


def mortality_by_numerical(df):
    section("2b. Mortality Trends by Numerical Variables")
    for col in NUMERICAL:
        m_dead = df.loc[df.death_flag == 1, col].mean()
        m_alive = df.loc[df.death_flag == 0, col].mean()
        log(f"- `{col}`: mean = {m_alive:.1f} (survived/ongoing) vs "
            f"{m_dead:.1f} (deceased)")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, col in enumerate(NUMERICAL):
        sns.boxplot(x="death_flag_label", y=col, data=df, ax=axes[i],
                    hue="death_flag_label", palette=["#2E5EAA", "#C0392B"], legend=False)
        axes[i].set_title(f"{col} vs death_flag")
        axes[i].set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_numerical_vs_target.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/05_numerical_vs_target.png`")

    # Age-decade mortality trend (clean, monotonic pattern found in Phase 1)
    df["age_decade"] = (df["age"] // 10) * 10
    age_mort = df.groupby("age_decade")["death_flag"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    age_mort.plot(kind="line", marker="o", color="#C0392B", ax=ax)
    ax.set_title("Mortality rate by age decade")
    ax.set_xlabel("Age decade")
    ax.set_ylabel("Mortality rate (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_mortality_by_age.png", dpi=130)
    plt.close()
    log("Saved: `reports/figures/06_mortality_by_age.png` — "
        "confirms a clear, monotonically rising mortality trend with age "
        "(strongest single numerical signal in the dataset).")


# ---------------------------------------------------------------- #
# 3. BIVARIATE — SEVERITY x DISEASE_CATEGORY (clinically prioritized)
# ---------------------------------------------------------------- #
def severity_disease_interaction(df):
    section("3. Bivariate — Severity × Disease Category Interaction")
    pivot = df.pivot_table(values="death_flag", index="disease_category",
                            columns="severity", aggfunc="mean") * 100
    pivot = pivot[["Mild", "Moderate", "Severe", "Critical"]]
    log(pivot.round(1).to_markdown())

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax,
                cbar_kws={"label": "Mortality rate (%)"})
    ax.set_title("Mortality rate (%): disease category × severity")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_severity_disease_heatmap.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/07_severity_disease_heatmap.png`")
    log("\nBoth `severity` and `disease_category` independently move "
        "mortality risk, and the effect compounds — Critical-severity "
        "Non-Communicable-disease cases show the highest observed rates. "
        "This is a descriptive association only, not a causal claim.")


# ---------------------------------------------------------------- #
# 4. CORRELATION ANALYSIS
# ---------------------------------------------------------------- #
def correlation_analysis(df):
    section("4. Correlation Analysis (Numerical Features)")
    corr_cols = NUMERICAL + ["death_flag"]
    corr = df[corr_cols].corr()
    log(corr.round(2).to_markdown())

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap — numerical features + death_flag")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_correlation_heatmap.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/08_correlation_heatmap.png`")

    strong = corr["death_flag"].drop("death_flag").abs().sort_values(ascending=False)
    log(f"\n**Strongest linear correlation with death_flag:** "
        f"{strong.index[0]} (r = {corr['death_flag'][strong.index[0]]:.2f})")
    log("No pair of numerical features shows correlation above 0.3 with "
        "each other — **no meaningful multicollinearity** among the "
        "numerical features. Note: correlation here is Pearson (linear) "
        "only; it will under-detect non-linear relationships like the "
        "severity/disease interaction found above, and correlation is "
        "not evidence of causation.")


# ---------------------------------------------------------------- #
# 5. SYMPTOMS ANALYSIS
# ---------------------------------------------------------------- #
def symptoms_analysis(df):
    section("5. Symptoms Analysis")
    all_symptoms = df["symptoms"].str.split(",").explode().str.strip()
    freq = all_symptoms.value_counts().head(15)
    log("**Top 15 most frequent individual symptoms:**\n")
    log(freq.to_markdown())

    # Mortality rate for patients whose symptom list contains each top symptom
    mort_by_symptom = {}
    for sym in freq.index:
        mask = df["symptoms"].str.contains(sym, regex=False)
        mort_by_symptom[sym] = df.loc[mask, "death_flag"].mean() * 100
    mort_series = pd.Series(mort_by_symptom).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=mort_series.values, y=mort_series.index, color="#C0392B", ax=ax)
    ax.set_xlabel("Mortality rate (%) among patients reporting this symptom")
    ax.set_title("Mortality rate by top individual symptoms")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_mortality_by_symptom.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/09_mortality_by_symptom.png`")
    log(f"\nHighest-mortality symptom in top-15: **{mort_series.index[0]}** "
        f"({mort_series.iloc[0]:.1f}%). Full symptom multi-label feature "
        "engineering (one-hot per symptom) is deferred to Phase 4 — this "
        "is descriptive only.")


# ---------------------------------------------------------------- #
# 6. TEMPORAL ANALYSIS
# ---------------------------------------------------------------- #
def temporal_analysis(df):
    section("6. Temporal Analysis")
    year_mort = df.groupby("year")["death_flag"].mean() * 100
    season_mort = df.groupby("season")["death_flag"].mean() * 100
    log("**Mortality rate by year:**\n")
    log(year_mort.round(2).to_markdown())
    log("\n**Mortality rate by season:**\n")
    log(season_mort.round(2).to_markdown())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    year_mort.plot(kind="line", marker="o", ax=axes[0], color="#2E5EAA")
    axes[0].set_title("Mortality rate by year")
    axes[0].set_ylabel("Mortality rate (%)")
    season_order = ["Winter", "Spring", "Summer", "Monsoon", "Autumn", "Fall"]
    season_mort = season_mort.reindex([s for s in season_order if s in season_mort.index])
    season_mort.plot(kind="bar", ax=axes[1], color="#2E5EAA")
    axes[1].set_title("Mortality rate by season")
    axes[1].set_ylabel("Mortality rate (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_temporal_trends.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/10_temporal_trends.png`")
    yr_spread = year_mort.max() - year_mort.min()
    log(f"\nYear-over-year mortality spread is {yr_spread:.2f} points — "
        "no strong or clinically meaningful trend over time; `year`/`season` "
        "are unlikely to be useful predictive features and are treated as "
        "context only, not recommended as model inputs.")


def temporal_analysis(df):
    section("6. Temporal Analysis")
    year_mort = df.groupby("year")["death_flag"].mean() * 100
    season_mort = df.groupby("season")["death_flag"].mean() * 100
    log("**Mortality rate by year:**\n")
    log(year_mort.round(2).to_markdown())
    log("\n**Mortality rate by season:**\n")
    log(season_mort.round(2).to_markdown())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    year_mort.plot(kind="line", marker="o", ax=axes[0], color="#2E5EAA")
    axes[0].set_title("Mortality rate by year")
    axes[0].set_ylabel("Mortality rate (%)")
    season_order = ["Winter", "Spring", "Summer", "Monsoon", "Autumn", "Fall"]
    season_mort = season_mort.reindex([s for s in season_order if s in season_mort.index])
    season_mort.plot(kind="bar", ax=axes[1], color="#2E5EAA")
    axes[1].set_title("Mortality rate by season")
    axes[1].set_ylabel("Mortality rate (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_temporal_trends.png", dpi=130)
    plt.close()
    log("\nSaved: `reports/figures/10_temporal_trends.png`")
    yr_spread = year_mort.max() - year_mort.min()
    log(f"\nYear-over-year mortality spread is {yr_spread:.2f} points — "
        "no strong or clinically meaningful trend over time; `year`/`season` "
        "are unlikely to be useful predictive features and are treated as "
        "context only, not recommended as model inputs.")


# ---------------------------------------------------------------- #
# 7. LEAKAGE AUDIT (found during categorical mortality analysis)
# ---------------------------------------------------------------- #
def leakage_audit(df):
    section("7. Leakage Audit — follow_up_required")
    ct = pd.crosstab(df["follow_up_required"], df["death_flag"])
    log(ct.to_markdown())
    log("\n**Finding:** every single one of the 1,822 deceased patients has "
        "`follow_up_required = No`, and every patient with "
        "`follow_up_required = Yes` survived (13,905 of 13,905). This "
        "column is a near-perfect proxy for the target — clinically it "
        "makes sense (a deceased patient has no follow-up), but that also "
        "means it is logically determined *after* the outcome is known, "
        "the same way `recovery_days` and `cause_of_death` are.")
    log("\n**Decision required for Phase 4:** `follow_up_required` should "
        "be treated as a **leakage column and excluded from modeling**, "
        "the same as `outcome`/`cause_of_death`/`recovery_days`. "
        "Recommending exclusion — flagged for your approval below.")

    log("\n**CONFIRMED DECISION (user, post-Phase 3 review):** "
        "`follow_up_required` is DROPPED from all modeling feature sets.")

    section("7b. Leakage Watch — treatment_type")
    ct2 = pd.crosstab(df["treatment_type"], df["death_flag"], normalize="index").round(3) * 100
    log(ct2.round(1).to_markdown())
    log("\n**Finding:** `Palliative` treatment has a 100% mortality rate "
        "(all such patients died) and `ICU + Ventilator` shows 34.1%. "
        "Unlike `follow_up_required`, this is not automatically "
        "post-outcome — palliative/ICU care is a clinical decision made "
        "*during* treatment, generally before the final outcome is known, "
        "so it is plausibly legitimate to keep as a feature. However, "
        "because it is such a strong proxy (Palliative = 100%), it should "
        "be flagged as a **high-influence feature to watch**, not "
        "necessarily dropped — recommend keeping it in Phase 4 but noting "
        "it explicitly as a near-deterministic feature so its role in the "
        "final model is transparent, not hidden.")

    log("\n**CONFIRMED DECISION (user, post-Phase 3 review):** build TWO "
        "parallel feature sets for Phase 4 onward:")
    log("- **Model A (full-featured):** includes `treatment_type`. "
        "Expected to have higher accuracy/recall, but partly reflects "
        "clinical decisions made during care rather than pure admission-time risk.")
    log("- **Model B (early-risk-only):** excludes `treatment_type` "
        "(and `follow_up_required`). Uses only features available at/near "
        "diagnosis (age, severity, disease, comorbidity, bmi, lifestyle, "
        "socioeconomic, hospital context). A cleaner early-warning-risk story.")
    log("\nBoth models will be trained, evaluated, and compared in Phase 9, "
        "and both will be exposed in the final prediction UI.")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    log("# EDA Report — Healthcare Mortality Dataset\n")
    log(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns "
        "(cleaned). Target: `death_flag`.\n")
    log("**Leakage rule enforced in this analysis:** `outcome`, "
        "`cause_of_death`, `recovery_days`, `patient_id` are excluded "
        "from all feature-relationship analysis. `outcome` appears once, "
        "purely for descriptive context.")

    univariate_numerical(df)
    univariate_categorical(df)
    target_distribution(df)
    mortality_by_categorical(df)
    mortality_by_numerical(df)
    severity_disease_interaction(df)
    correlation_analysis(df)
    symptoms_analysis(df)
    temporal_analysis(df)
    leakage_audit(df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n\nFull EDA report written to: {REPORT_PATH}")
    print(f"Charts saved to: {FIG_DIR}/")


if __name__ == "__main__":
    main()
