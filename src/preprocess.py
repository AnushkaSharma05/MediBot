# src/preprocess.py
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def clean_symptom(symptom):
    """
    Normalize symptom string:
    - strip whitespace
    - lowercase
    - replace spaces with underscore
    """
    if pd.isna(symptom):
        return None
    return str(symptom).strip().lower().replace(" ", "_")

def load_and_analyze():
    """
    Load all 4 CSVs, clean them, print a full data report.
    Returns cleaned dataframes.
    """
    print("=" * 55)
    print("       MEDICAL CHATBOT — DATA ANALYSIS REPORT")
    print("=" * 55)

    # ── 1. Load main dataset ───────────────────────────
    df = pd.read_csv(config.DATASET_PATH)
    print(f"\n[1] MAIN DATASET — dataset.csv")
    print(f"    Shape        : {df.shape}")
    print(f"    Columns      : {list(df.columns)}")
    print(f"    Null values  :\n{df.isnull().sum()}")

    # ── 2. Clean symptom columns ───────────────────────
    symptom_cols = [c for c in df.columns if "Symptom" in c]
    for col in symptom_cols:
        df[col] = df[col].apply(clean_symptom)

    df["Disease"] = df["Disease"].str.strip()

    # ── 3. Extract unique values ───────────────────────
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(df[col].dropna().unique())

    all_diseases = sorted(df["Disease"].unique())

    print(f"\n    Unique diseases  : {len(all_diseases)}")
    print(f"    Unique symptoms  : {len(all_symptoms)}")
    print(f"\n    Diseases found:")
    for i, d in enumerate(all_diseases):
        print(f"      {i:>2}. {d}")

    # ── 4. Load supporting files ───────────────────────
    desc_df  = pd.read_csv(config.DESCRIPTION_PATH)
    sev_df   = pd.read_csv(config.SEVERITY_PATH)
    prec_df  = pd.read_csv(config.PRECAUTION_PATH)

    print(f"\n[2] DESCRIPTION FILE  — shape: {desc_df.shape}")
    print(f"    Columns: {list(desc_df.columns)}")

    print(f"\n[3] SEVERITY FILE     — shape: {sev_df.shape}")
    print(f"    Columns: {list(sev_df.columns)}")

    print(f"\n[4] PRECAUTION FILE   — shape: {prec_df.shape}")
    print(f"    Columns: {list(prec_df.columns)}")

    # ── 5. Severity — clean symptom column ────────────
    sev_col = sev_df.columns[0]   # usually 'Symptom'
    sev_df[sev_col] = sev_df[sev_col].apply(clean_symptom)

    print("\n" + "=" * 55)
    print("  ✅ Data loaded and cleaned successfully!")
    print("=" * 55)

    return df, desc_df, sev_df, prec_df, sorted(all_symptoms), all_diseases


if __name__ == "__main__":
    df, desc_df, sev_df, prec_df, symptoms, diseases = load_and_analyze()
    print(f"\nSample symptoms: {symptoms[:10]}")