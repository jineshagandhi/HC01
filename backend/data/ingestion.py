"""
Data ingestion pipeline for ICU Diagnostic Risk Assistant.

Loads MIMIC-IV demo data (gzipped CSVs) and PhysioNet Sepsis Challenge
PSV files, providing clean DataFrames for downstream analysis.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.config import MIMIC_DIR, SEPSIS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
HOSP_DIR = Path(MIMIC_DIR) / "hosp"
ICU_DIR = Path(MIMIC_DIR) / "icu"
SEPSIS_TRAINING_DIR = Path(SEPSIS_DIR) / "training_setA"

# ---------------------------------------------------------------------------
# Key MIMIC-IV item IDs
# ---------------------------------------------------------------------------
VITAL_ITEM_IDS = {
    "Heart Rate": [220045],
    "Systolic BP": [220050, 220179],
    "Diastolic BP": [220051, 220180],
    "Respiratory Rate": [220210],
    "SpO2": [220277],
    "Temperature (C)": [223761, 223762],
    "GCS Total": [220739],
}

LAB_ITEM_IDS = {
    "WBC": 51301,
    "Lactate": 50813,
    "Creatinine": 50912,
    "Platelets": 51265,
    "Bilirubin Total": 50885,
    "BUN": 51006,
}

ALL_VITAL_IDS = [iid for ids in VITAL_ITEM_IDS.values() for iid in ids]
ALL_LAB_IDS = list(LAB_ITEM_IDS.values())

# ---------------------------------------------------------------------------
# Sepsis challenge column names
# ---------------------------------------------------------------------------
SEPSIS_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel",
]

# ---------------------------------------------------------------------------
# Low-level loaders (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_patients_raw() -> pd.DataFrame:
    """Load hosp/patients.csv.gz."""
    path = HOSP_DIR / "patients.csv.gz"
    logger.info("Loading patients from %s", path)
    df = pd.read_csv(path, compression="gzip")
    # Normalise column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    return df


@lru_cache(maxsize=1)
def _load_admissions_raw() -> pd.DataFrame:
    """Load hosp/admissions.csv.gz with datetime parsing."""
    path = HOSP_DIR / "admissions.csv.gz"
    logger.info("Loading admissions from %s", path)
    df = pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    )
    df.columns = df.columns.str.lower().str.strip()
    return df


@lru_cache(maxsize=1)
def _load_labevents_raw() -> pd.DataFrame:
    """Load hosp/labevents.csv.gz (can be large even in demo)."""
    path = HOSP_DIR / "labevents.csv.gz"
    logger.info("Loading labevents from %s", path)
    df = pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["charttime"],
        dtype={"value": str},
    )
    df.columns = df.columns.str.lower().str.strip()
    # Coerce valuenum to float where possible
    if "valuenum" in df.columns:
        df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def _load_d_labitems_raw() -> pd.DataFrame:
    """Load hosp/d_labitems.csv.gz (lookup table)."""
    path = HOSP_DIR / "d_labitems.csv.gz"
    logger.info("Loading d_labitems from %s", path)
    df = pd.read_csv(path, compression="gzip")
    df.columns = df.columns.str.lower().str.strip()
    return df


@lru_cache(maxsize=1)
def _load_diagnoses_icd_raw() -> pd.DataFrame:
    """Load hosp/diagnoses_icd.csv.gz."""
    path = HOSP_DIR / "diagnoses_icd.csv.gz"
    logger.info("Loading diagnoses_icd from %s", path)
    df = pd.read_csv(path, compression="gzip")
    df.columns = df.columns.str.lower().str.strip()
    return df


@lru_cache(maxsize=1)
def _load_chartevents_raw() -> pd.DataFrame:
    """Load icu/chartevents.csv.gz with datetime parsing."""
    path = ICU_DIR / "chartevents.csv.gz"
    logger.info("Loading chartevents from %s", path)
    df = pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["charttime"],
        dtype={"value": str},
    )
    df.columns = df.columns.str.lower().str.strip()
    if "valuenum" in df.columns:
        df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def _load_d_items_raw() -> pd.DataFrame:
    """Load icu/d_items.csv.gz (item definitions)."""
    path = ICU_DIR / "d_items.csv.gz"
    logger.info("Loading d_items from %s", path)
    df = pd.read_csv(path, compression="gzip")
    df.columns = df.columns.str.lower().str.strip()
    return df


@lru_cache(maxsize=1)
def _load_icustays_raw() -> pd.DataFrame:
    """Load icu/icustays.csv.gz."""
    path = ICU_DIR / "icustays.csv.gz"
    logger.info("Loading icustays from %s", path)
    df = pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["intime", "outtime"],
    )
    df.columns = df.columns.str.lower().str.strip()
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mimic_patients() -> pd.DataFrame:
    """
    Return patient demographics joined with their first admission record.

    Columns include: subject_id, gender, anchor_age, hadm_id, admittime,
    dischtime, admission_type, and diagnosis (where available).
    """
    patients = _load_patients_raw().copy()
    admissions = _load_admissions_raw().copy()

    # Keep the earliest admission per patient for a concise demographics view
    admissions_sorted = admissions.sort_values("admittime")
    first_admission = admissions_sorted.drop_duplicates(subset=["subject_id"], keep="first")

    # Select useful admission columns
    adm_cols = ["subject_id", "hadm_id", "admittime", "dischtime",
                "admission_type"]
    # Some MIMIC-IV versions dropped the 'diagnosis' column from admissions
    if "diagnosis" in first_admission.columns:
        adm_cols.append("diagnosis")

    merged = patients.merge(
        first_admission[adm_cols],
        on="subject_id",
        how="left",
    )

    # If diagnosis column doesn't exist, try to pull primary ICD diagnosis text
    if "diagnosis" not in merged.columns:
        diag = _load_diagnoses_icd_raw()
        primary = diag[diag["seq_num"] == 1][["subject_id", "icd_code"]].drop_duplicates(
            subset=["subject_id"], keep="first"
        )
        merged = merged.merge(primary, on="subject_id", how="left")
        merged.rename(columns={"icd_code": "diagnosis"}, inplace=True)

    return merged


def load_mimic_labs(subject_id: int) -> pd.DataFrame:
    """
    Return lab results for a given patient with human-readable lab names.

    Parameters
    ----------
    subject_id : int
        The MIMIC subject_id of the patient.

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, hadm_id, itemid, label, charttime, value,
        valuenum, valueuom, ref_range_lower, ref_range_upper
    """
    labs = _load_labevents_raw()
    d_lab = _load_d_labitems_raw()

    patient_labs = labs[labs["subject_id"] == subject_id].copy()
    if patient_labs.empty:
        logger.warning("No lab events found for subject_id=%d", subject_id)
        return pd.DataFrame()

    # Join with lab item definitions to get label
    merge_cols = ["itemid"]
    lab_info_cols = ["itemid", "label"]
    if "fluid" in d_lab.columns:
        lab_info_cols.append("fluid")
    if "category" in d_lab.columns:
        lab_info_cols.append("category")

    merged = patient_labs.merge(
        d_lab[lab_info_cols],
        on="itemid",
        how="left",
    )

    # Ensure charttime is datetime
    if not pd.api.types.is_datetime64_any_dtype(merged.get("charttime")):
        merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")

    merged.sort_values("charttime", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def load_mimic_vitals(subject_id: int, stay_id: Optional[int] = None) -> pd.DataFrame:
    """
    Return charted vitals for a patient/stay with human-readable item names.

    Parameters
    ----------
    subject_id : int
        The MIMIC subject_id.
    stay_id : int, optional
        If provided, filter to a specific ICU stay.

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, hadm_id, stay_id, itemid, label, category,
        charttime, value, valuenum, valueuom
    """
    chart = _load_chartevents_raw()
    d_items = _load_d_items_raw()

    mask = chart["subject_id"] == subject_id
    if stay_id is not None:
        mask &= chart["stay_id"] == stay_id

    patient_vitals = chart[mask].copy()
    if patient_vitals.empty:
        logger.warning(
            "No chart events for subject_id=%d, stay_id=%s", subject_id, stay_id
        )
        return pd.DataFrame()

    # Join item definitions
    item_cols = ["itemid", "label"]
    if "category" in d_items.columns:
        item_cols.append("category")

    merged = patient_vitals.merge(
        d_items[item_cols],
        on="itemid",
        how="left",
    )

    if not pd.api.types.is_datetime64_any_dtype(merged.get("charttime")):
        merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")

    merged.sort_values("charttime", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def load_mimic_diagnoses(subject_id: int) -> pd.DataFrame:
    """
    Return ICD diagnoses for a patient.

    Parameters
    ----------
    subject_id : int

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, hadm_id, seq_num, icd_code, icd_version
    """
    diag = _load_diagnoses_icd_raw()
    patient_diag = diag[diag["subject_id"] == subject_id].copy()
    if patient_diag.empty:
        logger.warning("No diagnoses found for subject_id=%d", subject_id)
    patient_diag.reset_index(drop=True, inplace=True)
    return patient_diag


def load_sepsis_patient(patient_file: str) -> pd.DataFrame:
    """
    Load a single PhysioNet Sepsis Challenge .psv file.

    Parameters
    ----------
    patient_file : str
        Filename (e.g. 'p000001.psv') or full path. If just a filename,
        it is resolved relative to SEPSIS_TRAINING_DIR.

    Returns
    -------
    pd.DataFrame
        One row per ICU hour with all 41 columns.
    """
    filepath = Path(patient_file)
    if not filepath.is_absolute():
        filepath = SEPSIS_TRAINING_DIR / filepath

    if not filepath.exists():
        raise FileNotFoundError(f"Sepsis patient file not found: {filepath}")

    df = pd.read_csv(filepath, sep="|", na_values=["NaN", "nan", ""])
    # Validate expected columns are present
    expected_subset = {"HR", "O2Sat", "Temp", "SBP", "Resp", "SepsisLabel", "ICULOS"}
    missing = expected_subset - set(df.columns)
    if missing:
        logger.warning("Sepsis file %s missing expected columns: %s", filepath, missing)

    # Convert numeric columns
    numeric_cols = [c for c in df.columns if c not in ("Gender",)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_patient_list() -> list[dict]:
    """
    Return a lightweight list of all MIMIC-IV demo patients.

    Each entry is a dict with keys: patient_id, age, gender, diagnosis.
    """
    patients = load_mimic_patients()

    records = []
    for _, row in patients.iterrows():
        records.append({
            "patient_id": int(row["subject_id"]),
            "age": int(row.get("anchor_age", 0)) if pd.notna(row.get("anchor_age")) else None,
            "gender": row.get("gender", "Unknown"),
            "diagnosis": row.get("diagnosis", "Unknown") if pd.notna(row.get("diagnosis")) else "Unknown",
        })

    return records


def get_patient_full_data(subject_id: int) -> dict:
    """
    Aggregate all available data for a single MIMIC-IV patient.

    Returns DataFrames directly so Streamlit pages can use them natively.

    Returns
    -------
    dict with keys:
        - demographics: dict of patient info
        - admissions: pd.DataFrame
        - icu_stays: pd.DataFrame
        - vitals: pd.DataFrame (all stays combined, with normalized columns)
        - labs: pd.DataFrame (with normalized columns)
        - diagnoses: pd.DataFrame
    """
    # Demographics
    patients = _load_patients_raw()
    pat_row = patients[patients["subject_id"] == subject_id]
    if pat_row.empty:
        raise ValueError(f"Patient subject_id={subject_id} not found in MIMIC-IV demo data.")

    demographics = pat_row.iloc[0].to_dict()
    demographics = {k: (None if pd.isna(v) else v) for k, v in demographics.items()}

    # Admissions
    admissions = _load_admissions_raw()
    pat_adm = admissions[admissions["subject_id"] == subject_id].copy()
    pat_adm.sort_values("admittime", inplace=True)

    # ICU stays
    icustays = _load_icustays_raw()
    pat_icu = icustays[icustays["subject_id"] == subject_id].copy()
    pat_icu.sort_values("intime", inplace=True)

    # Vitals — combine all stays into one DataFrame with pivoted columns
    vitals_frames = []
    for _, stay_row in pat_icu.iterrows():
        sid = int(stay_row["stay_id"])
        vdf = load_mimic_vitals(subject_id, stay_id=sid)
        if not vdf.empty:
            vitals_frames.append(vdf)

    if vitals_frames:
        all_vitals = pd.concat(vitals_frames, ignore_index=True)
        # Pivot vitals into wide format: one column per vital sign
        vitals_df = _pivot_vitals(all_vitals)
    else:
        vitals_df = pd.DataFrame()

    # Labs
    labs_df = load_mimic_labs(subject_id)

    # Diagnoses
    diag_df = load_mimic_diagnoses(subject_id)

    return {
        "demographics": demographics,
        "admissions": pat_adm,
        "icu_stays": pat_icu,
        "vitals": vitals_df,
        "labs": labs_df,
        "diagnoses": diag_df,
    }


def _pivot_vitals(vitals_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format vitals (one row per measurement) to wide format
    (one row per timestamp, one column per vital sign).
    """
    if vitals_long.empty:
        return pd.DataFrame()

    time_col = "charttime" if "charttime" in vitals_long.columns else None
    label_col = "label" if "label" in vitals_long.columns else None
    value_col = "valuenum" if "valuenum" in vitals_long.columns else None

    if not all([time_col, label_col, value_col]):
        return vitals_long

    # Map labels to standard column names
    # Verified against actual MIMIC-IV demo d_items labels
    label_map = {
        "Heart Rate": "heart_rate",
        "Arterial Blood Pressure systolic": "sbp",
        "Non Invasive Blood Pressure systolic": "sbp",
        "Manual Blood Pressure Systolic Left": "sbp",
        "Manual Blood Pressure Systolic Right": "sbp",
        "Arterial Blood Pressure diastolic": "dbp",
        "Non Invasive Blood Pressure diastolic": "dbp",
        "Manual Blood Pressure Diastolic Left": "dbp",
        "Manual Blood Pressure Diastolic Right": "dbp",
        "Respiratory Rate": "respiratory_rate",
        "Respiratory Rate (Total)": "respiratory_rate",
        "O2 saturation pulseoxymetry": "spo2",
        "SpO2": "spo2",
        "Temperature Fahrenheit": "temperature_f",
        "Temperature Celsius": "temperature",
        "GCS - Eye Opening": "gcs_eye",
        "GCS - Verbal Response": "gcs_verbal",
        "GCS - Motor Response": "gcs_motor",
    }

    vitals_long = vitals_long.copy()
    vitals_long["param"] = vitals_long[label_col].map(label_map)
    vitals_long = vitals_long.dropna(subset=["param", value_col])

    if vitals_long.empty:
        return pd.DataFrame()

    # Pivot — take mean if duplicates at same time
    try:
        pivoted = vitals_long.pivot_table(
            index=time_col, columns="param", values=value_col, aggfunc="mean"
        ).reset_index()

        # Convert Fahrenheit to Celsius if present
        if "temperature_f" in pivoted.columns:
            if "temperature" not in pivoted.columns:
                pivoted["temperature"] = pivoted["temperature_f"].apply(
                    lambda f: (f - 32) * 5 / 9 if pd.notna(f) else None
                )
            else:
                # Fill missing Celsius with converted Fahrenheit
                pivoted["temperature"] = pivoted["temperature"].fillna(
                    pivoted["temperature_f"].apply(
                        lambda f: (f - 32) * 5 / 9 if pd.notna(f) else None
                    )
                )
            pivoted.drop(columns=["temperature_f"], errors="ignore", inplace=True)

        # Calculate GCS Total from components (MIMIC-IV demo has Eye/Verbal/Motor, not Total)
        gcs_components = ["gcs_eye", "gcs_verbal", "gcs_motor"]
        available_components = [c for c in gcs_components if c in pivoted.columns]
        if available_components:
            # Sum available components (min_count=1 so we get partial GCS if not all 3 present)
            pivoted["gcs"] = pivoted[available_components].sum(axis=1, min_count=1)
            pivoted.drop(columns=available_components, errors="ignore", inplace=True)

        pivoted.sort_values(time_col, inplace=True)
        return pivoted
    except Exception as exc:
        logger.warning("Failed to pivot vitals: %s", exc)
        return vitals_long


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Convert a DataFrame to a list of dicts, replacing NaN/NaT with None
    and converting Timestamps to ISO strings for JSON compatibility.
    """
    if df.empty:
        return []

    records = df.to_dict(orient="records")
    clean = []
    for rec in records:
        cleaned = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                cleaned[k] = v.isoformat() if pd.notna(v) else None
            elif isinstance(v, float) and pd.isna(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        clean.append(cleaned)
    return clean


def list_sepsis_patient_files() -> list[str]:
    """Return sorted list of .psv filenames available in the sepsis training set."""
    if not SEPSIS_TRAINING_DIR.exists():
        logger.warning("Sepsis training directory not found: %s", SEPSIS_TRAINING_DIR)
        return []
    files = sorted(p.name for p in SEPSIS_TRAINING_DIR.glob("*.psv"))
    return files


def clear_cache() -> None:
    """Clear all LRU caches to force data reload."""
    _load_patients_raw.cache_clear()
    _load_admissions_raw.cache_clear()
    _load_labevents_raw.cache_clear()
    _load_d_labitems_raw.cache_clear()
    _load_diagnoses_icd_raw.cache_clear()
    _load_chartevents_raw.cache_clear()
    _load_d_items_raw.cache_clear()
    _load_icustays_raw.cache_clear()
    logger.info("All data caches cleared.")