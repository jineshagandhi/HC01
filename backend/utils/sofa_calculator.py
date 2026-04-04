"""
Clinical scoring systems for ICU diagnostic risk assessment.

Implements SOFA, qSOFA, SIRS scoring and lab trend analysis
based on published clinical criteria.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union


# ---------------------------------------------------------------------------
# SOFA Score (Sequential Organ Failure Assessment)  –  range 0-24
# ---------------------------------------------------------------------------

def _respiration_score(pao2_fio2: Optional[float], on_ventilation: bool) -> int:
    if pao2_fio2 is None:
        return 0
    if pao2_fio2 < 100 and on_ventilation:
        return 4
    if pao2_fio2 < 200 and on_ventilation:
        return 3
    if pao2_fio2 < 300:
        return 2
    if pao2_fio2 < 400:
        return 1
    return 0


def _coagulation_score(platelets: Optional[float]) -> int:
    if platelets is None:
        return 0
    if platelets < 20:
        return 4
    if platelets < 50:
        return 3
    if platelets < 100:
        return 2
    if platelets < 150:
        return 1
    return 0


def _liver_score(bilirubin: Optional[float]) -> int:
    if bilirubin is None:
        return 0
    if bilirubin >= 12.0:
        return 4
    if bilirubin >= 6.0:
        return 3
    if bilirubin >= 2.0:
        return 2
    if bilirubin >= 1.2:
        return 1
    return 0


def _cardiovascular_score(
    map_value: Optional[float],
    on_vasopressors: bool,
    dopamine_dose: float = 0.0,
    epinephrine_dose: float = 0.0,
) -> int:
    if on_vasopressors:
        if dopamine_dose > 15 or epinephrine_dose > 0.1:
            return 4
        if dopamine_dose > 5 or epinephrine_dose > 0:
            return 3
        if dopamine_dose > 0:
            return 2
    if map_value is not None and map_value < 70:
        return 1
    return 0


def _cns_score(gcs: Optional[int]) -> int:
    if gcs is None:
        return 0
    if gcs < 6:
        return 4
    if gcs < 10:
        return 3
    if gcs < 13:
        return 2
    if gcs < 15:
        return 1
    return 0


def _renal_score(creatinine: Optional[float]) -> int:
    if creatinine is None:
        return 0
    if creatinine >= 5.0:
        return 4
    if creatinine >= 3.5:
        return 3
    if creatinine >= 2.0:
        return 2
    if creatinine >= 1.2:
        return 1
    return 0


def calculate_sofa_score(
    pao2_fio2_ratio: Optional[float] = None,
    platelets: Optional[float] = None,
    bilirubin: Optional[float] = None,
    map_value: Optional[float] = None,
    gcs: Optional[int] = None,
    creatinine: Optional[float] = None,
    on_vasopressors: bool = False,
    on_ventilation: bool = False,
    dopamine_dose: float = 0.0,
    epinephrine_dose: float = 0.0,
) -> dict:
    """Calculate SOFA score from individual clinical values.

    Parameters
    ----------
    pao2_fio2_ratio : float, optional  – PaO2/FiO2 ratio (mmHg)
    platelets : float, optional        – Platelet count (x10^3/uL)
    bilirubin : float, optional        – Total bilirubin (mg/dL)
    map_value : float, optional        – Mean Arterial Pressure (mmHg)
    gcs : int, optional                – Glasgow Coma Scale (3-15)
    creatinine : float, optional       – Serum creatinine (mg/dL)
    on_vasopressors : bool             – Whether patient is on vasopressors
    on_ventilation : bool              – Whether patient is mechanically ventilated
    dopamine_dose : float              – Dopamine dose (mcg/kg/min)
    epinephrine_dose : float           – Epinephrine dose (mcg/kg/min)

    Returns
    -------
    dict with component scores and total.
    """
    resp = _respiration_score(pao2_fio2_ratio, on_ventilation)
    coag = _coagulation_score(platelets)
    liver = _liver_score(bilirubin)
    cardio = _cardiovascular_score(map_value, on_vasopressors, dopamine_dose, epinephrine_dose)
    cns = _cns_score(gcs)
    renal = _renal_score(creatinine)

    total = resp + coag + liver + cardio + cns + renal

    # Mortality risk estimate based on published SOFA data
    if total <= 1:
        mortality_risk = "< 10%"
    elif total <= 3:
        mortality_risk = "15-20%"
    elif total <= 6:
        mortality_risk = "20-25%"
    elif total <= 9:
        mortality_risk = "30-40%"
    elif total <= 12:
        mortality_risk = "40-55%"
    elif total <= 15:
        mortality_risk = "55-70%"
    else:
        mortality_risk = "> 70%"

    return {
        "score_name": "SOFA",
        "components": {
            "respiration": resp,
            "coagulation": coag,
            "liver": liver,
            "cardiovascular": cardio,
            "cns": cns,
            "renal": renal,
        },
        "total": total,
        "max_possible": 24,
        "mortality_risk": mortality_risk,
        "interpretation": _interpret_sofa(total),
    }


def _interpret_sofa(total: int) -> str:
    if total <= 1:
        return "Minimal organ dysfunction"
    if total <= 6:
        return "Moderate organ dysfunction"
    if total <= 12:
        return "Severe organ dysfunction – consider ICU escalation"
    return "Very severe organ dysfunction – high mortality risk"


def calculate_sofa_from_patient_data(
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    timestamp: Optional[datetime] = None,
) -> dict:
    """Calculate SOFA score from patient vitals and labs DataFrames.

    Parameters
    ----------
    vitals_df : DataFrame
        Expected columns: timestamp, parameter, value.
        Parameters: MAP, GCS (and optionally on_ventilation, on_vasopressors).
    labs_df : DataFrame
        Expected columns: timestamp, parameter, value.
        Parameters: PaO2_FiO2, platelets, bilirubin, creatinine, dopamine_dose, epinephrine_dose.
    timestamp : datetime, optional
        Point-in-time for score. Uses most recent values within 24 h window.
        Defaults to the latest available timestamp.

    Returns
    -------
    dict – same structure as calculate_sofa_score output.
    """
    if timestamp is None:
        ts_candidates = []
        if not vitals_df.empty and "timestamp" in vitals_df.columns:
            ts_candidates.append(vitals_df["timestamp"].max())
        if not labs_df.empty and "timestamp" in labs_df.columns:
            ts_candidates.append(labs_df["timestamp"].max())
        timestamp = max(ts_candidates) if ts_candidates else datetime.now()

    window_start = timestamp - timedelta(hours=24)

    def _latest_value(df: pd.DataFrame, param: str) -> Optional[float]:
        if df.empty or "parameter" not in df.columns:
            return None
        mask = (
            (df["parameter"] == param)
            & (df["timestamp"] >= window_start)
            & (df["timestamp"] <= timestamp)
        )
        subset = df.loc[mask]
        if subset.empty:
            return None
        return float(subset.sort_values("timestamp", ascending=False).iloc[0]["value"])

    def _latest_flag(df: pd.DataFrame, param: str) -> bool:
        val = _latest_value(df, param)
        if val is None:
            return False
        return bool(val)

    return calculate_sofa_score(
        pao2_fio2_ratio=_latest_value(labs_df, "PaO2_FiO2"),
        platelets=_latest_value(labs_df, "platelets"),
        bilirubin=_latest_value(labs_df, "bilirubin"),
        map_value=_latest_value(vitals_df, "MAP"),
        gcs=int(_latest_value(vitals_df, "GCS")) if _latest_value(vitals_df, "GCS") is not None else None,
        creatinine=_latest_value(labs_df, "creatinine"),
        on_vasopressors=_latest_flag(vitals_df, "on_vasopressors"),
        on_ventilation=_latest_flag(vitals_df, "on_ventilation"),
        dopamine_dose=_latest_value(labs_df, "dopamine_dose") or 0.0,
        epinephrine_dose=_latest_value(labs_df, "epinephrine_dose") or 0.0,
    )


# ---------------------------------------------------------------------------
# qSOFA Score (Quick SOFA)  –  range 0-3
# ---------------------------------------------------------------------------

def calculate_qsofa(
    sbp: Optional[float] = None,
    rr: Optional[float] = None,
    gcs: Optional[int] = None,
) -> dict:
    """Calculate qSOFA score.

    Parameters
    ----------
    sbp : float, optional – Systolic blood pressure (mmHg)
    rr  : float, optional – Respiratory rate (breaths/min)
    gcs : int, optional   – Glasgow Coma Scale (3-15)

    Returns
    -------
    dict with component flags and total score.
    """
    sbp_flag = 1 if sbp is not None and sbp <= 100 else 0
    rr_flag = 1 if rr is not None and rr >= 22 else 0
    gcs_flag = 1 if gcs is not None and gcs < 15 else 0

    total = sbp_flag + rr_flag + gcs_flag

    return {
        "score_name": "qSOFA",
        "components": {
            "low_sbp": bool(sbp_flag),
            "high_rr": bool(rr_flag),
            "altered_mentation": bool(gcs_flag),
        },
        "total": total,
        "max_possible": 3,
        "positive": total >= 2,
        "interpretation": (
            "Positive qSOFA (>=2) – higher risk of poor outcome; consider sepsis workup"
            if total >= 2
            else "Negative qSOFA (<2) – lower risk but does not exclude sepsis"
        ),
    }


def calculate_qsofa_from_patient_data(
    vitals_df: pd.DataFrame,
    timestamp: Optional[datetime] = None,
) -> dict:
    """Calculate qSOFA from a vitals DataFrame.

    Parameters
    ----------
    vitals_df : DataFrame
        Expected columns: timestamp, parameter, value.
        Parameters looked up: SBP, RR, GCS.
    timestamp : datetime, optional
        Defaults to latest available timestamp.

    Returns
    -------
    dict – same structure as calculate_qsofa output.
    """
    if timestamp is None:
        if not vitals_df.empty and "timestamp" in vitals_df.columns:
            timestamp = vitals_df["timestamp"].max()
        else:
            timestamp = datetime.now()

    window_start = timestamp - timedelta(hours=6)

    def _latest(param: str) -> Optional[float]:
        if vitals_df.empty or "parameter" not in vitals_df.columns:
            return None
        mask = (
            (vitals_df["parameter"] == param)
            & (vitals_df["timestamp"] >= window_start)
            & (vitals_df["timestamp"] <= timestamp)
        )
        subset = vitals_df.loc[mask]
        if subset.empty:
            return None
        return float(subset.sort_values("timestamp", ascending=False).iloc[0]["value"])

    sbp = _latest("SBP")
    rr = _latest("RR")
    gcs_val = _latest("GCS")

    return calculate_qsofa(
        sbp=sbp,
        rr=rr,
        gcs=int(gcs_val) if gcs_val is not None else None,
    )


# ---------------------------------------------------------------------------
# SIRS Criteria  –  meet >= 2 of 4 = positive
# ---------------------------------------------------------------------------

def calculate_sirs(
    temp: Optional[float] = None,
    hr: Optional[float] = None,
    rr: Optional[float] = None,
    wbc: Optional[float] = None,
    paco2: Optional[float] = None,
) -> dict:
    """Calculate SIRS criteria.

    Parameters
    ----------
    temp  : float, optional – Body temperature (°C)
    hr    : float, optional – Heart rate (bpm)
    rr    : float, optional – Respiratory rate (breaths/min)
    wbc   : float, optional – White blood cell count (cells/uL)
    paco2 : float, optional – Partial pressure CO2 (mmHg)

    Returns
    -------
    dict with component flags, total criteria met, and positivity.
    """
    temp_flag = (temp is not None) and (temp > 38.0 or temp < 36.0)
    hr_flag = (hr is not None) and (hr > 90)
    rr_flag = (rr is not None and rr > 20) or (paco2 is not None and paco2 < 32)
    # MIMIC-IV reports WBC in K/uL (e.g., 15.2 = 15,200 cells/uL)
    # Handle both units: if value > 100, assume cells/uL; else assume K/uL
    wbc_flag = False
    if wbc is not None:
        if wbc > 100:
            # Raw cell count (cells/uL)
            wbc_flag = wbc > 12000 or wbc < 4000
        else:
            # K/uL (MIMIC-IV standard)
            wbc_flag = wbc > 12 or wbc < 4

    criteria_met = sum([temp_flag, hr_flag, rr_flag, wbc_flag])

    return {
        "score_name": "SIRS",
        "components": {
            "temperature_abnormal": bool(temp_flag),
            "tachycardia": bool(hr_flag),
            "tachypnea_or_hypocapnia": bool(rr_flag),
            "wbc_abnormal": bool(wbc_flag),
        },
        "total": criteria_met,
        "max_possible": 4,
        "positive": criteria_met >= 2,
        "interpretation": (
            f"SIRS positive ({criteria_met}/4 criteria met) – systemic inflammatory response present"
            if criteria_met >= 2
            else f"SIRS negative ({criteria_met}/4 criteria met)"
        ),
    }


# ---------------------------------------------------------------------------
# Lab Trend Analysis
# ---------------------------------------------------------------------------

def detect_lab_trends(
    lab_values: list,
    lab_name: str,
) -> dict:
    """Analyse trend in a time-series of lab values.

    Parameters
    ----------
    lab_values : list of (timestamp, value) tuples
        Chronologically ordered lab measurements. Timestamps may be
        datetime objects or ISO-format strings.
    lab_name : str
        Name of the lab (e.g. "creatinine", "lactate").

    Returns
    -------
    dict with trend direction, rate of change, and concern flag.
    """
    if not lab_values or len(lab_values) < 2:
        return {
            "lab_name": lab_name,
            "trend": "insufficient_data",
            "rate_of_change": None,
            "is_concerning": False,
            "message": f"Not enough data points to determine trend for {lab_name}.",
        }

    # Ensure timestamps are datetime objects and sort chronologically
    parsed: list[tuple[datetime, float]] = []
    for ts, val in lab_values:
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                continue
        try:
            float_val = float(val)
        except (ValueError, TypeError):
            continue  # Skip non-numeric values like '___'
        parsed.append((ts, float_val))
    parsed.sort(key=lambda x: x[0])

    if len(parsed) < 2:
        return {
            "lab_name": lab_name,
            "trend": "insufficient_data",
            "rate_of_change": None,
            "is_concerning": False,
            "message": f"Not enough valid data points for {lab_name}.",
        }

    timestamps = [p[0] for p in parsed]
    values = [p[1] for p in parsed]

    # Convert timestamps to hours from the first measurement
    t0 = timestamps[0]
    hours = np.array([(t - t0).total_seconds() / 3600.0 for t in timestamps])
    vals = np.array(values)

    # Linear regression for trend
    if hours[-1] - hours[0] == 0:
        slope = 0.0
    else:
        coeffs = np.polyfit(hours, vals, 1)
        slope = float(coeffs[0])  # units per hour

    # Determine direction using a relative threshold
    mean_val = float(np.mean(vals))
    relative_threshold = abs(mean_val) * 0.01 if mean_val != 0 else 0.01

    if slope > relative_threshold:
        direction = "rising"
    elif slope < -relative_threshold:
        direction = "falling"
    else:
        direction = "stable"

    # Concern heuristics per lab type
    concern_rules: dict[str, dict] = {
        "creatinine":  {"concerning_direction": "rising",  "critical_rate": 0.3},
        "lactate":     {"concerning_direction": "rising",  "critical_rate": 0.5},
        "bilirubin":   {"concerning_direction": "rising",  "critical_rate": 1.0},
        "platelets":   {"concerning_direction": "falling", "critical_rate": -10.0},
        "wbc":         {"concerning_direction": "rising",  "critical_rate": 2.0},
        "potassium":   {"concerning_direction": "rising",  "critical_rate": 0.5},
        "troponin":    {"concerning_direction": "rising",  "critical_rate": 0.1},
        "inr":         {"concerning_direction": "rising",  "critical_rate": 0.5},
    }

    lab_key = lab_name.lower().strip()
    is_concerning = False
    if lab_key in concern_rules:
        rule = concern_rules[lab_key]
        if direction == rule["concerning_direction"]:
            is_concerning = True
        if abs(slope) >= abs(rule["critical_rate"]):
            is_concerning = True
    else:
        # Generic: flag if changing more than 20% of mean per hour
        if mean_val != 0 and abs(slope) / abs(mean_val) > 0.20:
            is_concerning = True

    pct_change = None
    if values[0] != 0:
        pct_change = round(((values[-1] - values[0]) / abs(values[0])) * 100, 1)

    return {
        "lab_name": lab_name,
        "trend": direction,
        "rate_of_change": round(slope, 4),
        "rate_of_change_unit": f"{lab_name} units/hour",
        "is_concerning": is_concerning,
        "percent_change": pct_change,
        "first_value": values[0],
        "last_value": values[-1],
        "num_data_points": len(values),
        "time_span_hours": round(float(hours[-1] - hours[0]), 1),
        "message": (
            f"{lab_name} is {direction} at {abs(slope):.4f} units/hour "
            f"({'CONCERNING' if is_concerning else 'within expected range'})."
        ),
    }