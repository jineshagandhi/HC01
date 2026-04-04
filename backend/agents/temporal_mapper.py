"""
Temporal Lab Mapper Agent – builds disease progression timelines from vitals, labs, and notes.

Calculates clinical severity scores over time and uses Gemini to generate
narrative disease progression summaries.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import google.generativeai as genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE
from backend.utils.sofa_calculator import (
    calculate_sofa_score,
    calculate_qsofa,
    calculate_sirs,
    detect_lab_trends,
)

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(GEMINI_MODEL)

# ---------------------------------------------------------------------------
# Abnormal vital sign ranges for filtering
# ---------------------------------------------------------------------------
_VITAL_ABNORMAL_RANGES = {
    "HR": {"low": 60, "high": 100, "critical_low": 40, "critical_high": 150},
    "SBP": {"low": 90, "high": 180, "critical_low": 70, "critical_high": 200},
    "DBP": {"low": 60, "high": 110, "critical_low": 40, "critical_high": 120},
    "MAP": {"low": 65, "high": 110, "critical_low": 55, "critical_high": 130},
    "RR": {"low": 12, "high": 20, "critical_low": 8, "critical_high": 35},
    "SpO2": {"low": 94, "high": 100, "critical_low": 88, "critical_high": 101},
    "Temperature": {"low": 36.0, "high": 38.0, "critical_low": 35.0, "critical_high": 40.0},
    "GCS": {"low": 15, "high": 15, "critical_low": 8, "critical_high": 16},
}

# Abnormal lab ranges
_LAB_ABNORMAL_RANGES = {
    "WBC": {"low": 4.0, "high": 11.0, "unit": "x10^3/uL"},
    "Hemoglobin": {"low": 12.0, "high": 17.0, "unit": "g/dL"},
    "Platelets": {"low": 150, "high": 400, "unit": "x10^3/uL"},
    "Creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL"},
    "BUN": {"low": 7, "high": 20, "unit": "mg/dL"},
    "Lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L"},
    "Bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
    "Potassium": {"low": 3.5, "high": 5.0, "unit": "mEq/L"},
    "Sodium": {"low": 136, "high": 145, "unit": "mEq/L"},
    "Glucose": {"low": 70, "high": 180, "unit": "mg/dL"},
    "Troponin": {"low": 0, "high": 0.04, "unit": "ng/mL"},
    "INR": {"low": 0.8, "high": 1.2, "unit": ""},
    "PaO2_FiO2": {"low": 300, "high": 500, "unit": "mmHg"},
}


def _classify_vital_severity(parameter: str, value: float) -> str:
    """Classify a vital sign value as normal, warning, or critical."""
    ranges = _VITAL_ABNORMAL_RANGES.get(parameter)
    if ranges is None:
        return "normal"
    if value <= ranges.get("critical_low", float("-inf")) or value >= ranges.get("critical_high", float("inf")):
        return "critical"
    if value < ranges["low"] or value > ranges["high"]:
        return "warning"
    return "normal"


def _classify_lab_severity(parameter: str, value: float) -> str:
    """Classify a lab value as normal, warning, or critical."""
    ranges = _LAB_ABNORMAL_RANGES.get(parameter)
    if ranges is None:
        return "normal"
    low = ranges["low"]
    high = ranges["high"]
    # Critical if more than 2x outside normal range
    critical_low = low - (high - low)
    critical_high = high + (high - low)
    if value <= critical_low or value >= critical_high:
        return "critical"
    if value < low or value > high:
        return "warning"
    return "normal"


def _is_vital_abnormal(parameter: str, value: float) -> bool:
    """Return True if a vital sign is outside normal range."""
    return _classify_vital_severity(parameter, value) != "normal"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_timeline(
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    notes: list[dict],
    scores: list[dict],
) -> list[dict]:
    """Build a unified disease progression timeline.

    Parameters
    ----------
    vitals_df : DataFrame
        Columns: timestamp, parameter, value.
    labs_df : DataFrame
        Columns: timestamp, parameter, value.
    notes : list of dict
        Each with ``timestamp``, ``text``, and optionally ``note_type``.
    scores : list of dict
        Output from ``calculate_scores_over_time``.

    Returns
    -------
    list of dict – timeline events sorted chronologically.
    Each event: {timestamp, event_type, category, description, value, severity}.
    """
    timeline: list[dict] = []

    # 1. Vital sign events (only abnormal ones)
    if not vitals_df.empty and "timestamp" in vitals_df.columns:
        for _, row in vitals_df.iterrows():
            param = row.get("parameter", "")
            val = row.get("value")
            ts = row.get("timestamp")
            if val is None or ts is None:
                continue
            val = float(val)
            if _is_vital_abnormal(param, val):
                severity = _classify_vital_severity(param, val)
                timeline.append({
                    "timestamp": ts,
                    "event_type": "vital",
                    "category": param,
                    "description": f"{param}: {val}",
                    "value": val,
                    "severity": severity,
                })

    # 2. Lab result events (all of them, mark abnormal)
    if not labs_df.empty and "timestamp" in labs_df.columns:
        for _, row in labs_df.iterrows():
            param = row.get("parameter", "")
            val = row.get("value")
            ts = row.get("timestamp")
            if val is None or ts is None:
                continue
            val = float(val)
            severity = _classify_lab_severity(param, val)
            unit = _LAB_ABNORMAL_RANGES.get(param, {}).get("unit", "")
            desc = f"{param}: {val}"
            if unit:
                desc += f" {unit}"
            if severity != "normal":
                desc += f" [ABNORMAL]"
            timeline.append({
                "timestamp": ts,
                "event_type": "lab",
                "category": param,
                "description": desc,
                "value": val,
                "severity": severity,
            })

    # 3. Clinical note events (summarized)
    for note in notes:
        ts = note.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now()
        elif ts is None:
            ts = datetime.now()

        text = note.get("text", "")
        note_type = note.get("note_type", "Clinical Note")
        # Summarize: first 150 characters
        summary = text[:150].strip()
        if len(text) > 150:
            summary += "..."

        timeline.append({
            "timestamp": ts,
            "event_type": "note",
            "category": note_type,
            "description": summary,
            "value": None,
            "severity": "normal",
        })

    # 4. SOFA/qSOFA score events
    for score_entry in scores:
        ts = score_entry.get("timestamp")
        if ts is None:
            continue
        sofa_total = score_entry.get("sofa_total", 0)
        qsofa_total = score_entry.get("qsofa_total", 0)

        sofa_severity = "normal"
        if sofa_total >= 10:
            sofa_severity = "critical"
        elif sofa_total >= 6:
            sofa_severity = "warning"

        timeline.append({
            "timestamp": ts,
            "event_type": "score",
            "category": "SOFA",
            "description": f"SOFA Score: {sofa_total}/24",
            "value": sofa_total,
            "severity": sofa_severity,
        })

        qsofa_severity = "warning" if qsofa_total >= 2 else "normal"
        timeline.append({
            "timestamp": ts,
            "event_type": "score",
            "category": "qSOFA",
            "description": f"qSOFA Score: {qsofa_total}/3",
            "value": qsofa_total,
            "severity": qsofa_severity,
        })

    # 5. Alert events for concerning trends
    if not labs_df.empty and "parameter" in labs_df.columns:
        alert_labs = {
            "WBC": ["wbc", "white blood"],
            "Lactate": ["lactate"],
            "Creatinine": ["creatinine"],
            "Platelets": ["platelet"],
            "Bilirubin": ["bilirubin"],
        }
        for lab_name, search_terms in alert_labs.items():
            lab_mask = labs_df["parameter"].str.lower().apply(
                lambda x: any(term in str(x).lower() for term in search_terms)
            )
            lab_data = labs_df.loc[lab_mask].sort_values("timestamp")
            if len(lab_data) < 2:
                continue
            values_list = list(zip(lab_data["timestamp"], lab_data["value"]))
            trend = detect_lab_trends(values_list, lab_name)
            if trend.get("is_concerning", False):
                latest_ts = lab_data["timestamp"].max()
                timeline.append({
                    "timestamp": latest_ts,
                    "event_type": "alert",
                    "category": lab_name,
                    "description": f"ALERT: {trend['message']}",
                    "value": trend.get("last_value"),
                    "severity": "critical" if trend["trend"] == "rising" and lab_name == "Lactate" else "warning",
                })

    # Sort by timestamp
    timeline.sort(key=lambda e: e["timestamp"] if isinstance(e["timestamp"], datetime) else datetime.min)

    return timeline


def calculate_scores_over_time(
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
) -> list[dict]:
    """Calculate SOFA, qSOFA, and SIRS scores at each available timepoint.

    Parameters
    ----------
    vitals_df : DataFrame
        Columns: timestamp, parameter, value.
    labs_df : DataFrame
        Columns: timestamp, parameter, value.

    Returns
    -------
    list of dict – one entry per timepoint with all three scores.
    """
    # Collect all unique timestamps
    timestamps: set[datetime] = set()
    if not vitals_df.empty and "timestamp" in vitals_df.columns:
        timestamps.update(vitals_df["timestamp"].dropna().unique())
    if not labs_df.empty and "timestamp" in labs_df.columns:
        timestamps.update(labs_df["timestamp"].dropna().unique())

    if not timestamps:
        return []

    # Convert numpy datetime64 to Python datetime if needed
    clean_timestamps: list[datetime] = []
    for ts in timestamps:
        if isinstance(ts, (pd.Timestamp, np.datetime64)):
            ts = pd.Timestamp(ts).to_pydatetime()
        clean_timestamps.append(ts)
    clean_timestamps.sort()

    # Sample at reasonable intervals (every 6 hours or all points if fewer than 20)
    if len(clean_timestamps) > 20:
        step = max(1, len(clean_timestamps) // 20)
        sampled = clean_timestamps[::step]
        if clean_timestamps[-1] not in sampled:
            sampled.append(clean_timestamps[-1])
    else:
        sampled = clean_timestamps

    results: list[dict] = []

    for ts in sampled:
        window = timedelta(hours=24)
        # Ensure ts is pd.Timestamp for comparison with DataFrame timestamps
        ts_pd = pd.Timestamp(ts)
        window_start = ts_pd - window

        def _latest(df: pd.DataFrame, param: str, _ts=ts_pd, _ws=window_start) -> Optional[float]:
            if df.empty or "parameter" not in df.columns:
                return None
            try:
                # Ensure timestamp column is comparable
                timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
                mask_time = (timestamps >= _ws) & (timestamps <= _ts)
            except Exception:
                # If time filtering fails, use all data
                mask_time = pd.Series(True, index=df.index)

            # Try exact match first, then case-insensitive contains
            subset = df.loc[mask_time & (df["parameter"] == param)]
            if subset.empty:
                subset = df.loc[mask_time & df["parameter"].str.contains(param, case=False, na=False)]
            if subset.empty:
                # Last resort: try without time filter
                subset = df.loc[df["parameter"].str.contains(param, case=False, na=False)]
            if subset.empty:
                return None
            # Get most recent value
            try:
                if "timestamp" in subset.columns:
                    subset_sorted = subset.sort_values("timestamp", ascending=False)
                else:
                    subset_sorted = subset
                val = subset_sorted.iloc[0]["value"]
                return float(val)
            except (ValueError, TypeError, IndexError):
                return None

        def _latest_flag(df: pd.DataFrame, param: str) -> bool:
            val = _latest(df, param)
            return bool(val) if val is not None else False

        # SOFA components — try multiple MIMIC-IV label variants
        # Use explicit None checks (not `or` which fails on 0.0 values)
        platelets = _latest(labs_df, "Platelet")
        if platelets is None:
            platelets = _latest(labs_df, "Platelets")
        bilirubin = _latest(labs_df, "Bilirubin")
        creatinine = _latest(labs_df, "Creatinine")
        gcs_val = _latest(vitals_df, "GCS")

        # Calculate MAP from SBP/DBP if not directly available
        sbp_val = _latest(vitals_df, "SBP")
        dbp_val = _latest(vitals_df, "DBP")
        map_value = None
        if sbp_val is not None and dbp_val is not None:
            map_value = dbp_val + (sbp_val - dbp_val) / 3.0

        # Estimate PaO2/FiO2 ratio — use SpO2 as proxy if PaO2 not available
        pao2_fio2 = _latest(labs_df, "pO2")
        if pao2_fio2 is None:
            pao2_fio2 = _latest(labs_df, "PaO2")
        if pao2_fio2 is None:
            # Use SpO2 as rough proxy: SpO2 ~97% -> PF ~400, SpO2 ~90% -> PF ~200
            spo2_val = _latest(vitals_df, "SpO2")
            if spo2_val is not None:
                if spo2_val >= 96:
                    pao2_fio2 = 400
                elif spo2_val >= 92:
                    pao2_fio2 = 300
                elif spo2_val >= 88:
                    pao2_fio2 = 200
                else:
                    pao2_fio2 = 100

        sofa = calculate_sofa_score(
            pao2_fio2_ratio=pao2_fio2,
            platelets=platelets,
            bilirubin=bilirubin,
            map_value=map_value,
            gcs=int(gcs_val) if gcs_val is not None else None,
            creatinine=creatinine,
            on_vasopressors=False,
            on_ventilation=False,
        )

        # qSOFA
        sbp = _latest(vitals_df, "SBP")
        rr = _latest(vitals_df, "RR")
        if rr is None:
            rr = _latest(vitals_df, "Resp")
        qsofa = calculate_qsofa(
            sbp=sbp,
            rr=rr,
            gcs=int(gcs_val) if gcs_val is not None else None,
        )

        # SIRS
        temp = _latest(vitals_df, "Temperature")
        if temp is None:
            temp = _latest(vitals_df, "Temp")
        hr = _latest(vitals_df, "HR")
        if hr is None:
            hr = _latest(vitals_df, "Heart Rate")
        wbc = _latest(labs_df, "White Blood")
        if wbc is None:
            wbc = _latest(labs_df, "WBC")
        paco2 = _latest(labs_df, "pCO2")
        if paco2 is None:
            paco2 = _latest(labs_df, "PaCO2")
        sirs = calculate_sirs(
            temp=temp,
            hr=hr,
            rr=rr,
            wbc=wbc,
            paco2=paco2,
        )

        results.append({
            "timestamp": ts,
            "sofa_total": sofa["total"],
            "sofa_components": sofa["components"],
            "qsofa_total": qsofa["total"],
            "qsofa_components": qsofa["components"],
            "sirs_total": sirs["total"],
            "sirs_components": sirs["components"],
        })

    return results


def detect_all_trends(labs_df: pd.DataFrame) -> list[dict]:
    """Detect trends for key ICU lab values.

    Parameters
    ----------
    labs_df : DataFrame
        Columns: timestamp, parameter, value.

    Returns
    -------
    list of dict – one per lab with trend info.
    """
    # Map display name -> search patterns (MIMIC-IV uses verbose labels)
    key_labs = {
        "WBC": ["wbc", "white blood"],
        "Lactate": ["lactate"],
        "Creatinine": ["creatinine"],
        "Platelets": ["platelet"],
        "Bilirubin": ["bilirubin"],
    }
    trends: list[dict] = []

    if labs_df.empty or "parameter" not in labs_df.columns:
        return trends

    for lab_name, search_terms in key_labs.items():
        # Search using contains (case-insensitive) for any of the search terms
        mask = labs_df["parameter"].str.lower().apply(
            lambda x: any(term in str(x).lower() for term in search_terms)
        )
        lab_data = labs_df.loc[mask].sort_values("timestamp")

        if len(lab_data) < 2:
            trends.append({
                "lab_name": lab_name,
                "trend": "insufficient_data",
                "values": [],
                "is_concerning": False,
                "description": f"Not enough data points for {lab_name} trend analysis.",
            })
            continue

        values_list = list(zip(lab_data["timestamp"], lab_data["value"]))
        trend_result = detect_lab_trends(values_list, lab_name)

        values_for_display = [
            {"timestamp": ts, "value": float(val)}
            for ts, val in values_list
        ]

        trends.append({
            "lab_name": lab_name,
            "trend": trend_result["trend"],
            "values": values_for_display,
            "is_concerning": trend_result["is_concerning"],
            "description": trend_result["message"],
        })

    return trends


def get_disease_progression_summary(
    timeline: list[dict],
    scores: list[dict],
    trends: list[dict],
) -> str:
    """Generate a narrative disease progression summary using Gemini.

    Parameters
    ----------
    timeline : list of dict
        Output from ``build_timeline``.
    scores : list of dict
        Output from ``calculate_scores_over_time``.
    trends : list of dict
        Output from ``detect_all_trends``.

    Returns
    -------
    str – narrative summary of disease progression.
    """
    # Build context for Gemini
    score_summary = ""
    if scores:
        first_score = scores[0]
        last_score = scores[-1]
        score_summary = (
            f"SOFA score changed from {first_score['sofa_total']} to {last_score['sofa_total']} "
            f"over the observation period. "
            f"qSOFA changed from {first_score['qsofa_total']} to {last_score['qsofa_total']}. "
            f"SIRS criteria met changed from {first_score['sirs_total']} to {last_score['sirs_total']}."
        )

    trend_summary = ""
    concerning_trends = [t for t in trends if t.get("is_concerning", False)]
    if concerning_trends:
        parts = []
        for t in concerning_trends:
            vals = t.get("values", [])
            if vals:
                first_val = vals[0]["value"] if isinstance(vals[0], dict) else vals[0][1]
                last_val = vals[-1]["value"] if isinstance(vals[-1], dict) else vals[-1][1]
                parts.append(f"{t['lab_name']}: {first_val} -> {last_val} ({t['trend']})")
        trend_summary = "Concerning lab trends: " + "; ".join(parts)

    # Count critical events
    critical_events = [e for e in timeline if e.get("severity") == "critical"]
    alert_events = [e for e in timeline if e.get("event_type") == "alert"]

    # Calculate time span
    time_span_str = "unknown duration"
    if timeline:
        timestamps = [
            e["timestamp"] for e in timeline
            if isinstance(e.get("timestamp"), datetime)
        ]
        if timestamps:
            span = max(timestamps) - min(timestamps)
            hours = span.total_seconds() / 3600
            time_span_str = f"{hours:.0f} hours"

    prompt = f"""You are a critical care physician summarizing a patient's disease progression.

Clinical Data:
- Observation period: {time_span_str}
- {score_summary}
- {trend_summary}
- Number of critical events: {len(critical_events)}
- Number of clinical alerts: {len(alert_events)}
- Alert details: {'; '.join(e['description'] for e in alert_events[:5])}

Write a concise 3-5 sentence clinical narrative describing the disease progression.
Focus on trajectory (improving, stable, or deteriorating), key drivers of organ dysfunction,
and the most clinically significant changes. Use precise clinical language.
Do not include recommendations - only describe what happened."""

    try:
        response = _model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=LLM_TEMPERATURE),
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning("Gemini disease progression summary failed: %s", exc)

        # Fallback: generate a basic summary from the data
        parts = []
        parts.append(f"Observation period: {time_span_str}.")

        if scores:
            first = scores[0]
            last = scores[-1]
            sofa_direction = "increased" if last["sofa_total"] > first["sofa_total"] else \
                             "decreased" if last["sofa_total"] < first["sofa_total"] else "remained stable"
            parts.append(
                f"SOFA score {sofa_direction} from {first['sofa_total']} to {last['sofa_total']}."
            )

        if concerning_trends:
            parts.append(
                f"Concerning trends detected in: {', '.join(t['lab_name'] for t in concerning_trends)}."
            )

        if critical_events:
            parts.append(f"{len(critical_events)} critical events recorded.")

        return " ".join(parts)