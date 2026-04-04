"""
Statistical outlier detection for ICU lab values.

Uses z-score based detection with configurable thresholds
to flag potentially erroneous or clinically significant lab results.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def detect_outlier(
    values: list[float],
    new_value: float,
    z_threshold: float = 3.0,
) -> dict:
    """Detect whether a new value is an outlier relative to a series.

    Parameters
    ----------
    values : list of float
        Historical values to establish the baseline distribution.
    new_value : float
        The new observation to evaluate.
    z_threshold : float
        Number of standard deviations from the mean to flag as outlier.

    Returns
    -------
    dict with is_outlier, z_score, mean, std, recommendation.
    """
    if not values:
        return {
            "is_outlier": False,
            "z_score": 0.0,
            "mean": new_value,
            "std": 0.0,
            "recommendation": "Insufficient historical data to evaluate outlier status.",
        }

    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    if std == 0.0:
        # All historical values are identical
        is_outlier = new_value != mean
        z_score = float("inf") if is_outlier else 0.0
        recommendation = (
            f"All prior values are identical ({mean}). "
            f"New value {new_value} differs and should be verified."
            if is_outlier
            else "Value is consistent with prior measurements."
        )
        return {
            "is_outlier": is_outlier,
            "z_score": z_score,
            "mean": mean,
            "std": std,
            "recommendation": recommendation,
        }

    z_score = float((new_value - mean) / std)
    is_outlier = abs(z_score) >= z_threshold

    if is_outlier:
        recommendation = (
            f"Value of {new_value} is {abs(z_score):.1f} standard deviations from the "
            f"mean of {mean:.1f} (std={std:.2f}). "
            f"This may represent a lab error. Recommend redraw to confirm."
        )
    else:
        recommendation = "Value is within expected range."

    return {
        "is_outlier": is_outlier,
        "z_score": round(z_score, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "recommendation": recommendation,
    }


def detect_outliers_in_series(
    lab_series: list,
    window_hours: int = 72,
) -> list[dict]:
    """Scan a time-ordered lab series and flag outliers within a rolling window.

    Parameters
    ----------
    lab_series : list of (timestamp, value) tuples
        Chronologically ordered. Timestamps can be datetime objects or
        ISO-format strings.
    window_hours : int
        Lookback window in hours for establishing the baseline.

    Returns
    -------
    list of dicts – one entry per detected outlier point, each containing
    timestamp, value, z_score, mean, std, and is_outlier flag.
    """
    if not lab_series or len(lab_series) < 2:
        return []

    # Parse and sort
    parsed: list[tuple[datetime, float]] = []
    for ts, val in lab_series:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        parsed.append((ts, float(val)))
    parsed.sort(key=lambda x: x[0])

    outliers: list[dict] = []
    window_delta = timedelta(hours=window_hours)

    for i, (ts, val) in enumerate(parsed):
        # Collect values within the lookback window *before* this point
        window_start = ts - window_delta
        baseline_values = [
            v for t, v in parsed[:i] if t >= window_start
        ]

        if len(baseline_values) < 3:
            # Not enough prior data to reliably detect outliers
            continue

        result = detect_outlier(baseline_values, val)

        if result["is_outlier"]:
            outliers.append({
                "timestamp": ts,
                "value": val,
                "z_score": result["z_score"],
                "mean": result["mean"],
                "std": result["std"],
                "is_outlier": True,
                "window_hours": window_hours,
                "baseline_count": len(baseline_values),
            })

    return outliers


def format_outlier_alert(
    outlier_info: dict,
    lab_name: str,
) -> str:
    """Format an outlier detection result into a human-readable clinical alert.

    Parameters
    ----------
    outlier_info : dict
        Output from detect_outlier or an entry from detect_outliers_in_series.
    lab_name : str
        Name of the lab value (e.g. "Potassium", "Creatinine").

    Returns
    -------
    str – formatted alert message.
    """
    value = outlier_info.get("value", outlier_info.get("z_score", "N/A"))
    # If called from detect_outlier output directly, value might not be present;
    # try to reconstruct from z_score, mean, std.
    if "value" not in outlier_info:
        mean = outlier_info.get("mean", 0)
        std = outlier_info.get("std", 0)
        z = outlier_info.get("z_score", 0)
        value = mean + z * std

    z_score = outlier_info.get("z_score", 0.0)
    mean = outlier_info.get("mean", 0.0)
    window = outlier_info.get("window_hours", 72)

    if not outlier_info.get("is_outlier", False):
        return f"{lab_name}: Value within expected range (z-score: {z_score:.1f})."

    return (
        f"Value of {value} for {lab_name} is {abs(z_score):.1f} standard deviations "
        f"from the {window}-hour mean of {mean:.1f}. This may represent a lab error. "
        f"Recommend redraw to confirm before updating clinical assessment."
    )