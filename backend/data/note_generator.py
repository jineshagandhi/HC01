"""
Clinical Note Generator for ICU Diagnostic Risk Assistant.

Generates realistic ICU clinical notes based on patient data patterns,
simulating worsening sepsis scenarios across a multi-day ICU stay.
Since MIMIC-IV clinical notes require credentialed access, this module
produces synthetic notes with authentic medical terminology and formatting.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Staff rosters
# ---------------------------------------------------------------------------
ATTENDING_PHYSICIANS = ["Dr. Sharma", "Dr. Patel", "Dr. Mehta", "Dr. Reddy", "Dr. Gupta"]
RESIDENTS = ["Dr. Rao", "Dr. Iyer", "Dr. Joshi", "Dr. Nair", "Dr. Das"]
NURSES = ["Nurse Priya", "Nurse Anita", "Nurse Rajesh", "Nurse Kavita", "Nurse Deepa"]

# ---------------------------------------------------------------------------
# Vital-sign and lab trajectories keyed by ICU day (1-indexed)
# ---------------------------------------------------------------------------
VITALS_BY_DAY = {
    1: {  # Admission — relatively stable
        "hr": (78, 95), "sbp": (110, 130), "dbp": (65, 80),
        "rr": (14, 20), "temp": (36.8, 37.6), "spo2": (95, 99),
    },
    2: {  # Early signs — mild tachycardia, slight fever
        "hr": (90, 108), "sbp": (100, 120), "dbp": (60, 75),
        "rr": (18, 24), "temp": (37.4, 38.4), "spo2": (93, 97),
    },
    3: {  # Concerning — fever, rising RR, BP dipping
        "hr": (100, 120), "sbp": (88, 105), "dbp": (52, 68),
        "rr": (22, 28), "temp": (38.0, 39.2), "spo2": (91, 95),
    },
    4: {  # Critical — organ dysfunction emerging
        "hr": (115, 140), "sbp": (75, 95), "dbp": (42, 58),
        "rr": (26, 34), "temp": (38.5, 39.8), "spo2": (87, 93),
    },
    5: {  # Mixed — improving or deteriorating branch
        "improving": {
            "hr": (88, 105), "sbp": (100, 118), "dbp": (58, 72),
            "rr": (18, 24), "temp": (37.2, 38.0), "spo2": (94, 98),
        },
        "worsening": {
            "hr": (125, 150), "sbp": (68, 85), "dbp": (38, 52),
            "rr": (30, 38), "temp": (39.0, 40.2), "spo2": (84, 90),
        },
    },
}

LABS_BY_DAY = {
    1: {
        "wbc": (8.5, 11.0), "lactate": (1.0, 1.8), "creatinine": (0.8, 1.2),
        "platelets": (180, 320), "bilirubin": (0.5, 1.0), "hemoglobin": (11.5, 14.0),
        "sodium": (136, 144), "potassium": (3.6, 4.8), "bicarbonate": (22, 28),
        "bun": (10, 22), "inr": (0.9, 1.1), "fibrinogen": (250, 400),
        "procalcitonin": (0.1, 0.5), "crp": (5, 30),
    },
    2: {
        "wbc": (11.5, 15.0), "lactate": (1.6, 2.6), "creatinine": (1.0, 1.5),
        "platelets": (140, 220), "bilirubin": (0.8, 1.5), "hemoglobin": (10.5, 13.0),
        "sodium": (134, 143), "potassium": (3.5, 5.0), "bicarbonate": (20, 26),
        "bun": (15, 30), "inr": (1.0, 1.3), "fibrinogen": (220, 380),
        "procalcitonin": (0.5, 2.0), "crp": (30, 80),
    },
    3: {
        "wbc": (14.0, 20.0), "lactate": (2.4, 4.2), "creatinine": (1.4, 2.2),
        "platelets": (100, 160), "bilirubin": (1.2, 2.5), "hemoglobin": (9.5, 12.0),
        "sodium": (132, 142), "potassium": (3.8, 5.3), "bicarbonate": (18, 24),
        "bun": (22, 42), "inr": (1.2, 1.6), "fibrinogen": (180, 320),
        "procalcitonin": (2.0, 10.0), "crp": (80, 180),
    },
    4: {
        "wbc": (18.0, 30.0), "lactate": (3.8, 6.5), "creatinine": (2.0, 3.5),
        "platelets": (55, 115), "bilirubin": (2.0, 4.5), "hemoglobin": (8.5, 11.0),
        "sodium": (130, 140), "potassium": (4.0, 5.8), "bicarbonate": (15, 21),
        "bun": (30, 58), "inr": (1.4, 2.0), "fibrinogen": (120, 250),
        "procalcitonin": (8.0, 40.0), "crp": (150, 300),
    },
    5: {
        "improving": {
            "wbc": (12.0, 16.0), "lactate": (1.8, 3.0), "creatinine": (1.5, 2.2),
            "platelets": (90, 150), "bilirubin": (1.5, 2.8), "hemoglobin": (9.0, 11.5),
            "sodium": (134, 142), "potassium": (3.6, 5.0), "bicarbonate": (19, 25),
            "bun": (25, 45), "inr": (1.2, 1.5), "fibrinogen": (180, 320),
            "procalcitonin": (4.0, 12.0), "crp": (100, 200),
        },
        "worsening": {
            "wbc": (25.0, 42.0), "lactate": (5.5, 9.0), "creatinine": (3.0, 5.0),
            "platelets": (25, 60), "bilirubin": (4.0, 8.0), "hemoglobin": (7.0, 9.5),
            "sodium": (128, 138), "potassium": (4.5, 6.2), "bicarbonate": (12, 18),
            "bun": (45, 80), "inr": (1.8, 2.8), "fibrinogen": (80, 160),
            "procalcitonin": (30.0, 80.0), "crp": (250, 400),
        },
    },
}

# SOFA component scoring helpers
SOFA_BY_DAY = {1: 2, 2: 4, 3: 7, 4: 10, 5: {"improving": 6, "worsening": 13}}

DIAGNOSES_TEMPLATES = {
    "sepsis": {
        "chief_complaint": [
            "fever and altered mental status",
            "hypotension and tachycardia",
            "febrile illness with respiratory distress",
        ],
        "sources": ["pneumonia", "urinary tract infection", "intra-abdominal source",
                     "line-related bacteremia", "soft tissue infection"],
        "organisms": ["E. coli", "Klebsiella pneumoniae", "Staphylococcus aureus",
                       "Pseudomonas aeruginosa", "Streptococcus pneumoniae"],
        "antibiotics": [
            "Meropenem 1g IV q8h + Vancomycin 1g IV q12h",
            "Piperacillin-Tazobactam 4.5g IV q6h + Vancomycin 1g IV q12h",
            "Cefepime 2g IV q8h + Metronidazole 500mg IV q8h",
        ],
    },
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rand(low: float, high: float, decimals: int = 1) -> float:
    return round(random.uniform(low, high), decimals)


def _pick(lst):
    return random.choice(lst)


def _trend_arrow(current: float, previous: float) -> str:
    if current > previous * 1.05:
        return "↑"
    elif current < previous * 0.95:
        return "↓"
    return "→"


def _sample_vitals(day: int, outcome: str = "worsening") -> Dict[str, float]:
    ranges = VITALS_BY_DAY[day]
    if isinstance(ranges, dict) and "improving" in ranges:
        ranges = ranges[outcome]
    return {
        "hr": _rand(*ranges["hr"], 0),
        "sbp": _rand(*ranges["sbp"], 0),
        "dbp": _rand(*ranges["dbp"], 0),
        "rr": _rand(*ranges["rr"], 0),
        "temp": _rand(*ranges["temp"], 1),
        "spo2": _rand(*ranges["spo2"], 0),
    }


def _sample_labs(day: int, outcome: str = "worsening") -> Dict[str, float]:
    ranges = LABS_BY_DAY[day]
    if isinstance(ranges, dict) and "improving" in ranges:
        ranges = ranges[outcome]
    return {k: _rand(*v, 1) for k, v in ranges.items()}


def _sofa_score(day: int, outcome: str = "worsening") -> int:
    s = SOFA_BY_DAY[day]
    if isinstance(s, dict):
        return s[outcome]
    return s


def _qsofa(vitals: Dict[str, float]) -> int:
    score = 0
    if vitals["sbp"] <= 100:
        score += 1
    if vitals["rr"] >= 22:
        score += 1
    # Altered mentation approximated by day severity
    return score


def _o2_device(spo2: float) -> str:
    if spo2 >= 96:
        return "room air"
    elif spo2 >= 93:
        return f"{random.randint(2, 4)}L NC"
    elif spo2 >= 90:
        return f"{random.randint(4, 6)}L NC"
    elif spo2 >= 87:
        return f"{random.randint(6, 10)}L high-flow NC"
    else:
        return "non-rebreather mask 15L"


def _gcs(day: int) -> tuple:
    """Return (total, E, V, M) GCS based on day severity."""
    mapping = {
        1: (15, 4, 5, 6),
        2: (14, 4, 4, 6),
        3: (13, 3, 4, 6),
        4: (11, 3, 3, 5),
        5: (13, 3, 4, 6),  # default; overridden by outcome
    }
    return mapping.get(day, (15, 4, 5, 6))


def _mental_status(day: int) -> str:
    options = {
        1: "Alert and oriented x4. Cooperative.",
        2: "Alert, oriented x3. Mildly anxious.",
        3: "Alert but intermittently confused. Oriented to person and place only.",
        4: "Drowsy, arousable to voice. Oriented to person only. Intermittent agitation.",
        5: "Variable — see neuro exam.",
    }
    return options.get(day, "Alert and oriented.")


def _subjective(day: int, diagnosis: str) -> str:
    templates = {
        1: [
            "Patient reports feeling unwell for the past 2-3 days. Describes subjective fevers, "
            "chills, and generalized malaise. Endorses productive cough with yellowish sputum. "
            "Denies chest pain or hemoptysis.",
            "Patient states 'I've been feeling weak and achy.' Reports decreased oral intake "
            "over past 48 hours. Denies recent travel or sick contacts.",
        ],
        2: [
            "Patient reports mild improvement in energy but still feeling 'off.' Complains of "
            "persistent low-grade fevers. Tolerating clear liquids. Denies new symptoms.",
            "Patient endorses continued fatigue and mild dyspnea with exertion. Reports sleeping "
            "poorly overnight. Tolerating tube feeds at goal rate.",
        ],
        3: [
            "Patient reports worsening shortness of breath. 'I can't catch my breath.' Denies "
            "chest pain. Complains of feeling cold despite fever. Tolerating tube feeds.",
            "Patient appears more confused per nursing. Intermittently oriented. Reports mild "
            "abdominal discomfort. Denies nausea or vomiting.",
        ],
        4: [
            "Limited history obtainable — patient drowsy and intermittently confused. Nurse "
            "reports decreased urine output overnight. Patient moans with repositioning.",
            "Patient unable to provide reliable history. Per family, patient was 'not making "
            "sense' overnight. Appears uncomfortable. Tube feeds held due to abdominal distension.",
        ],
        5: [
            "Patient reports feeling 'a little better today.' More alert, able to engage in "
            "conversation. Tolerating oral sips. Denies chest pain or dyspnea at rest.",
            "Patient remains obtunded. Minimal response to verbal stimuli. Family at bedside, "
            "distressed. No subjective complaints elicitable.",
        ],
    }
    return _pick(templates.get(day, templates[1]))


def _physical_exam(day: int, vitals: Dict[str, float]) -> str:
    o2 = _o2_device(vitals["spo2"])
    gcs_total, gcs_e, gcs_v, gcs_m = _gcs(day)

    # Cardiovascular findings by severity
    cvs_map = {
        1: "Regular rate and rhythm. S1/S2 normal. No murmurs, rubs, or gallops. Peripheral pulses 2+ bilaterally.",
        2: "Mildly tachycardic, regular rhythm. S1/S2 normal. No murmurs. Peripheral pulses 2+ bilaterally.",
        3: "Tachycardic, regular rhythm. No murmurs. Capillary refill 3 seconds. Peripheral pulses 1+ distally.",
        4: "Tachycardic, regular rhythm. S1 diminished. Capillary refill 4-5 seconds. Extremities cool and mottled. "
           "Weak peripheral pulses.",
        5: "Rate controlled, regular rhythm. Capillary refill 2-3 seconds. Extremities warm.",
    }

    resp_map = {
        1: "Clear to auscultation bilaterally. No wheezes, rhonchi, or rales. Normal respiratory effort.",
        2: "Scattered rhonchi bilaterally, R>L. Mild accessory muscle use. On {o2}.",
        3: "Bilateral crackles, L>R base. Moderate accessory muscle use. On {o2}. Productive cough noted.",
        4: "Diffuse bilateral crackles. Significant accessory muscle use. Labored breathing on {o2}. "
           "Discussed intubation threshold with team.",
        5: "Bilateral crackles improving from prior. Decreased accessory muscle use. On {o2}.",
    }

    abd_map = {
        1: "Soft, non-tender, non-distended. Bowel sounds present in all 4 quadrants.",
        2: "Soft, mildly distended. Non-tender. Bowel sounds present but hypoactive.",
        3: "Soft, mildly tender in RUQ. Non-distended. Bowel sounds hypoactive.",
        4: "Distended, diffusely tender to deep palpation. Guarding present. Bowel sounds absent.",
        5: "Soft, mildly distended. Improving tenderness. Bowel sounds returning.",
    }

    neuro_status = _mental_status(day)

    lines = [
        f"General: {_general_appearance(day)}",
        f"Vitals: T {vitals['temp']}°C, HR {int(vitals['hr'])}, "
        f"BP {int(vitals['sbp'])}/{int(vitals['dbp'])}, RR {int(vitals['rr'])}, "
        f"SpO2 {int(vitals['spo2'])}% on {o2}",
        f"CVS: {cvs_map.get(day, cvs_map[1])}",
        f"Resp: {resp_map.get(day, resp_map[1]).format(o2=o2)}",
        f"Abd: {abd_map.get(day, abd_map[1])}",
        f"Neuro: GCS {gcs_total} (E{gcs_e}V{gcs_v}M{gcs_m}). {neuro_status}",
        f"Skin: {_skin_exam(day)}",
        f"Lines/Drains: {_lines_drains(day)}",
    ]
    return "\n".join(lines)


def _general_appearance(day: int) -> str:
    options = {
        1: "Ill-appearing but in no acute distress. Awake, cooperative.",
        2: "Mildly ill-appearing. Fatigued but interactive.",
        3: "Moderately ill-appearing. Diaphoretic. Appears uncomfortable.",
        4: "Critically ill-appearing. Diaphoretic, pale, lethargic. Minimal interaction.",
        5: "Ill-appearing but improved from yesterday. More alert.",
    }
    return options.get(day, options[1])


def _skin_exam(day: int) -> str:
    options = {
        1: "Warm, dry. No rashes or lesions. IV site clean without erythema.",
        2: "Warm, slightly diaphoretic. No rashes. IV site clean.",
        3: "Warm, diaphoretic. Mild peripheral edema bilateral lower extremities.",
        4: "Cool, mottled extremities. 2+ pitting edema bilateral lower extremities. Livedo reticularis noted.",
        5: "Warm, dry. Mottling resolved. Edema improving.",
    }
    return options.get(day, options[1])


def _lines_drains(day: int) -> str:
    options = {
        1: "R IJ triple-lumen CVC (day 0) — clean, no erythema. 18G PIV L forearm. Foley catheter draining clear yellow urine.",
        2: "R IJ triple-lumen CVC (day 1) — clean, dressing intact. Foley catheter in place. A-line R radial.",
        3: "R IJ triple-lumen CVC (day 2) — dressing intact. Foley catheter draining concentrated urine. A-line R radial.",
        4: "R IJ triple-lumen CVC (day 3) — consider line change if source not identified. "
           "Foley — low urine output (15 mL/hr). A-line R radial. OG tube in place.",
        5: "R IJ triple-lumen CVC (day 4) — changed to L SC CVC today. Foley — UOP improving to 30 mL/hr. A-line R radial.",
    }
    return options.get(day, options[1])


def _format_labs(labs: Dict[str, float], prev_labs: Optional[Dict[str, float]]) -> str:
    """Format lab values with trend arrows when previous values available."""
    key_labs = [
        ("WBC", "wbc", "K/uL"),
        ("Hgb", "hemoglobin", "g/dL"),
        ("Plt", "platelets", "K/uL"),
        ("Na", "sodium", "mEq/L"),
        ("K", "potassium", "mEq/L"),
        ("HCO3", "bicarbonate", "mEq/L"),
        ("BUN", "bun", "mg/dL"),
        ("Cr", "creatinine", "mg/dL"),
        ("Lactate", "lactate", "mmol/L"),
        ("T.Bili", "bilirubin", "mg/dL"),
        ("INR", "inr", ""),
        ("Fibrinogen", "fibrinogen", "mg/dL"),
        ("Procalcitonin", "procalcitonin", "ng/mL"),
        ("CRP", "crp", "mg/L"),
    ]
    lines = []
    # Group into CBC, BMP, Other
    cbc_keys = {"wbc", "hemoglobin", "platelets"}
    bmp_keys = {"sodium", "potassium", "bicarbonate", "bun", "creatinine"}

    def _fmt(label, key, unit):
        val = labs[key]
        s = f"{label} {val}"
        if unit:
            s += f" {unit}"
        if prev_labs and key in prev_labs:
            arrow = _trend_arrow(val, prev_labs[key])
            s += f" ({arrow} from {prev_labs[key]})"
        return s

    cbc_items = [_fmt(l, k, u) for l, k, u in key_labs if k in cbc_keys]
    bmp_items = [_fmt(l, k, u) for l, k, u in key_labs if k in bmp_keys]
    other_items = [_fmt(l, k, u) for l, k, u in key_labs if k not in cbc_keys and k not in bmp_keys]

    lines.append("CBC: " + ", ".join(cbc_items))
    lines.append("BMP: " + ", ".join(bmp_items))
    lines.append("Other: " + ", ".join(other_items))
    return "\n".join(lines)


def _assessment_plan(day: int, vitals: Dict[str, float], labs: Dict[str, float],
                     prev_labs: Optional[Dict[str, float]], diagnosis_info: dict,
                     sofa: int, prev_sofa: Optional[int], outcome: str) -> str:
    """Generate the A/P section of the progress note."""
    source = _pick(diagnosis_info.get("sources", ["unknown source"]))
    abx = _pick(diagnosis_info.get("antibiotics", ["broad-spectrum antibiotics"]))

    sofa_trend = ""
    if prev_sofa is not None:
        arrow = _trend_arrow(sofa, prev_sofa)
        sofa_trend = f" ({arrow} from {prev_sofa} yesterday)"

    qsofa = _qsofa(vitals)
    qsofa_components = []
    if vitals["sbp"] <= 100:
        qsofa_components.append("low BP")
    if vitals["rr"] >= 22:
        qsofa_components.append("high RR")
    if day >= 3:
        qsofa += 1  # altered mentation
        qsofa_components.append("altered mentation")
    qsofa = min(qsofa, 3)
    qsofa_str = f"qSOFA: {qsofa}/3"
    if qsofa_components:
        qsofa_str += f" ({', '.join(qsofa_components)})"

    problems = []

    # Problem 1: Sepsis / Septic Shock
    if day <= 2:
        problems.append(
            f"1. Sepsis — Likely {source} source. Early presentation, SOFA {sofa}{sofa_trend}.\n"
            f"   - Initiated empiric {abx}\n"
            f"   - Blood cultures x2 drawn prior to antibiotics\n"
            f"   - 30 mL/kg crystalloid bolus administered\n"
            f"   - Target MAP ≥65 mmHg\n"
            f"   - Repeat lactate in 6 hours\n"
            f"   - Source control imaging pending (CT abdomen/pelvis if indicated)"
        )
    elif day == 3:
        problems.append(
            f"1. Sepsis — Likely {source} source. SOFA {sofa}{sofa_trend}, trending up concerning "
            f"for worsening organ dysfunction.\n"
            f"   - Continue {abx}\n"
            f"   - Repeat blood cultures drawn\n"
            f"   - Target MAP ≥65, consider vasopressor initiation (norepinephrine) if not "
            f"responsive to fluid resuscitation\n"
            f"   - Lactate clearance suboptimal — repeat q6h\n"
            f"   - ID consult for abx optimization\n"
            f"   - Stress-dose hydrocortisone 50mg IV q6h if vasopressor-dependent"
        )
    elif day == 4:
        problems.append(
            f"1. Septic Shock — {source} source. SOFA {sofa}{sofa_trend}. Multi-organ dysfunction "
            f"evolving.\n"
            f"   - Norepinephrine started at 0.05 mcg/kg/min, titrate to MAP ≥65\n"
            f"   - Add vasopressin 0.04 units/min if norepinephrine >0.2 mcg/kg/min\n"
            f"   - Continue {abx} — adjust per sensitivity data when available\n"
            f"   - Stress-dose hydrocortisone 50mg IV q6h initiated\n"
            f"   - Repeat lactate q4h — poor clearance\n"
            f"   - Discussed goals of care with family"
        )
    else:  # day 5
        if outcome == "improving":
            problems.append(
                f"1. Sepsis — Improving. SOFA {sofa}{sofa_trend}. Source: {source}.\n"
                f"   - Norepinephrine weaning, currently at 0.02 mcg/kg/min\n"
                f"   - Lactate trending down — clearance improving\n"
                f"   - De-escalate antibiotics per culture sensitivities\n"
                f"   - Continue hydrocortisone taper\n"
                f"   - Consider step-down to floor if hemodynamically stable off pressors x24h"
            )
        else:
            problems.append(
                f"1. Refractory Septic Shock — {source} source. SOFA {sofa}{sofa_trend}. "
                f"Worsening despite maximal therapy.\n"
                f"   - On norepinephrine 0.3 mcg/kg/min + vasopressin 0.04 units/min\n"
                f"   - Consider adding epinephrine if refractory\n"
                f"   - Lactate >6, poor clearance\n"
                f"   - Repeat source imaging — CT abdomen/pelvis to evaluate for abscess or "
                f"undrained collection\n"
                f"   - Goals of care discussion with family — transition to comfort care if "
                f"no improvement in 24-48 hours"
            )

    # Problem 2: AKI
    cr = labs["creatinine"]
    if cr >= 1.5:
        stage = 1 if cr < 2.0 else (2 if cr < 3.0 else 3)
        problems.append(
            f"2. Acute Kidney Injury — Creatinine {cr}, KDIGO Stage {stage}.\n"
            f"   - Strict I/O monitoring\n"
            f"   - Avoid nephrotoxic agents — hold NSAIDs, aminoglycosides\n"
            f"   - Fluid resuscitation to maintain renal perfusion\n"
            + ("   - Nephrology consult placed\n   - CRRT evaluation if oliguric and refractory to diuretics"
               if stage >= 2 else "   - Nephrology consult if continues to worsen")
        )

    # Problem 3: Thrombocytopenia / DIC risk
    plt = labs["platelets"]
    if plt < 150:
        dic_comment = ""
        if plt < 80:
            dic_comment = " High concern for DIC."
        problems.append(
            f"{len(problems)+1}. Thrombocytopenia — Platelets {plt} K/uL.{dic_comment}\n"
            f"   - Check fibrinogen, D-dimer, PT/INR, peripheral smear\n"
            f"   - Hold heparin products — evaluate for HIT if applicable\n"
            + ("   - Transfuse platelets if <10K or active bleeding\n"
               "   - Hematology consult" if plt < 50 else
               "   - Monitor trend q12h")
        )

    # Problem 4: Respiratory failure
    if vitals["spo2"] < 94 or vitals["rr"] > 24:
        o2 = _o2_device(vitals["spo2"])
        problems.append(
            f"{len(problems)+1}. Acute Hypoxemic Respiratory Failure\n"
            f"   - Currently on {o2}\n"
            f"   - ABG to assess ventilation and oxygenation\n"
            + ("   - Intubation kit at bedside, RSI medications prepared\n"
               "   - Discuss intubation threshold: SpO2 <88% sustained or severe WOB"
               if vitals["spo2"] < 90 else
               "   - Titrate supplemental O2 to target SpO2 ≥92%\n"
               "   - Chest physiotherapy, incentive spirometry if able to participate")
        )

    # Problem 5: Nutrition
    if day >= 2:
        problems.append(
            f"{len(problems)+1}. Nutrition\n"
            + ("   - Enteral nutrition via OG tube at goal rate 55 mL/hr\n"
               "   - Monitor gastric residuals q4h\n"
               "   - Dietitian following" if day <= 3 else
               "   - Tube feeds held due to hemodynamic instability / ileus\n"
               "   - Start TPN if enteral nutrition not feasible within 48h\n"
               "   - Glycemic control with insulin drip, target BG 140-180")
        )

    # Problem 6: DVT prophylaxis
    problems.append(
        f"{len(problems)+1}. DVT Prophylaxis\n"
        + ("   - Enoxaparin 40mg SC daily"
           if plt >= 50 else
           "   - SCDs bilateral lower extremities (chemical prophylaxis held for thrombocytopenia)")
    )

    header = (
        f"SOFA Score: {sofa}{sofa_trend}\n"
        f"{qsofa_str}\n\n"
        f"A/P:\n"
    )
    return header + "\n\n".join(problems)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_progress_note(patient_id: str, day_num: int, vitals_summary: Optional[Dict] = None,
                           labs_summary: Optional[Dict] = None, diagnosis: str = "sepsis",
                           trend_info: Optional[Dict] = None) -> str:
    """
    Generate a realistic ICU attending progress note for a given day.

    Parameters
    ----------
    patient_id : str
    day_num : int
        ICU day (1-5).
    vitals_summary : dict, optional
        Pre-computed vitals; if None, sampled from day-specific ranges.
    labs_summary : dict, optional
        Pre-computed labs; if None, sampled from day-specific ranges.
    diagnosis : str
        Primary diagnosis key (default "sepsis").
    trend_info : dict, optional
        Dictionary with keys 'prev_vitals', 'prev_labs', 'prev_sofa' for trending.

    Returns
    -------
    str
        Formatted progress note text.
    """
    clamped_day = max(1, min(day_num, 5))
    outcome = "worsening" if random.random() < 0.6 else "improving"
    if clamped_day < 5:
        outcome = "worsening"  # only day 5 branches

    vitals = vitals_summary or _sample_vitals(clamped_day, outcome)
    labs = labs_summary or _sample_labs(clamped_day, outcome)
    sofa = _sofa_score(clamped_day, outcome)
    prev_sofa = _sofa_score(clamped_day - 1) if clamped_day > 1 else None
    prev_labs = (trend_info or {}).get("prev_labs")
    diagnosis_info = DIAGNOSES_TEMPLATES.get(diagnosis, DIAGNOSES_TEMPLATES["sepsis"])
    attending = _pick(ATTENDING_PHYSICIANS)

    note = (
        f"PROGRESS NOTE — ICU Day {day_num}\n"
        f"Patient: {patient_id}\n"
        f"Attending: {attending}\n"
        f"{'=' * 60}\n\n"
        f"S:\n{_subjective(clamped_day, diagnosis)}\n\n"
        f"O:\n{_physical_exam(clamped_day, vitals)}\n\n"
        f"Labs (06:00):\n{_format_labs(labs, prev_labs)}\n\n"
        f"{_assessment_plan(clamped_day, vitals, labs, prev_labs, diagnosis_info, sofa, prev_sofa, outcome)}\n\n"
        f"Discussed with {attending}. Family updated on clinical status and plan.\n"
        f"Code Status: Full code.\n"
        f"{'=' * 60}"
    )
    return note


def generate_nursing_note(patient_id: str, shift: str, vitals_snapshot: Optional[Dict] = None,
                          concerns: Optional[List[str]] = None) -> str:
    """
    Generate a realistic ICU nursing assessment note.

    Parameters
    ----------
    patient_id : str
    shift : str
        'day' (07:00-19:00) or 'night' (19:00-07:00).
    vitals_snapshot : dict, optional
        Vitals at time of assessment. Sampled if not provided.
    concerns : list of str, optional
        Specific nursing concerns to highlight.

    Returns
    -------
    str
        Formatted nursing note text.
    """
    nurse = _pick(NURSES)
    vitals = vitals_snapshot or _sample_vitals(random.randint(1, 4))
    o2 = _o2_device(vitals["spo2"])
    concerns = concerns or []

    shift_label = "Day Shift (07:00-19:00)" if shift == "day" else "Night Shift (19:00-07:00)"

    pain_score = random.randint(0, 6)
    rass = random.choice([-2, -1, 0, 0, 0, 1])
    cam_icu = "Positive" if rass < 0 and random.random() > 0.5 else "Negative"
    braden_score = random.randint(10, 20)
    uop = random.randint(10, 80)
    intake = random.randint(800, 3500)

    # Nursing interventions based on vitals
    interventions = []
    if vitals["sbp"] < 90:
        interventions.append("Notified physician of hypotension. Fluid bolus 500 mL NS administered per order.")
    if vitals["hr"] > 120:
        interventions.append("Continuous telemetry monitoring. Physician aware of persistent tachycardia.")
    if vitals["temp"] > 38.5:
        interventions.append("Fever noted. Blood cultures drawn x2 from peripheral and CVC per protocol. "
                             "Acetaminophen 1g IV administered. Cooling measures initiated.")
    if vitals["spo2"] < 92:
        interventions.append(f"Supplemental O2 titrated up to {o2}. RT notified for ABG draw. "
                             "HOB elevated to 30 degrees. IS encouraged q1h when awake.")
    if uop < 30:
        interventions.append("Low urine output noted. Physician notified. Fluid challenge ordered.")
    if not interventions:
        interventions.append("Routine assessments and medications administered per orders. No acute interventions required.")

    concern_section = ""
    if concerns:
        concern_section = "\nCONCERNS COMMUNICATED TO TEAM:\n" + "\n".join(f"  - {c}" for c in concerns)

    note = (
        f"NURSING ASSESSMENT — {shift_label}\n"
        f"Patient: {patient_id}\n"
        f"RN: {nurse}\n"
        f"{'-' * 50}\n\n"
        f"VITAL SIGNS:\n"
        f"  Temp: {vitals['temp']}°C | HR: {int(vitals['hr'])} | "
        f"BP: {int(vitals['sbp'])}/{int(vitals['dbp'])} | RR: {int(vitals['rr'])} | "
        f"SpO2: {int(vitals['spo2'])}% on {o2}\n\n"
        f"NEUROLOGICAL:\n"
        f"  RASS: {rass:+d} | CAM-ICU: {cam_icu}\n"
        f"  Pain: {pain_score}/10 ({'none' if pain_score == 0 else 'mild' if pain_score <= 3 else 'moderate' if pain_score <= 6 else 'severe'})\n"
        f"  {'Oriented x4' if rass >= 0 else 'Drowsy, arousable to voice'}\n\n"
        f"CARDIOVASCULAR:\n"
        f"  Heart rhythm: {'NSR' if vitals['hr'] < 100 else 'Sinus tachycardia'}\n"
        f"  Peripheral pulses: {'2+ bilaterally' if vitals['sbp'] > 90 else 'Weak, thready'}\n"
        f"  Extremities: {'Warm, well-perfused' if vitals['sbp'] > 95 else 'Cool, mottled'}\n\n"
        f"RESPIRATORY:\n"
        f"  O2 delivery: {o2}\n"
        f"  Breath sounds: {'Clear bilaterally' if vitals['spo2'] > 95 else 'Bilateral crackles'}\n"
        f"  Work of breathing: {'Normal' if vitals['rr'] < 22 else 'Increased — accessory muscle use noted'}\n"
        f"  Cough: {'Nonproductive' if vitals['spo2'] > 93 else 'Productive, thick yellow sputum'}\n\n"
        f"GI/GU:\n"
        f"  Abdomen: {'Soft, non-tender' if vitals['hr'] < 110 else 'Mildly distended, tender'}\n"
        f"  Diet: {'Tube feeds at goal rate' if vitals['sbp'] > 85 else 'NPO — tube feeds held'}\n"
        f"  Urine output: {uop} mL/hr ({'adequate' if uop >= 30 else 'LOW — below 0.5 mL/kg/hr'})\n"
        f"  Foley catheter: Patent, draining {'clear yellow' if uop >= 30 else 'concentrated dark'} urine\n\n"
        f"SKIN/WOUNDS:\n"
        f"  Braden Score: {braden_score} ({'adequate' if braden_score >= 15 else 'AT RISK — reposition q2h'})\n"
        f"  Central line dressing: Intact, no erythema\n"
        f"  {'No skin breakdown noted' if braden_score >= 15 else 'Stage I pressure injury sacrum — barrier cream applied'}\n\n"
        f"I/O (past 12 hours):\n"
        f"  Intake: {intake} mL (IVF + meds + tube feeds)\n"
        f"  Output: {uop * 12} mL (urine) + {random.randint(0, 200)} mL (other drains)\n"
        f"  Net: {'+' if intake > uop * 12 else ''}{intake - uop * 12} mL\n\n"
        f"INTERVENTIONS:\n" + "\n".join(f"  - {i}" for i in interventions) +
        f"\n{concern_section}\n\n"
        f"SAFETY:\n"
        f"  Bed alarm: ON | Side rails: Up x2 | Call bell: Within reach\n"
        f"  Fall risk: {'Standard precautions' if braden_score >= 15 else 'High — bed alarm active, 1:1 sitter evaluated'}\n"
        f"{'-' * 50}"
    )
    return note


def generate_discharge_summary(patient_id: str, admission_data: Optional[Dict] = None,
                               course_summary: Optional[Dict] = None) -> str:
    """
    Generate a realistic ICU discharge summary.

    Parameters
    ----------
    patient_id : str
    admission_data : dict, optional
        Keys: admit_date, admit_diagnosis, age, sex, pmh, source.
    course_summary : dict, optional
        Keys: los_days, outcome, complications, discharge_disposition.

    Returns
    -------
    str
        Formatted discharge summary text.
    """
    admission = admission_data or {}
    course = course_summary or {}

    age = admission.get("age", random.randint(45, 82))
    sex = admission.get("sex", random.choice(["Male", "Female"]))
    admit_date = admission.get("admit_date", datetime.now() - timedelta(days=7))
    if isinstance(admit_date, str):
        admit_date = datetime.fromisoformat(admit_date)
    los = course.get("los_days", random.randint(5, 14))
    discharge_date = admit_date + timedelta(days=los)
    source = _pick(DIAGNOSES_TEMPLATES["sepsis"]["sources"])
    organism = _pick(DIAGNOSES_TEMPLATES["sepsis"]["organisms"])
    outcome = course.get("outcome", random.choice(["improved", "deceased"]))
    attending = _pick(ATTENDING_PHYSICIANS)
    disposition = course.get("discharge_disposition",
                             "home with home health" if outcome == "improved" else "expired")

    pmh = admission.get("pmh", [
        "Hypertension", "Type 2 Diabetes Mellitus", "Chronic Kidney Disease Stage III",
        "Atrial Fibrillation", "COPD (GOLD Stage II)"
    ])
    pmh_str = "\n".join(f"  - {p}" for p in pmh)

    complications = course.get("complications", [])
    if not complications:
        complications = ["Acute Kidney Injury (KDIGO Stage 2)", "Thrombocytopenia",
                         "ICU-acquired weakness", "Hospital-acquired pneumonia"]
        if outcome == "deceased":
            complications.append("Refractory septic shock")
            complications.append("Multi-organ failure")

    complication_str = "\n".join(f"  - {c}" for c in complications)

    if outcome == "improved":
        hospital_course = (
            f"The patient was admitted to the MICU with {source}-related sepsis. Initial presentation "
            f"included fever, tachycardia, and hypotension requiring aggressive fluid resuscitation. "
            f"Empiric broad-spectrum antibiotics were initiated within 1 hour of presentation. Blood "
            f"cultures returned positive for {organism} on hospital day 2.\n\n"
            f"Hospital Day 1-2: Initial stabilization with IV fluids and antibiotics. Lactate cleared "
            f"appropriately after initial resuscitation.\n\n"
            f"Hospital Day 3-4: Clinical deterioration with worsening hypotension requiring "
            f"norepinephrine infusion. SOFA score peaked at 10. Developed AKI with creatinine rising "
            f"to 3.2. Nephrology consulted. Stress-dose steroids initiated. ID recommended antibiotic "
            f"de-escalation to targeted therapy based on culture sensitivities.\n\n"
            f"Hospital Day 5-7: Gradual clinical improvement. Vasopressors weaned and discontinued. "
            f"Lactate normalized. Creatinine trending down. Renal function recovering. Patient "
            f"extubated on day 6 (had required intubation on day 4 for hypoxemic respiratory failure). "
            f"Tolerating oral diet. Physical therapy initiated.\n\n"
            f"Hospital Day 8-{los}: Continued recovery. Transferred to step-down unit on day 8. "
            f"Completed 10-day antibiotic course. Discharged to {disposition} with close follow-up."
        )
        discharge_meds = (
            "  1. Amoxicillin-Clavulanate 875mg PO BID x 4 days (to complete antibiotic course)\n"
            "  2. Metoprolol Tartrate 25mg PO BID\n"
            "  3. Lisinopril 10mg PO daily (restarted at lower dose given AKI)\n"
            "  4. Metformin 500mg PO BID (held until renal function fully recovered)\n"
            "  5. Pantoprazole 40mg PO daily\n"
            "  6. Enoxaparin 40mg SC daily x 2 weeks (DVT prophylaxis post-discharge)"
        )
        followup = (
            "  - PCP: Dr. Sharma — within 1 week\n"
            "  - Nephrology: Dr. Iyer — within 2 weeks (renal function monitoring)\n"
            "  - Infectious Disease: Dr. Patel — within 2 weeks\n"
            "  - Repeat BMP in 3 days at outpatient lab"
        )
    else:
        hospital_course = (
            f"The patient was admitted to the MICU with {source}-related sepsis. Despite aggressive "
            f"resuscitation, the patient developed refractory septic shock with progressive "
            f"multi-organ dysfunction.\n\n"
            f"Hospital Day 1-2: Hemodynamic instability requiring vasopressor support within 6 hours "
            f"of admission. Blood cultures positive for {organism}.\n\n"
            f"Hospital Day 3-4: Escalating vasopressor requirements (norepinephrine + vasopressin + "
            f"epinephrine). SOFA score rose to 14. Developed anuric AKI — CRRT initiated. "
            f"Intubated for hypoxemic respiratory failure. DIC with platelet count nadir of 22K.\n\n"
            f"Hospital Day 5-{los}: Despite maximal medical therapy including broad-spectrum "
            f"antibiotics, vasopressors, stress-dose steroids, and CRRT, the patient continued to "
            f"deteriorate. Lactate >8 mmol/L persistently. After extensive discussion with family "
            f"regarding prognosis and goals of care, decision was made to transition to comfort-focused "
            f"care. Vasopressors discontinued. Patient expired on hospital day {los} at "
            f"{random.randint(1,12):02d}:{random.randint(0,59):02d} with family at bedside.\n\n"
            f"Cause of death: Refractory septic shock with multi-organ failure secondary to "
            f"{source} ({organism})."
        )
        discharge_meds = "  N/A — Patient deceased."
        followup = "  N/A — Patient deceased."

    note = (
        f"DISCHARGE SUMMARY\n"
        f"{'=' * 60}\n"
        f"Patient: {patient_id}\n"
        f"Age/Sex: {age}/{sex[0]}\n"
        f"Attending: {attending}\n"
        f"Admission Date: {admit_date.strftime('%Y-%m-%d')}\n"
        f"Discharge Date: {discharge_date.strftime('%Y-%m-%d')}\n"
        f"Length of Stay: {los} days\n"
        f"Discharge Disposition: {disposition.title()}\n"
        f"{'=' * 60}\n\n"
        f"PRINCIPAL DIAGNOSIS:\n"
        f"  Sepsis secondary to {source} ({organism})\n\n"
        f"SECONDARY DIAGNOSES:\n{complication_str}\n\n"
        f"PAST MEDICAL HISTORY:\n{pmh_str}\n\n"
        f"HOSPITAL COURSE:\n{hospital_course}\n\n"
        f"DISCHARGE MEDICATIONS:\n{discharge_meds}\n\n"
        f"FOLLOW-UP:\n{followup}\n\n"
        f"DISCHARGE INSTRUCTIONS:\n"
        f"  - Monitor temperature daily; return to ED if fever >38.3°C\n"
        f"  - Maintain adequate hydration (goal >2L fluids daily)\n"
        f"  - Complete full antibiotic course as prescribed\n"
        f"  - Activity: Light activity as tolerated; avoid strenuous exertion x 4 weeks\n"
        f"  - Return to ED for: fever, confusion, shortness of breath, chest pain, decreased urine output\n\n"
        f"DICTATED BY: {attending}\n"
        f"{'=' * 60}"
    )
    return note


def generate_clinical_notes(patient_id: str, vitals_df: Optional[pd.DataFrame] = None,
                            labs_df: Optional[pd.DataFrame] = None,
                            diagnosis: str = "sepsis",
                            num_days: int = 5) -> List[Dict]:
    """
    Generate a full set of clinical notes across an ICU stay.

    Produces attending progress notes, resident night notes, and nursing
    assessments for each day/shift, simulating a worsening sepsis trajectory.

    Parameters
    ----------
    patient_id : str
    vitals_df : pd.DataFrame, optional
        Patient vitals DataFrame (columns: timestamp, hr, sbp, dbp, rr, temp, spo2).
        If None, synthetic vitals are generated per day.
    labs_df : pd.DataFrame, optional
        Patient labs DataFrame. If None, synthetic labs are generated per day.
    diagnosis : str
        Primary diagnosis key (default "sepsis").
    num_days : int
        Number of ICU days to generate notes for (default 5, max 5).

    Returns
    -------
    list of dict
        Each dict: {timestamp, author, note_type, text, shift}
    """
    num_days = max(1, min(num_days, 5))
    admit_time = datetime.now().replace(hour=14, minute=30, second=0, microsecond=0) - timedelta(days=num_days)
    notes = []

    prev_labs = None
    prev_sofa = None
    outcome = random.choice(["improving", "worsening"])

    for day in range(1, num_days + 1):
        day_date = admit_time + timedelta(days=day - 1)
        clamped = max(1, min(day, 5))
        day_outcome = outcome if clamped == 5 else "worsening"

        # --- Sample or extract vitals/labs for this day ---
        if vitals_df is not None and not vitals_df.empty:
            day_vitals = _extract_day_vitals(vitals_df, day_date)
        else:
            day_vitals = _sample_vitals(clamped, day_outcome)

        if labs_df is not None and not labs_df.empty:
            day_labs = _extract_day_labs(labs_df, day_date)
        else:
            day_labs = _sample_labs(clamped, day_outcome)

        sofa = _sofa_score(clamped, day_outcome)

        # ---- Nursing note: night shift (early AM) ----
        night_ts = day_date.replace(hour=random.randint(2, 5), minute=random.randint(0, 59))
        night_nurse = _pick(NURSES)
        night_concerns = _derive_nursing_concerns(day_vitals, day_labs, clamped)
        night_vitals = _perturb_vitals(day_vitals, noise=0.05)
        notes.append({
            "timestamp": night_ts.isoformat(),
            "author": night_nurse,
            "note_type": "Nursing Assessment",
            "text": generate_nursing_note(patient_id, "night", night_vitals, night_concerns),
            "shift": "night",
        })

        # ---- Attending progress note (morning rounds) ----
        am_ts = day_date.replace(hour=random.randint(8, 10), minute=random.randint(0, 45))
        attending = _pick(ATTENDING_PHYSICIANS)
        trend = {"prev_labs": prev_labs, "prev_sofa": prev_sofa}
        progress_text = generate_progress_note(
            patient_id, day, day_vitals, day_labs, diagnosis, trend
        )
        notes.append({
            "timestamp": am_ts.isoformat(),
            "author": attending,
            "note_type": "Progress Note",
            "text": progress_text,
            "shift": "day",
        })

        # ---- Nursing note: day shift ----
        day_nurse_ts = day_date.replace(hour=random.randint(13, 17), minute=random.randint(0, 59))
        day_nurse = _pick(NURSES)
        pm_vitals = _perturb_vitals(day_vitals, noise=0.08)
        pm_concerns = _derive_nursing_concerns(pm_vitals, day_labs, clamped)
        notes.append({
            "timestamp": day_nurse_ts.isoformat(),
            "author": day_nurse,
            "note_type": "Nursing Assessment",
            "text": generate_nursing_note(patient_id, "day", pm_vitals, pm_concerns),
            "shift": "day",
        })

        # ---- Night resident note (evening) ----
        pm_ts = day_date.replace(hour=random.randint(20, 23), minute=random.randint(0, 59))
        resident = _pick(RESIDENTS)
        night_resident_text = _generate_resident_night_note(
            patient_id, day, pm_vitals, day_labs, attending, resident, clamped
        )
        notes.append({
            "timestamp": pm_ts.isoformat(),
            "author": resident,
            "note_type": "Night Resident Note",
            "text": night_resident_text,
            "shift": "night",
        })

        prev_labs = day_labs
        prev_sofa = sofa

    # Sort by timestamp
    notes.sort(key=lambda n: n["timestamp"])
    return notes


# ---------------------------------------------------------------------------
# Internal helpers for generate_clinical_notes
# ---------------------------------------------------------------------------

def _extract_day_vitals(df: pd.DataFrame, day_date: datetime) -> Dict[str, float]:
    """Pull average vitals from a DataFrame for a given day, falling back to sampling."""
    try:
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col])
        mask = df[ts_col].dt.date == day_date.date()
        subset = df[mask]
        if subset.empty:
            return _sample_vitals(1)
        mapping = {"hr": "hr", "sbp": "sbp", "dbp": "dbp", "rr": "rr",
                    "temp": "temp", "spo2": "spo2"}
        result = {}
        for key, col in mapping.items():
            if col in subset.columns:
                result[key] = round(float(subset[col].mean()), 1)
            else:
                result[key] = _sample_vitals(1)[key]
        return result
    except Exception:
        return _sample_vitals(1)


def _extract_day_labs(df: pd.DataFrame, day_date: datetime) -> Dict[str, float]:
    """Pull lab values from a DataFrame for a given day, falling back to sampling."""
    try:
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col])
        mask = df[ts_col].dt.date == day_date.date()
        subset = df[mask]
        if subset.empty:
            return _sample_labs(1)
        lab_keys = ["wbc", "lactate", "creatinine", "platelets", "bilirubin",
                     "hemoglobin", "sodium", "potassium", "bicarbonate", "bun",
                     "inr", "fibrinogen", "procalcitonin", "crp"]
        result = {}
        for key in lab_keys:
            if key in subset.columns:
                result[key] = round(float(subset[key].mean()), 1)
            else:
                result[key] = _sample_labs(1)[key]
        return result
    except Exception:
        return _sample_labs(1)


def _perturb_vitals(vitals: Dict[str, float], noise: float = 0.05) -> Dict[str, float]:
    """Add small random perturbation to vitals for realism across shifts."""
    result = {}
    for k, v in vitals.items():
        delta = v * random.uniform(-noise, noise)
        result[k] = round(v + delta, 1)
    return result


def _derive_nursing_concerns(vitals: Dict[str, float], labs: Dict[str, float],
                             day: int) -> List[str]:
    """Generate a list of nursing concerns based on vitals, labs, and day."""
    concerns = []
    if vitals.get("sbp", 120) < 90:
        concerns.append("Persistent hypotension despite fluid boluses — physician notified")
    if vitals.get("hr", 80) > 120:
        concerns.append("Tachycardia >120 sustained — r/o dehydration, pain, sepsis progression")
    if vitals.get("temp", 37) > 38.5:
        concerns.append("Fever — new blood cultures sent, antipyretics administered")
    if vitals.get("spo2", 98) < 92:
        concerns.append("Desaturation event — O2 supplementation increased, ABG sent")
    if labs.get("lactate", 1.0) > 4.0:
        concerns.append("Elevated lactate — physician aware, repeat ordered in 4 hours")
    if labs.get("platelets", 200) < 80:
        concerns.append("Significant thrombocytopenia — bleeding precautions in place")
    if day >= 4:
        concerns.append("Decreasing urine output — strict I/O monitoring, physician notified")
    return concerns


def _generate_resident_night_note(patient_id: str, day: int, vitals: Dict[str, float],
                                  labs: Dict[str, float], attending: str,
                                  resident: str, clamped_day: int) -> str:
    """Generate an overnight resident check-in note."""
    o2 = _o2_device(vitals.get("spo2", 95))

    events = []
    if clamped_day >= 3:
        events.append(f"Patient spiked temp to {_rand(38.2, 39.5)}°C at 21:30. Blood cultures "
                      f"drawn x2 (peripheral + CVC). Acetaminophen 1g IV given with effect.")
    if clamped_day >= 4:
        events.append(f"Hypotensive episode at 22:15 — BP {int(vitals['sbp']-8)}/{int(vitals['dbp']-5)}. "
                      f"500 mL NS bolus administered. Norepinephrine uptitrated by 0.02 mcg/kg/min.")
    if vitals.get("spo2", 98) < 92:
        events.append(f"Desaturation to {int(vitals['spo2']-2)}% at 23:00. O2 increased to {o2}. "
                      f"ABG obtained — results pending. RT at bedside.")
    if not events:
        events.append("Uneventful overnight. Patient resting comfortably with stable vitals.")

    overnight_plan = []
    if clamped_day <= 2:
        overnight_plan = [
            "Continue current antibiotics and IVF",
            "Repeat lactate at 04:00",
            "AM labs ordered: CBC, BMP, lactate, LFTs",
            "Call parameters: SBP <90, HR >130, SpO2 <90, UOP <20 mL/hr, temp >39°C",
        ]
    elif clamped_day == 3:
        overnight_plan = [
            "Continue current antibiotics and vasopressor support",
            "Repeat lactate q4h",
            "Strict I/O — target UOP ≥0.5 mL/kg/hr",
            "AM labs: CBC, BMP, lactate, LFTs, coags, fibrinogen",
            "Low threshold for intubation if respiratory status worsens",
            f"Attending {attending} aware and available for escalation",
        ]
    else:
        overnight_plan = [
            "Continue maximal ICU support",
            "Vasopressor titration per MAP target ≥65",
            "CRRT to continue per nephrology recs" if labs.get("creatinine", 1) > 3 else "Monitor renal function closely",
            "Repeat lactate q4h, ABG q6h",
            "Platelet transfusion if <10K or active bleeding",
            f"Goals of care discussion scheduled for AM with {attending} and family",
            f"Attending {attending} notified of overnight events",
        ]

    note = (
        f"NIGHT RESIDENT NOTE — ICU Day {day}\n"
        f"Patient: {patient_id}\n"
        f"Resident: {resident} | Supervising Attending: {attending}\n"
        f"{'-' * 50}\n\n"
        f"OVERNIGHT VITALS:\n"
        f"  T {vitals['temp']}°C | HR {int(vitals['hr'])} | "
        f"BP {int(vitals['sbp'])}/{int(vitals['dbp'])} | RR {int(vitals['rr'])} | "
        f"SpO2 {int(vitals['spo2'])}% on {o2}\n\n"
        f"OVERNIGHT EVENTS:\n" +
        "\n".join(f"  - {e}" for e in events) +
        f"\n\nOVERNIGHT PLAN:\n" +
        "\n".join(f"  {i+1}. {p}" for i, p in enumerate(overnight_plan)) +
        f"\n\nSIGN-OUT TO DAY TEAM:\n"
        f"  Primary concern: {'Hemodynamic instability' if clamped_day >= 3 else 'Early sepsis — monitor for deterioration'}\n"
        f"  Anticipatory guidance: {'Possible intubation if resp status worsens' if clamped_day >= 3 else 'Watch for SOFA score increase'}\n"
        f"  Code status: Full code\n"
        f"{'-' * 50}"
    )
    return note


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    pid = "PT-2024-001"

    print("=" * 70)
    print("GENERATING 5-DAY ICU NOTE SET")
    print("=" * 70)

    all_notes = generate_clinical_notes(pid, diagnosis="sepsis", num_days=5)
    for n in all_notes:
        print(f"\n[{n['timestamp']}] {n['note_type']} by {n['author']} ({n['shift']} shift)")
        print(n["text"])
        print()

    print("\n" + "=" * 70)
    print("DISCHARGE SUMMARY")
    print("=" * 70)
    ds = generate_discharge_summary(pid, admission_data={"age": 67, "sex": "Male"})
    print(ds)
