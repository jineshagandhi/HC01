"""
Note Parser Agent – extracts structured medical information from clinical notes.

Primary extraction uses Google Gemini LLM for high-quality entity recognition.
Falls back to regex-based extraction if the API call fails.
"""

import json
import re
import logging
from datetime import datetime
from typing import Optional

import google.generativeai as genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(GEMINI_MODEL)

# ---------------------------------------------------------------------------
# Regex-based fallback patterns
# ---------------------------------------------------------------------------

_SYMPTOM_PATTERNS = [
    r"\b(fever|chills|rigors|diaphoresis|tachycardia|tachypnea|hypotension|"
    r"dyspnea|shortness of breath|chest pain|abdominal pain|nausea|vomiting|"
    r"diarrhea|altered mental status|confusion|lethargy|oliguria|anuria|"
    r"cough|wheezing|edema|rash|headache|dizziness|fatigue|weakness|"
    r"pain|bleeding|seizure|syncope|tremor|agitation|delirium)\b",
]

_MEDICATION_PATTERNS = [
    r"\b(norepinephrine|vasopressin|epinephrine|dopamine|dobutamine|"
    r"phenylephrine|milrinone|vancomycin|meropenem|piperacillin|tazobactam|"
    r"ceftriaxone|cefepime|metronidazole|azithromycin|levofloxacin|"
    r"ciprofloxacin|amoxicillin|ampicillin|gentamicin|tobramycin|"
    r"fluconazole|micafungin|heparin|enoxaparin|warfarin|aspirin|"
    r"morphine|fentanyl|hydromorphone|propofol|midazolam|lorazepam|"
    r"dexmedetomidine|ketamine|insulin|metformin|furosemide|"
    r"pantoprazole|omeprazole|acetaminophen|ibuprofen|"
    r"normal saline|lactated ringer|albumin|crystalloid)\b",
]

_CONDITION_PATTERNS = [
    r"\b(sepsis|septic shock|pneumonia|ards|acute respiratory distress|"
    r"respiratory failure|renal failure|aki|acute kidney injury|"
    r"heart failure|chf|myocardial infarction|mi|stroke|cva|"
    r"diabetic ketoacidosis|dka|copd exacerbation|pulmonary embolism|"
    r"dvt|deep vein thrombosis|gi bleed|gastrointestinal bleed|"
    r"pancreatitis|cirrhosis|hepatic failure|meningitis|encephalitis|"
    r"bacteremia|urinary tract infection|uti|cellulitis|endocarditis|"
    r"atrial fibrillation|afib|hypertension|diabetes|covid|"
    r"multiorgan failure|dic|disseminated intravascular coagulation)\b",
]

_PROCEDURE_PATTERNS = [
    r"\b(intubation|extubation|mechanical ventilation|central line|"
    r"arterial line|chest tube|thoracentesis|paracentesis|lumbar puncture|"
    r"dialysis|crrt|ecmo|bronchoscopy|endoscopy|colonoscopy|"
    r"ct scan|mri|x-ray|ultrasound|echocardiogram|ekg|"
    r"blood culture|urine culture|sputum culture|biopsy|"
    r"tracheostomy|cardioversion|defibrillation|cpr|"
    r"transfusion|surgery|debridement|drain placement)\b",
]

_VITAL_PATTERNS = [
    (r"\bhr\s*[:=]?\s*(\d+)", "heart_rate"),
    (r"\bheart rate\s*[:=]?\s*(\d+)", "heart_rate"),
    (r"\bbp\s*[:=]?\s*(\d+/\d+)", "blood_pressure"),
    (r"\bsbp\s*[:=]?\s*(\d+)", "sbp"),
    (r"\bdbp\s*[:=]?\s*(\d+)", "dbp"),
    (r"\bmap\s*[:=]?\s*(\d+)", "map"),
    (r"\brr\s*[:=]?\s*(\d+)", "respiratory_rate"),
    (r"\bresp(?:iratory)? rate\s*[:=]?\s*(\d+)", "respiratory_rate"),
    (r"\bspo2\s*[:=]?\s*(\d+)", "spo2"),
    (r"\btemp(?:erature)?\s*[:=]?\s*([\d.]+)", "temperature"),
    (r"\bgcs\s*[:=]?\s*(\d+)", "gcs"),
    (r"\bfio2\s*[:=]?\s*([\d.]+)", "fio2"),
]

_URGENCY_KEYWORDS = {
    "critical": [
        "code blue", "cardiac arrest", "respiratory arrest", "emergent",
        "stat", "critical", "life-threatening", "deteriorating rapidly",
        "crashing", "unresponsive", "pulseless", "apneic",
    ],
    "concerning": [
        "worsening", "declining", "unstable", "concerning", "escalating",
        "new onset", "acute change", "decompensating", "requiring vasopressors",
        "intubated", "icu transfer",
    ],
}


# ---------------------------------------------------------------------------
# Regex fallback extraction
# ---------------------------------------------------------------------------

def _extract_with_regex(note_text: str) -> dict:
    """Extract structured data from a clinical note using regex patterns."""
    text_lower = note_text.lower()

    def _find_all(patterns: list[str]) -> list[str]:
        found: set[str] = set()
        for pat in patterns:
            matches = re.findall(pat, text_lower, re.IGNORECASE)
            for m in matches:
                found.add(m.strip().title())
        return sorted(found)

    symptoms = _find_all(_SYMPTOM_PATTERNS)
    medications = _find_all(_MEDICATION_PATTERNS)
    conditions = _find_all(_CONDITION_PATTERNS)
    procedures = _find_all(_PROCEDURE_PATTERNS)

    # Vitals
    vitals_mentioned: list[dict] = []
    for pat, vital_name in _VITAL_PATTERNS:
        matches = re.findall(pat, text_lower, re.IGNORECASE)
        for m in matches:
            vitals_mentioned.append({"parameter": vital_name, "value": m})

    # Urgency
    urgency_level = "routine"
    for keyword in _URGENCY_KEYWORDS["critical"]:
        if keyword in text_lower:
            urgency_level = "critical"
            break
    if urgency_level == "routine":
        for keyword in _URGENCY_KEYWORDS["concerning"]:
            if keyword in text_lower:
                urgency_level = "concerning"
                break

    # Key findings = symptoms + conditions combined
    key_findings = list(set(symptoms + conditions))

    return {
        "symptoms": symptoms,
        "medications": medications,
        "conditions": conditions,
        "procedures": procedures,
        "vitals_mentioned": vitals_mentioned,
        "urgency_level": urgency_level,
        "key_findings": key_findings,
    }


# ---------------------------------------------------------------------------
# Gemini-based extraction (primary)
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a medical NLP system. Extract structured information from this ICU clinical note.

Note: {note_text}

Extract and return as JSON:
{{
  "symptoms": ["list of symptoms mentioned"],
  "medications": ["list of medications mentioned"],
  "conditions": ["list of conditions/diagnoses mentioned"],
  "procedures": ["list of procedures mentioned or planned"],
  "key_findings": ["list of important clinical findings"],
  "urgency_level": "routine|concerning|critical"
}}

Only return valid JSON. No explanation."""


def _extract_with_gemini(note_text: str) -> Optional[dict]:
    """Use Gemini to extract structured medical entities from a clinical note."""
    try:
        prompt = _EXTRACTION_PROMPT.format(note_text=note_text)
        response = _model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=LLM_TEMPERATURE),
        )
        result = response.text

        # Strip markdown code fences if present
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            result = "\n".join(lines)

        parsed = json.loads(result)

        # Validate expected keys
        expected_keys = {"symptoms", "medications", "conditions", "procedures",
                         "key_findings", "urgency_level"}
        for key in expected_keys:
            if key not in parsed:
                parsed[key] = [] if key != "urgency_level" else "routine"

        # Ensure urgency_level is valid
        if parsed["urgency_level"] not in ("routine", "concerning", "critical"):
            parsed["urgency_level"] = "routine"

        # Add empty vitals_mentioned (Gemini prompt doesn't extract these)
        if "vitals_mentioned" not in parsed:
            parsed["vitals_mentioned"] = []

        return parsed

    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse Gemini JSON response: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Gemini extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_clinical_note(note_text: str) -> dict:
    """Extract structured medical information from a single clinical note.

    Uses Gemini LLM as the primary extraction method.  Falls back to
    regex-based extraction if the API call fails.

    Parameters
    ----------
    note_text : str
        Raw clinical note text.

    Returns
    -------
    dict with keys: symptoms, medications, conditions, procedures,
    vitals_mentioned, urgency_level, key_findings.
    """
    if not note_text or not note_text.strip():
        return {
            "symptoms": [],
            "medications": [],
            "conditions": [],
            "procedures": [],
            "vitals_mentioned": [],
            "urgency_level": "routine",
            "key_findings": [],
        }

    # Primary: Gemini-based extraction
    gemini_result = _extract_with_gemini(note_text)
    if gemini_result is not None:
        # Augment with regex-detected vitals (Gemini prompt doesn't cover these)
        regex_result = _extract_with_regex(note_text)
        if not gemini_result["vitals_mentioned"] and regex_result["vitals_mentioned"]:
            gemini_result["vitals_mentioned"] = regex_result["vitals_mentioned"]
        return gemini_result

    # Fallback: regex-based extraction
    logger.info("Using regex fallback for note parsing")
    return _extract_with_regex(note_text)


def parse_all_notes(notes: list[dict]) -> dict:
    """Parse multiple clinical notes and aggregate findings with timestamps.

    Parameters
    ----------
    notes : list of dict
        Each dict should have at least ``text`` (str) and optionally
        ``timestamp`` (datetime or ISO string) and ``note_type`` (str).

    Returns
    -------
    dict with keys: all_symptoms, all_medications, all_conditions,
    urgency_timeline, critical_findings.
    """
    all_symptoms: list[dict] = []
    all_medications: list[dict] = []
    all_conditions: list[dict] = []
    urgency_timeline: list[tuple] = []
    critical_findings: list[str] = []

    for note in notes:
        text = note.get("text", "")
        timestamp = note.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        elif timestamp is None:
            timestamp = datetime.now()

        note_type = note.get("note_type", "unknown")

        parsed = parse_clinical_note(text)

        for symptom in parsed["symptoms"]:
            all_symptoms.append({
                "symptom": symptom,
                "timestamp": timestamp,
                "note_type": note_type,
            })

        for med in parsed["medications"]:
            all_medications.append({
                "medication": med,
                "timestamp": timestamp,
                "note_type": note_type,
            })

        for condition in parsed["conditions"]:
            all_conditions.append({
                "condition": condition,
                "timestamp": timestamp,
                "note_type": note_type,
            })

        urgency_timeline.append((timestamp, parsed["urgency_level"]))

        # Critical findings are key findings from notes flagged as critical
        if parsed["urgency_level"] == "critical":
            for finding in parsed["key_findings"]:
                critical_findings.append(
                    f"[{timestamp.strftime('%Y-%m-%d %H:%M')}] {finding}"
                )

    # Sort by timestamp
    all_symptoms.sort(key=lambda x: x["timestamp"])
    all_medications.sort(key=lambda x: x["timestamp"])
    all_conditions.sort(key=lambda x: x["timestamp"])
    urgency_timeline.sort(key=lambda x: x[0])

    return {
        "all_symptoms": all_symptoms,
        "all_medications": all_medications,
        "all_conditions": all_conditions,
        "urgency_timeline": urgency_timeline,
        "critical_findings": critical_findings,
    }
