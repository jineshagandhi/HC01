"""
Guideline RAG Agent – retrieves and cites relevant clinical guidelines.

Uses a ChromaDB vector store for semantic retrieval of guideline excerpts
and Gemini LLM to generate evidence-based clinical recommendations.
"""

import json
import logging
from typing import Optional

import google.generativeai as genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE, RAG_TOP_K
from backend.rag.vector_store import query_guidelines, get_guideline_context

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_relevant_guidelines(
    risk_flags: list[str],
    patient_context: str,
) -> list[dict]:
    """Retrieve clinical guideline excerpts relevant to the given risk flags.

    Parameters
    ----------
    risk_flags : list of str
        Clinical conditions or risk flags to search for (e.g. ["Sepsis", "AKI"]).
    patient_context : str
        Brief description of the patient's clinical context for query enrichment.

    Returns
    -------
    list of dict – each with guideline_name, section, text, relevance_score,
    source_document.
    """
    if not risk_flags:
        return []

    all_results: list[dict] = []
    seen_texts: set[str] = set()

    for flag in risk_flags:
        # Enrich query with patient context for better retrieval
        query = f"{flag} {patient_context}".strip()
        try:
            hits = query_guidelines(query, n_results=RAG_TOP_K)
        except Exception as exc:
            logger.warning("Vector store query failed for '%s': %s", flag, exc)
            continue

        for hit in hits:
            text = hit.get("text", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)

            source = hit.get("source", "unknown")
            guideline_name = source.replace(".pdf", "").replace("-", " ").replace("_", " ").title()
            page = hit.get("page", 0)

            all_results.append({
                "guideline_name": guideline_name,
                "section": f"Page {page}",
                "text": text,
                "relevance_score": hit.get("relevance_score", 0.0),
                "source_document": source,
                "risk_flag": flag,
            })

    # Sort by relevance score (best first)
    all_results.sort(key=lambda r: r["relevance_score"], reverse=True)

    return all_results


def format_citations(guidelines: list[dict]) -> str:
    """Format guideline excerpts as numbered citations.

    Parameters
    ----------
    guidelines : list of dict
        Output from ``get_relevant_guidelines``.

    Returns
    -------
    str – formatted citation block.
    """
    if not guidelines:
        return "No relevant guidelines found."

    lines: list[str] = []
    for i, g in enumerate(guidelines, start=1):
        name = g.get("guideline_name", "Unknown Guideline")
        section = g.get("section", "")
        text = g.get("text", "")
        # Truncate long excerpts for readability
        if len(text) > 300:
            text = text[:297] + "..."
        lines.append(f'[{i}] {name} ({section}): "{text}"')

    return "\n".join(lines)


def get_guideline_recommendations(
    condition: str,
    sofa_score: int,
    qsofa_score: int,
) -> dict:
    """Generate specific clinical recommendations backed by guideline evidence.

    Parameters
    ----------
    condition : str
        Primary clinical condition (e.g. "Sepsis", "ARDS").
    sofa_score : int
        Current SOFA score.
    qsofa_score : int
        Current qSOFA score.

    Returns
    -------
    dict with recommendations (list[str]), citations (list[dict]),
    evidence_level (str).
    """
    # Retrieve guideline context via RAG
    risk_flags = [condition]
    # Add score-specific queries
    if sofa_score >= 2:
        risk_flags.append(f"{condition} SOFA score management")
    if qsofa_score >= 2:
        risk_flags.append(f"{condition} qSOFA positive management")

    try:
        rag_context = get_guideline_context(risk_flags)
    except Exception as exc:
        logger.warning("Failed to retrieve guideline context: %s", exc)
        rag_context = "No guideline context available."

    # Get structured guideline hits for citations
    try:
        guideline_hits = get_relevant_guidelines(risk_flags, f"SOFA {sofa_score}, qSOFA {qsofa_score}")
    except Exception as exc:
        logger.warning("Failed to retrieve guidelines: %s", exc)
        guideline_hits = []

    # Use Gemini to generate recommendations
    prompt = f"""Based on the following clinical guidelines:
{rag_context}

Patient has: {condition}
SOFA Score: {sofa_score}
qSOFA Score: {qsofa_score}

Provide specific, actionable clinical recommendations with citations to the guidelines above.
Format as JSON: {{"recommendations": ["recommendation 1", "recommendation 2", ...], "evidence_summary": "brief summary of the evidence base"}}

Only return valid JSON. No explanation."""

    recommendations: list[str] = []
    evidence_summary = ""

    try:
        response = _model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=LLM_TEMPERATURE),
        )
        result = response.text.strip()

        # Strip markdown code fences if present
        if result.startswith("```"):
            lines = result.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result = "\n".join(lines)

        parsed = json.loads(result)
        recommendations = parsed.get("recommendations", [])
        evidence_summary = parsed.get("evidence_summary", "")

    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse Gemini recommendations JSON: %s", exc)
    except Exception as exc:
        logger.warning("Gemini recommendation generation failed: %s", exc)

    # Fallback recommendations if Gemini failed
    if not recommendations:
        recommendations = _fallback_recommendations(condition, sofa_score, qsofa_score)
        evidence_summary = "Recommendations generated from standard clinical protocols (LLM unavailable)."

    # Determine evidence level
    if guideline_hits:
        avg_relevance = sum(g["relevance_score"] for g in guideline_hits) / len(guideline_hits)
        if avg_relevance > 0.7:
            evidence_level = "Strong – well-supported by retrieved guidelines"
        elif avg_relevance > 0.4:
            evidence_level = "Moderate – partially supported by retrieved guidelines"
        else:
            evidence_level = "Limited – weak guideline match; use clinical judgment"
    else:
        evidence_level = "Insufficient – no guideline evidence retrieved"

    return {
        "recommendations": recommendations,
        "citations": guideline_hits[:5],  # Top 5 citations
        "evidence_level": evidence_level,
        "evidence_summary": evidence_summary,
    }


def _fallback_recommendations(condition: str, sofa_score: int, qsofa_score: int) -> list[str]:
    """Generate basic recommendations when Gemini is unavailable."""
    condition_lower = condition.lower()
    recs: list[str] = []

    if "sepsis" in condition_lower or "septic" in condition_lower:
        recs = [
            "Obtain blood cultures before initiating antimicrobials.",
            "Initiate broad-spectrum antimicrobial therapy within 1 hour of recognition.",
            "Administer 30 mL/kg crystalloid fluid bolus for hypotension or lactate >= 4 mmol/L.",
            "Measure serum lactate; remeasure within 2-4 hours if initial lactate is elevated.",
            "Apply vasopressors (norepinephrine first-line) if MAP < 65 mmHg after fluid resuscitation.",
        ]
    elif "ards" in condition_lower or "respiratory" in condition_lower:
        recs = [
            "Initiate lung-protective ventilation with tidal volume 6 mL/kg predicted body weight.",
            "Maintain plateau pressure < 30 cmH2O.",
            "Consider prone positioning for P/F ratio < 150.",
            "Use conservative fluid management strategy.",
            "Titrate PEEP using FiO2-PEEP table or best compliance method.",
        ]
    elif "aki" in condition_lower or "renal" in condition_lower or "kidney" in condition_lower:
        recs = [
            "Identify and treat reversible causes of AKI (e.g., obstruction, nephrotoxins).",
            "Optimize volume status and hemodynamics.",
            "Avoid nephrotoxic agents where possible.",
            "Monitor urine output and serum creatinine closely.",
            "Consider renal replacement therapy if indicated (refractory fluid overload, severe acidosis, hyperkalemia).",
        ]
    else:
        recs = [
            f"Continue monitoring for {condition} progression.",
            "Reassess clinical status and scoring at regular intervals.",
            "Consult specialist team if clinical trajectory worsens.",
        ]

    if sofa_score >= 6:
        recs.append("SOFA >= 6 indicates severe organ dysfunction; consider ICU escalation of care.")
    if qsofa_score >= 2:
        recs.append("Positive qSOFA (>= 2); maintain high clinical suspicion for sepsis.")

    return recs
