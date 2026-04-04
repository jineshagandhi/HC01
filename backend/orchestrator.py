"""
Multi-Agent Orchestrator
Coordinates all 4 agents: Note Parser, Temporal Mapper, Guideline RAG, Chief Synthesis
Runs Agents 1-2 in parallel, then Agent 3, then Agent 4 synthesizes.
"""
import concurrent.futures
from datetime import datetime
import pandas as pd
from backend.agents.note_parser import parse_clinical_note, parse_all_notes
from backend.agents.temporal_mapper import (
    build_timeline, calculate_scores_over_time,
    detect_all_trends, get_disease_progression_summary
)
from backend.agents.guideline_rag import (
    get_relevant_guidelines, format_citations,
    get_guideline_recommendations
)
from backend.agents.chief_synthesis import (
    synthesize_report, generate_shift_handoff, generate_family_summary
)


def run_agent1(notes: list[dict]) -> dict:
    """Agent 1: Note Parser — Extract medical entities from clinical notes."""
    try:
        parsed_notes = []
        for note in notes:
            parsed = parse_clinical_note(note.get("text", ""))
            parsed["timestamp"] = note.get("timestamp")
            parsed["author"] = note.get("author", "Unknown")
            parsed["note_type"] = note.get("note_type", "Progress Note")
            parsed_notes.append(parsed)

        aggregated = parse_all_notes(notes)
        return {
            "status": "success",
            "parsed_notes": parsed_notes,
            "aggregated": aggregated
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "parsed_notes": [],
            "aggregated": {
                "all_symptoms": [], "all_medications": [],
                "all_conditions": [], "urgency_timeline": [],
                "critical_findings": []
            }
        }


def run_agent2(vitals_df, labs_df, notes: list[dict]) -> dict:
    """Agent 2: Temporal Lab Mapper — Build timeline and calculate scores."""
    try:
        scores = calculate_scores_over_time(vitals_df, labs_df)
        trends = detect_all_trends(labs_df)
        timeline = build_timeline(vitals_df, labs_df, notes, scores)
        progression = get_disease_progression_summary(timeline, scores, trends)

        return {
            "status": "success",
            "scores": scores,
            "trends": trends,
            "timeline": timeline,
            "disease_progression": progression
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "scores": [], "trends": [], "timeline": [],
            "disease_progression": "Unable to generate progression summary."
        }


def run_agent3(risk_flags: list[str], patient_context: str,
               sofa_score: int, qsofa_score: int) -> dict:
    """Agent 3: Guideline RAG — Retrieve relevant clinical guidelines."""
    try:
        guidelines = get_relevant_guidelines(risk_flags, patient_context)
        citations = format_citations(guidelines)

        primary_condition = risk_flags[0] if risk_flags else "sepsis"
        recommendations = get_guideline_recommendations(
            primary_condition, sofa_score, qsofa_score
        )

        return {
            "status": "success",
            "guidelines": guidelines,
            "citations": citations,
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "guidelines": [], "citations": "",
            "recommendations": {"recommendations": [], "evidence_summary": ""}
        }


def run_full_pipeline(patient_info: dict, vitals_df, labs_df,
                      notes: list[dict]) -> dict:
    """
    Run the complete multi-agent pipeline.

    Args:
        patient_info: dict with patient_id, age, gender, diagnosis
        vitals_df: DataFrame of vital signs
        labs_df: DataFrame of lab results
        notes: list of clinical note dicts

    Returns:
        Complete analysis results from all agents + synthesized report
    """
    results = {"started_at": datetime.now().isoformat()}

    # Normalize DataFrames to agent-expected column names
    vitals_norm = _normalize_vitals_for_agents(vitals_df)
    labs_norm = _normalize_labs_for_agents(labs_df)

    # === PHASE 1: Run Agents 1 & 2 in parallel, then Agent 3 sequentially ===
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_agent1 = executor.submit(run_agent1, notes)
        future_agent2 = executor.submit(run_agent2, vitals_norm, labs_norm, notes)

        # Wait for both to finish
        agent2_result = future_agent2.result()
        agent1_result = future_agent1.result()

    # Extract info for Agent 3
    scores = agent2_result.get("scores", [])
    latest_sofa = scores[-1]["sofa_total"] if scores else 0
    latest_qsofa = scores[-1]["qsofa_total"] if scores else 0
    trends = agent2_result.get("trends", [])

    # Build risk flags from Agent 1 & 2 outputs
    risk_flags = _build_risk_flags(agent1_result, agent2_result, patient_info)

    patient_context = (
        f"Patient: {patient_info.get('age', 'Unknown')}yo "
        f"{patient_info.get('gender', 'Unknown')}, "
        f"admitted for {patient_info.get('diagnosis', 'unknown')}. "
        f"Current SOFA: {latest_sofa}, qSOFA: {latest_qsofa}."
    )

    # Run Agent 3
    agent3_result = run_agent3(risk_flags, patient_context, latest_sofa, latest_qsofa)

    # === PHASE 2: Agent 4 — Chief Synthesis ===
    report = synthesize_report(
        note_parser_output=agent1_result,
        temporal_output=agent2_result,
        rag_output=agent3_result,
        patient_info=patient_info,
        vitals_df=vitals_norm,
        labs_df=labs_norm
    )

    results["agent1"] = agent1_result
    results["agent2"] = agent2_result
    results["agent3"] = agent3_result
    results["report"] = report
    results["completed_at"] = datetime.now().isoformat()

    return results


def run_shift_handoff(report: dict, patient_info: dict,
                      current_shift: str = "Day") -> dict:
    """Generate a shift handoff report from existing analysis."""
    return generate_shift_handoff(report, patient_info, current_shift)


def run_family_communication(report: dict, patient_info: dict,
                             regional_language: str = "Hindi") -> dict:
    """Generate a compassionate family-friendly summary from existing analysis.

    Produces a jargon-free summary of the patient's last 12 hours,
    translated into both English and a regional language.
    """
    return generate_family_summary(report, patient_info, regional_language)


def _build_risk_flags(agent1_result: dict, agent2_result: dict,
                      patient_info: dict) -> list[str]:
    """Build risk flag descriptions from agent outputs."""
    flags = []

    # From Agent 1: conditions and critical findings
    aggregated = agent1_result.get("aggregated", {})
    for condition in aggregated.get("all_conditions", []):
        if isinstance(condition, dict):
            flags.append(condition.get("condition", condition.get("value", "")))
        else:
            flags.append(str(condition))

    for finding in aggregated.get("critical_findings", []):
        flags.append(str(finding))

    # From Agent 2: trends
    for trend in agent2_result.get("trends", []):
        if trend.get("is_concerning"):
            flags.append(
                f"{trend['lab_name']} {trend['trend']} - {trend.get('description', '')}"
            )

    # From Agent 2: scores
    scores = agent2_result.get("scores", [])
    if scores:
        latest = scores[-1]
        if latest.get("sofa_total", 0) >= 2:
            flags.append(f"SOFA score {latest['sofa_total']} (organ dysfunction)")
        if latest.get("qsofa_total", 0) >= 2:
            flags.append(f"qSOFA {latest['qsofa_total']}/3 (high sepsis risk)")

    # From patient info
    diagnosis = patient_info.get("diagnosis", "")
    if diagnosis:
        flags.append(f"Admitted for: {diagnosis}")

    # Default if nothing found
    if not flags:
        flags = ["General ICU monitoring", "Assess for sepsis risk"]

    return flags[:10]  # Limit to top 10 flags


def _normalize_vitals_for_agents(vitals_df) -> pd.DataFrame:
    """
    Convert wide-format vitals (from ingestion) to long-format
    with columns: timestamp, parameter, value — as expected by agents.
    """
    if vitals_df is None or (isinstance(vitals_df, pd.DataFrame) and vitals_df.empty):
        return pd.DataFrame(columns=["timestamp", "parameter", "value"])

    if not isinstance(vitals_df, pd.DataFrame):
        return pd.DataFrame(columns=["timestamp", "parameter", "value"])

    # If already has agent columns, return as-is
    if "parameter" in vitals_df.columns and "timestamp" in vitals_df.columns:
        return vitals_df

    # Determine time column
    time_col = None
    for col in ["charttime", "timestamp", "time"]:
        if col in vitals_df.columns:
            time_col = col
            break

    if time_col is None:
        return pd.DataFrame(columns=["timestamp", "parameter", "value"])

    # Wide format (pivoted vitals from ingestion) -> long format
    # Columns like heart_rate, sbp, dbp, respiratory_rate, spo2, temperature, gcs
    param_map = {
        "heart_rate": "HR",
        "sbp": "SBP",
        "dbp": "DBP",
        "respiratory_rate": "RR",
        "spo2": "SpO2",
        "temperature": "Temperature",
        "gcs": "GCS",
    }

    rows = []
    for _, row in vitals_df.iterrows():
        ts = row[time_col]
        for col_name, param_name in param_map.items():
            if col_name in vitals_df.columns and pd.notna(row.get(col_name)):
                rows.append({
                    "timestamp": pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts,
                    "parameter": param_name,
                    "value": float(row[col_name]),
                })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "parameter", "value"])

    return pd.DataFrame(rows)


def _normalize_labs_for_agents(labs_df) -> pd.DataFrame:
    """
    Normalize lab DataFrame columns to: timestamp, parameter, value
    as expected by agents.
    """
    if labs_df is None or (isinstance(labs_df, pd.DataFrame) and labs_df.empty):
        return pd.DataFrame(columns=["timestamp", "parameter", "value"])

    if not isinstance(labs_df, pd.DataFrame):
        return pd.DataFrame(columns=["timestamp", "parameter", "value"])

    # If already normalized, return as-is
    if "parameter" in labs_df.columns and "timestamp" in labs_df.columns:
        return labs_df

    result = labs_df.copy()

    # Rename columns to match agent expectations
    rename_map = {}
    if "charttime" in result.columns and "timestamp" not in result.columns:
        rename_map["charttime"] = "timestamp"
    if "label" in result.columns and "parameter" not in result.columns:
        rename_map["label"] = "parameter"
    if "valuenum" in result.columns and "value" not in result.columns:
        rename_map["valuenum"] = "value"

    if rename_map:
        result = result.rename(columns=rename_map)

    # Ensure required columns exist
    for col in ["timestamp", "parameter", "value"]:
        if col not in result.columns:
            result[col] = None

    # Parse timestamps
    if result["timestamp"].dtype == object:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    # Ensure value column is numeric — drop rows with non-numeric values like '___'
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["value", "timestamp"])

    return result
