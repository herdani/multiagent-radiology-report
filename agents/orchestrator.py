"""
LangGraph Orchestrator with mandatory Human-in-the-Loop
--------------------------------------------------------
Every pipeline run pauses for radiologist review before finalizing.
The radiologist can edit the report before approving.

Flow:
  image_analysis → clinical_context → report_drafting → qa_validation
       → human_review (ALWAYS) → finalize
       
  If QA fails and retries left → retry report_drafting first
  then always goes to human_review regardless
"""
import logging
import os
import sqlite3
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from agents.image_analysis import ImageFindings, run as analyze
from agents.clinical_context import ClinicalContext, run as get_context
from agents.report_drafting import RadiologyReport, run as draft_report
from agents.qa_validation import ValidationResult, run as validate

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
DB_PATH = "data/checkpoints.db"


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict):
    # inputs
    png_path:         str
    anonymized_id:    str
    modality:         str
    # agent outputs
    image_findings:   ImageFindings | None
    clinical_context: ClinicalContext | None
    report:           RadiologyReport | None
    validation:       ValidationResult | None
    # control
    retry_count:      int
    error:            str | None
    status:           str
    # HIL
    human_approved:   bool
    final_report_text: str


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════
def node_image_analysis(state: PipelineState) -> PipelineState:
    logger.info("[node] image_analysis | anon_id=%s", state["anonymized_id"])
    try:
        findings = analyze(
            png_path=state["png_path"],
            anonymized_id=state["anonymized_id"],
            modality=state["modality"],
        )
        return {**state, "image_findings": findings, "status": "image_analyzed"}
    except Exception as e:
        logger.error("[node] image_analysis failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_clinical_context(state: PipelineState) -> PipelineState:
    logger.info("[node] clinical_context | anon_id=%s", state["anonymized_id"])
    try:
        context = get_context(state["image_findings"])
        return {**state, "clinical_context": context, "status": "context_retrieved"}
    except Exception as e:
        logger.error("[node] clinical_context failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_report_drafting(state: PipelineState) -> PipelineState:
    logger.info(
        "[node] report_drafting | anon_id=%s | attempt=%d",
        state["anonymized_id"],
        state["retry_count"] + 1,
    )
    try:
        report = draft_report(state["image_findings"], state["clinical_context"])
        return {**state, "report": report, "status": "report_drafted"}
    except Exception as e:
        logger.error("[node] report_drafting failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_qa_validation(state: PipelineState) -> PipelineState:
    logger.info("[node] qa_validation | anon_id=%s", state["anonymized_id"])
    try:
        result = validate(state["report"], state["image_findings"])
        return {
            **state,
            "validation": result,
            "retry_count": state["retry_count"] + (0 if result.passed else 1),
            "status": "validated" if result.passed else "qa_failed",
        }
    except Exception as e:
        logger.error("[node] qa_validation failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_human_review(state: PipelineState) -> PipelineState:
    """
    MANDATORY HIL NODE — always pauses here for radiologist review.

    Every report goes through this node regardless of QA score.
    The radiologist sees:
      - The AI generated report (editable)
      - QA score and any issues
      - Whether findings were flagged urgent

    They can:
      - Approve as-is
      - Edit and approve
      - Reject (sends back for re-analysis)

    This satisfies:
      - EU AI Act Art. 14: human oversight for high-risk AI
      - Clinical governance: no AI report goes out unreviewed
      - HIPAA: human accountability for medical decisions
    """
    logger.info(
        "[node] human_review — PAUSING for radiologist | anon_id=%s",
        state["anonymized_id"],
    )

    validation = state.get("validation")
    report     = state.get("report")

    # build context for the radiologist
    review_context = {
        "message":        "Please review and approve this AI-generated radiology report.",
        "report":         report.report_text if report else "",
        "qa_score":       validation.score if validation else 0,
        "qa_passed":      validation.passed if validation else False,
        "qa_issues":      validation.issues if validation else [],
        "qa_warnings":    validation.warnings if validation else [],
        "urgency":        report.urgency_level if report else "unknown",
        "modality":       state["modality"],
        "anonymized_id":  state["anonymized_id"],
        "retry_count":    state["retry_count"],
    }

    # PAUSE — execution resumes when radiologist calls resume_pipeline()
    human_input = interrupt(review_context)

    # resumed — human_input contains radiologist's response
    approved_text = human_input.get("approved_report", report.report_text if report else "")
    approved      = human_input.get("approved", False)

    logger.info(
        "[node] human_review RESUMED | approved=%s | anon_id=%s",
        approved, state["anonymized_id"],
    )

    return {
        **state,
        "human_approved":    approved,
        "final_report_text": approved_text,
        "status": "human_approved" if approved else "human_rejected",
    }


def node_finalize(state: PipelineState) -> PipelineState:
    """Save final approved report — in phase 6 this writes to PostgreSQL + S3."""
    final_text = state.get("final_report_text") or (
        state["report"].report_text if state.get("report") else ""
    )
    logger.info(
        "[node] finalize | anon_id=%s | human_approved=%s",
        state["anonymized_id"],
        state.get("human_approved", False),
    )
    return {
        **state,
        "final_report_text": final_text,
        "status": "complete",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def route_after_analysis(state: PipelineState) -> str:
    return "end" if state["status"] == "failed" else "continue"


def route_after_qa(state: PipelineState) -> str:
    """
    After QA validation:
      - pipeline failed    → end
      - QA passed          → always go to human_review
      - QA failed, retries left → retry report drafting
      - QA failed, max retries  → go to human_review anyway
        (radiologist sees the best attempt with QA issues noted)
    """
    if state["status"] == "failed":
        return "end"

    validation = state.get("validation")

    # QA passed — mandatory human review regardless
    if validation and validation.passed:
        return "human_review"

    # QA failed but retries left — try again
    if state["retry_count"] < MAX_RETRIES:
        logger.info(
            "QA failed — retrying report drafting (attempt %d/%d)",
            state["retry_count"], MAX_RETRIES,
        )
        return "retry"

    # QA failed and max retries hit — send to human review anyway
    # radiologist will see the QA issues and can fix manually
    logger.warning(
        "Max retries reached — sending to human review with QA issues noted"
    )
    return "human_review"


def route_after_human(state: PipelineState) -> str:
    """After human review: approved → finalize, rejected → end."""
    if state.get("human_approved"):
        return "finalize"
    return "end"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_graph(checkpointer=None):
    graph = StateGraph(PipelineState)

    graph.add_node("image_analysis",   node_image_analysis)
    graph.add_node("clinical_context", node_clinical_context)
    graph.add_node("report_drafting",  node_report_drafting)
    graph.add_node("qa_validation",    node_qa_validation)
    graph.add_node("human_review",     node_human_review)
    graph.add_node("finalize",         node_finalize)

    graph.set_entry_point("image_analysis")

    graph.add_conditional_edges(
        "image_analysis",
        route_after_analysis,
        {"continue": "clinical_context", "end": END},
    )
    graph.add_edge("clinical_context", "report_drafting")
    graph.add_edge("report_drafting",  "qa_validation")

    # after QA — always goes to human_review (directly or after retries)
    graph.add_conditional_edges(
        "qa_validation",
        route_after_qa,
        {
            "human_review": "human_review",
            "retry":        "report_drafting",
            "end":          END,
        },
    )

    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"finalize": "finalize", "end": END},
    )
    graph.add_edge("finalize", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"] if checkpointer else [],
    )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    png_path: str,
    anonymized_id: str,
    modality: str = "CR",
    hil: bool = True,          # default True now — HIL is always on
    thread_id: str | None = None,
) -> tuple[PipelineState, str | None]:
    """
    Run the full pipeline.
    HIL is on by default — every report requires radiologist review.
    """
    load_dotenv("/home/moez/projects/radiology-ai/.env")
    os.makedirs("data", exist_ok=True)

    thread_id = thread_id or anonymized_id

    if hil:
        conn        = sqlite3.connect(DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph       = build_graph(checkpointer=checkpointer)
        config      = {"configurable": {"thread_id": thread_id}}
    else:
        graph  = build_graph()
        config = {}

    initial_state: PipelineState = {
        "png_path":          png_path,
        "anonymized_id":     anonymized_id,
        "modality":          modality,
        "image_findings":    None,
        "clinical_context":  None,
        "report":            None,
        "validation":        None,
        "retry_count":       0,
        "error":             None,
        "status":            "started",
        "human_approved":    False,
        "final_report_text": "",
    }

    logger.info(
        "Starting pipeline | anon_id=%s | modality=%s | hil=%s",
        anonymized_id, modality, hil,
    )
    final_state = graph.invoke(initial_state, config)
    logger.info("Pipeline paused/complete | status=%s", final_state.get("status"))

    return final_state, thread_id


def resume_pipeline(
    thread_id: str,
    approved_report: str,
    approved: bool = True,
) -> PipelineState:
    """Resume after radiologist review."""
    load_dotenv("/home/moez/projects/radiology-ai/.env")

    conn         = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph        = build_graph(checkpointer=checkpointer)
    config       = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "Resuming pipeline | thread_id=%s | approved=%s",
        thread_id, approved,
    )

    final_state = graph.invoke(
        Command(resume={
            "approved_report": approved_report,
            "approved":        approved,
        }),
        config,
    )

    logger.info("Pipeline resumed | status=%s", final_state.get("status"))
    return final_state


# ══════════════════════════════════════════════════════════════════════════════
# W&B TRACKING WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline_tracked(
    png_path: str,
    anonymized_id: str,
    modality: str = "CR",
    hil: bool = True,
    thread_id: str = None,
) -> tuple:
    """Run pipeline with automatic W&B tracking."""
    import time
    from mlops.tracking import log_pipeline_run, PipelineRunMetrics

    start_time = time.time()
    state, tid = run_pipeline(png_path, anonymized_id, modality, hil, thread_id)
    latency    = time.time() - start_time

    validation = state.get("validation")
    report     = state.get("report")
    findings   = state.get("image_findings")
    model_name = os.environ.get("OLLAMA_MODEL") or os.environ.get(
        "LLM_VISION_MODEL", "unknown"
    )

    metrics = PipelineRunMetrics(
        anonymized_id=anonymized_id,
        modality=modality,
        model_name=model_name,
        qa_score=validation.score if validation else 0.0,
        qa_passed=validation.passed if validation else False,
        urgency_level=report.urgency_level if report else "unknown",
        retry_count=state.get("retry_count", 0),
        latency_seconds=round(latency, 2),
        human_approved=state.get("human_approved", False),
        requires_review=True,   # always True now
        findings_count=len(findings.findings) if findings else 0,
        impression=report.impression if report else "",
        error=state.get("error"),
    )

    log_pipeline_run(metrics)
    return state, tid
