"""
LangGraph Orchestrator with mandatory Human-in-the-Loop
--------------------------------------------------------
Every pipeline run pauses for radiologist review before finalizing.

Flow:
  image_analysis → clinical_context → report_drafting → qa_validation
  → human_review (ALWAYS) → finalize
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
DB_PATH     = "data/checkpoints.db"


# ── State ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # inputs
    png_path:          str
    anonymized_id:     str
    modality:          str
    clinical_note:     str              # free text from radiologist
    # agent outputs
    image_findings:    ImageFindings | None
    clinical_context:  ClinicalContext | None
    report:            RadiologyReport | None
    validation:        ValidationResult | None
    # control
    retry_count:       int
    error:             str | None
    status:            str
    # HIL
    human_approved:    bool
    final_report_text: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_image_analysis(state: PipelineState) -> PipelineState:
    """Run vision model on scan. Skip if findings already provided."""
    if state.get("image_findings") is not None:
        logger.info("[node] image_analysis — skipping (findings pre-computed)")
        return {**state, "status": "image_analyzed"}

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
    """Fetch medical knowledge via Qdrant RAG + prior reports via MCP."""
    logger.info("[node] clinical_context | anon_id=%s", state["anonymized_id"])
    try:
        context = get_context(
            state["image_findings"],
            clinical_note=state.get("clinical_note", ""),
        )
        return {**state, "clinical_context": context, "status": "context_retrieved"}
    except Exception as e:
        logger.error("[node] clinical_context failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_report_drafting(state: PipelineState) -> PipelineState:
    """Generate full structured radiology report."""
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
    """Validate report for completeness, consistency, urgency."""
    logger.info("[node] qa_validation | anon_id=%s", state["anonymized_id"])
    try:
        result = validate(state["report"], state["image_findings"])
        return {
            **state,
            "validation":  result,
            "retry_count": state["retry_count"] + (0 if result.passed else 1),
            "status":      "validated" if result.passed else "qa_failed",
        }
    except Exception as e:
        logger.error("[node] qa_validation failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_human_review(state: PipelineState) -> PipelineState:
    """
    MANDATORY HIL — always pauses here for radiologist review.
    Every report goes through this regardless of QA score.
    Satisfies EU AI Act Art. 14 human oversight requirement.
    """
    logger.info(
        "[node] human_review — PAUSING | anon_id=%s",
        state["anonymized_id"],
    )

    validation = state.get("validation")
    report     = state.get("report")
    context    = state.get("clinical_context")

    human_input = interrupt({
        "message":             "Please review and approve this AI-generated radiology report.",
        "report":              report.report_text if report else "",
        "qa_score":            validation.score if validation else 0,
        "qa_passed":           validation.passed if validation else False,
        "qa_issues":           validation.issues if validation else [],
        "qa_warnings":         validation.warnings if validation else [],
        "urgency":             report.urgency_level if report else "unknown",
        "modality":            state["modality"],
        "anonymized_id":       state["anonymized_id"],
        "clinical_note":       state.get("clinical_note", ""),
        "prior_reports_found": bool(context and context.prior_reports_summary),
        "retry_count":         state["retry_count"],
    })

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
        "status":            "human_approved" if approved else "human_rejected",
    }


def node_finalize(state: PipelineState) -> PipelineState:
    """Finalize approved report — save to PostgreSQL in production."""
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
        "status":            "complete",
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_analysis(state: PipelineState) -> str:
    return "end" if state["status"] == "failed" else "continue"


def route_after_qa(state: PipelineState) -> str:
    """
    After QA:
      failed pipeline    → end
      QA passed          → mandatory human review
      QA failed + retries → retry drafting
      QA failed + max    → human review anyway
    """
    if state["status"] == "failed":
        return "end"

    validation = state.get("validation")

    if validation and validation.passed:
        return "human_review"

    if state["retry_count"] < MAX_RETRIES:
        logger.info(
            "QA failed — retrying (%d/%d)",
            state["retry_count"], MAX_RETRIES,
        )
        return "retry"

    logger.warning("Max retries — sending to human review with QA issues")
    return "human_review"


def route_after_human(state: PipelineState) -> str:
    return "finalize" if state.get("human_approved") else "end"


# ── Graph builder ─────────────────────────────────────────────────────────────

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
    graph.add_conditional_edges(
        "qa_validation",
        route_after_qa,
        {"human_review": "human_review", "retry": "report_drafting", "end": END},
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


# ── Public entry points ───────────────────────────────────────────────────────

def run_pipeline(
    png_path:          str,
    anonymized_id:     str,
    modality:          str = "CR",
    hil:               bool = True,
    thread_id:         str | None = None,
    existing_findings = None,
    clinical_note:     str = "",
) -> tuple[PipelineState, str | None]:
    """
    Run the full multi-agent pipeline.
    - existing_findings: skip image analysis if already computed
    - clinical_note: radiologist context improves report quality
    - hil: always True in production
    """
    load_dotenv("/home/moez/projects/radiology-ai/.env")
    os.makedirs("data", exist_ok=True)

    thread_id = thread_id or anonymized_id

    if hil:
        conn         = sqlite3.connect(DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph        = build_graph(checkpointer=checkpointer)
        config       = {"configurable": {"thread_id": thread_id}}
    else:
        graph  = build_graph()
        config = {}

    initial_state: PipelineState = {
        "png_path":          png_path,
        "anonymized_id":     anonymized_id,
        "modality":          modality,
        "clinical_note":     clinical_note,
        "image_findings":    existing_findings,
        "clinical_context":  None,
        "report":            None,
        "validation":        None,
        "retry_count":       0,
        "error":             None,
        "status":            "image_analyzed" if existing_findings else "started",
        "human_approved":    False,
        "final_report_text": "",
    }

    logger.info(
        "Starting pipeline | anon_id=%s | modality=%s | hil=%s | has_note=%s | skip_analysis=%s",
        anonymized_id, modality, hil,
        bool(clinical_note), existing_findings is not None,
    )

    final_state = graph.invoke(initial_state, config)
    logger.info("Pipeline status: %s", final_state.get("status"))
    return final_state, thread_id


def resume_pipeline(
    thread_id:       str,
    approved_report: str,
    approved:        bool = True,
) -> PipelineState:
    """Resume after radiologist review."""
    load_dotenv("/home/moez/projects/radiology-ai/.env")

    conn         = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph        = build_graph(checkpointer=checkpointer)
    config       = {"configurable": {"thread_id": thread_id}}

    logger.info("Resuming pipeline | thread_id=%s | approved=%s", thread_id, approved)

    final_state = graph.invoke(
        Command(resume={
            "approved_report": approved_report,
            "approved":        approved,
        }),
        config,
    )

    logger.info("Pipeline resumed | status=%s", final_state.get("status"))
    return final_state


# ── W&B tracking wrapper ──────────────────────────────────────────────────────

def run_pipeline_tracked(
    png_path:          str,
    anonymized_id:     str,
    modality:          str = "CR",
    hil:               bool = True,
    thread_id:         str = None,
    clinical_note:     str = "",
) -> tuple:
    """Run pipeline with automatic W&B tracking."""
    import time
    from mlops.tracking import log_pipeline_run, PipelineRunMetrics

    start_time = time.time()
    state, tid = run_pipeline(
        png_path, anonymized_id, modality,
        hil, thread_id, clinical_note=clinical_note,
    )
    latency = time.time() - start_time

    validation = state.get("validation")
    report     = state.get("report")
    findings   = state.get("image_findings")
    model_name = os.environ.get("GROQ_MODEL") or os.environ.get(
        "OLLAMA_MODEL", "unknown"
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
        requires_review=True,
        findings_count=len(findings.findings) if findings else 0,
        impression=report.impression if report else "",
        error=state.get("error"),
    )

    log_pipeline_run(metrics)
    return state, tid
