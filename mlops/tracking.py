"""
W&B MLOps Tracking
-------------------
Logs every pipeline run to Weights & Biases automatically.
Tracks: model versions, QA scores, latency, findings, urgency distribution.

Every call to log_pipeline_run() creates a W&B run with:
  - config  (model, modality, pipeline version)
  - metrics (qa_score, latency, retry_count)
  - summary (impression, urgency, human_approved)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineRunMetrics:
    """Everything we want to track for one pipeline execution."""

    anonymized_id: str
    modality: str
    model_name: str
    qa_score: float
    qa_passed: bool
    urgency_level: str
    retry_count: int
    latency_seconds: float
    human_approved: bool
    requires_review: bool
    findings_count: int
    impression: str
    pipeline_version: str = "0.1.0"
    error: Optional[str] = None


def log_pipeline_run(metrics: PipelineRunMetrics) -> bool:
    """
    Log a completed pipeline run to W&B.
    Returns True if logging succeeded, False if W&B is unavailable.

    Safe to call in production — never raises, just logs a warning if W&B fails.
    """
    api_key = os.environ.get("WANDB_API_KEY", "")
    project = os.environ.get("WANDB_PROJECT", "radiology-ai")

    if not api_key or api_key == "your-wandb-key":
        logger.info("W&B not configured — skipping tracking")
        return False

    try:
        import wandb

        wandb.init(
            project=project,
            name=f"{metrics.modality}_{metrics.anonymized_id[:8]}",
            tags=[
                metrics.modality,
                metrics.urgency_level,
                "approved" if metrics.human_approved else "pending",
                "error" if metrics.error else "success",
            ],
            config={
                # model config
                "model_name": metrics.model_name,
                "modality": metrics.modality,
                "pipeline_version": metrics.pipeline_version,
            },
        )

        # log numeric metrics
        wandb.log(
            {
                "qa_score": metrics.qa_score,
                "retry_count": metrics.retry_count,
                "latency_seconds": metrics.latency_seconds,
                "findings_count": metrics.findings_count,
            }
        )

        # log summary (non-numeric)
        wandb.run.summary["urgency_level"] = metrics.urgency_level
        wandb.run.summary["qa_passed"] = metrics.qa_passed
        wandb.run.summary["human_approved"] = metrics.human_approved
        wandb.run.summary["requires_review"] = metrics.requires_review
        wandb.run.summary["impression"] = metrics.impression[:200]
        wandb.run.summary["anonymized_id"] = metrics.anonymized_id

        if metrics.error:
            wandb.run.summary["error"] = metrics.error

        wandb.finish()

        logger.info(
            "W&B run logged | project=%s | anon_id=%s | qa=%.2f",
            project,
            metrics.anonymized_id,
            metrics.qa_score,
        )
        return True

    except Exception as e:
        logger.warning("W&B logging failed (non-fatal): %s", e)
        return False


def log_model_evaluation(
    model_name: str,
    dataset_size: int,
    accuracy: float,
    avg_qa_score: float,
    avg_latency: float,
    notes: str = "",
) -> bool:
    """
    Log a model evaluation run to W&B.
    Use this when comparing Qwen vs MedGemma vs other models.
    """
    api_key = os.environ.get("WANDB_API_KEY", "")
    project = os.environ.get("WANDB_PROJECT", "radiology-ai")

    if not api_key or api_key == "your-wandb-key":
        return False

    try:
        import wandb

        wandb.init(
            project=project,
            name=f"eval_{model_name}",
            job_type="model_evaluation",
            tags=["evaluation", model_name],
            config={
                "model_name": model_name,
                "dataset_size": dataset_size,
                "notes": notes,
            },
        )

        wandb.log(
            {
                "accuracy": accuracy,
                "avg_qa_score": avg_qa_score,
                "avg_latency": avg_latency,
                "dataset_size": dataset_size,
            }
        )

        wandb.finish()
        return True

    except Exception as e:
        logger.warning("W&B evaluation logging failed: %s", e)
        return False
