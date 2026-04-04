import logging
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.models.report import Report
from pipeline.preprocessor import preprocess
from agents.orchestrator import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/analyze")
async def analyze_scan(
    file: UploadFile = File(...),
    modality: str = Form(default="CR"),
    db: Session = Depends(get_db),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = preprocess(dicom_path=tmp_path, output_dir="data/processed")
        png_path = result["png_path"]
        anonymized_id = result["anonymized_id"]

        state, thread_id = run_pipeline(
            png_path=png_path,
            anonymized_id=anonymized_id,
            modality=modality,
            hil=True,
        )

        if state.get("error"):
            return {"status": "error", "error": state["error"]}

        report_obj = state.get("report")
        validation = state.get("validation")

        if not report_obj:
            return {"status": "error", "error": "Pipeline produced no report"}

        db_report = Report(
            anonymized_id=anonymized_id,
            modality=modality,
            report_text=report_obj.report_text,
            clinical_indication=report_obj.clinical_indication,
            technique=report_obj.technique,
            findings=report_obj.findings,
            impression=report_obj.impression,
            recommendations=report_obj.recommendations,
            qa_score=validation.score if validation else 0.0,
            qa_passed=validation.passed if validation else False,
            urgency_level=report_obj.urgency_level,
            retry_count=state.get("retry_count", 0),
            png_path=png_path,
        )
        db.add(db_report)
        db.commit()
        db.refresh(db_report)

        logger.info("Pipeline complete | report_id=%s", db_report.id)

        return {
            "status": state.get("status"),
            "report_id": db_report.id,
            "anonymized_id": anonymized_id,
            "thread_id": thread_id,
            "report_text": report_obj.report_text,
            "qa_score": validation.score if validation else 0,
            "qa_passed": validation.passed if validation else False,
            "urgency_level": report_obj.urgency_level,
            "requires_human_review": validation.requires_human_review if validation else False,
        }

    finally:
        os.unlink(tmp_path)
