import json
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.models.report import Report, AuditLog

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])


class ReportCreate(BaseModel):
    anonymized_id:       str
    modality:            str
    report_text:         str
    clinical_indication: str = ""
    technique:           str = ""
    findings:            str = ""
    impression:          str = ""
    recommendations:     str = ""
    qa_score:            float = 0.0
    qa_passed:           bool = False
    urgency_level:       str = "routine"
    retry_count:         int = 0
    png_path:            Optional[str] = None


class ReportApprove(BaseModel):
    approved_report_text: str
    approved_by:          str = "radiologist"


class ReportResponse(BaseModel):
    id:             str
    anonymized_id:  str
    modality:       str
    report_text:    str
    qa_score:       float
    qa_passed:      bool
    urgency_level:  str
    human_approved: bool
    created_at:     datetime

    class Config:
        from_attributes = True


def log_action(
    db, action, anonymized_id,
    report_id=None, performed_by="system",
    request=None, details=None,
):
    entry = AuditLog(
        report_id=report_id,
        anonymized_id=anonymized_id,
        action=action,
        performed_by=performed_by,
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None,
        details=json.dumps(details) if details else None,
    )
    db.add(entry)
    db.commit()


@router.post("/", response_model=ReportResponse)
def create_report(data: ReportCreate, request: Request, db: Session = Depends(get_db)):
    report = Report(**data.model_dump())
    db.add(report)
    db.commit()
    db.refresh(report)
    log_action(db, "created", data.anonymized_id, report_id=report.id, request=request)
    logger.info("Report created | id=%s", report.id)
    return report


@router.get("/{report_id}", response_model=ReportResponse)
def get_report(report_id: str, request: Request, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    log_action(db, "viewed", report.anonymized_id, report_id=report_id, request=request)
    return report


@router.get("/scan/{anonymized_id}", response_model=list[ReportResponse])
def get_reports_by_scan(anonymized_id: str, db: Session = Depends(get_db)):
    return db.query(Report).filter(
        Report.anonymized_id == anonymized_id
    ).order_by(Report.created_at.desc()).all()


@router.post("/{report_id}/approve", response_model=ReportResponse)
def approve_report(
    report_id: str, data: ReportApprove,
    request: Request, db: Session = Depends(get_db)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report.report_text    = data.approved_report_text
    report.human_approved = True
    report.approved_by    = data.approved_by
    report.approved_at    = datetime.utcnow()
    db.commit()
    db.refresh(report)
    log_action(db, "approved", report.anonymized_id, report_id=report_id,
               performed_by=data.approved_by, request=request)
    return report


@router.post("/{report_id}/reject")
def reject_report(report_id: str, request: Request, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    log_action(db, "rejected", report.anonymized_id, report_id=report_id, request=request)
    return {"status": "rejected", "report_id": report_id}


@router.get("/")
def list_reports(
    skip: int = 0, limit: int = 20,
    urgency: Optional[str] = None,
    approved: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Report)
    if urgency:
        query = query.filter(Report.urgency_level == urgency)
    if approved is not None:
        query = query.filter(Report.human_approved == approved)
    return query.order_by(Report.created_at.desc()).offset(skip).limit(limit).all()
