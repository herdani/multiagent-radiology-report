"""
Compliance API endpoints
-------------------------
GDPR and HIPAA management endpoints.
In production these would require special admin authentication.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.compliance import (
    run_retention_cleanup,
    erase_patient_data,
    generate_compliance_report,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/compliance", tags=["compliance"])


@router.post("/retention/cleanup")
def retention_cleanup(db: Session = Depends(get_db)):
    """
    Run GDPR data retention cleanup.
    Deletes reports past their retention period.
    Should be called daily by a scheduler.
    """
    result = run_retention_cleanup(db)
    return result


@router.delete("/erase/{anonymized_id}")
def erase_data(
    anonymized_id: str,
    requested_by: str = "patient",
    db: Session = Depends(get_db),
):
    """
    GDPR Article 17 — right to erasure.
    Deletes all reports for the given anonymized ID.
    """
    result = erase_patient_data(db, anonymized_id, requested_by)
    return result


@router.get("/report")
def compliance_report(db: Session = Depends(get_db)):
    """
    Generate compliance summary for DPO reporting.
    """
    return generate_compliance_report(db)
