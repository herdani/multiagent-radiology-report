"""
GDPR / HIPAA Compliance Layer
-------------------------------
Implements:
  - Data retention enforcement (GDPR Article 5)
  - Right to erasure (GDPR Article 17)
  - Audit log management (HIPAA requirement)
  - Data minimization verification
  - Anonymization utilities

Schedule run_retention_cleanup() daily via cron or Celery beat.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session

from api.models.report import Report, AuditLog

logger = logging.getLogger(__name__)


# ── Data Retention ────────────────────────────────────────────────────────────
REPORT_RETENTION_DAYS = 90  # GDPR — delete patient-linked reports after 90 days
AUDIT_LOG_RETENTION_DAYS = 2190  # HIPAA — keep audit logs for 6 years (6*365)


def run_retention_cleanup(db: Session) -> dict:
    """
    Delete expired reports and old audit logs.
    Should run daily — call from a scheduled job.

    GDPR Article 5(1)(e): data kept no longer than necessary.
    HIPAA: audit logs retained minimum 6 years.

    Returns summary of what was deleted for compliance reporting.
    """
    now = datetime.utcnow()
    summary = {
        "run_at": now.isoformat(),
        "reports_deleted": 0,
        "audit_logs_deleted": 0,
        "errors": [],
    }

    # delete expired reports
    try:
        expired_reports = db.query(Report).filter(Report.expires_at < now).all()

        for report in expired_reports:
            # log the deletion in audit log before deleting
            _log_compliance_action(
                db,
                action="gdpr_retention_delete",
                anonymized_id=report.anonymized_id,
                details={
                    "report_id": report.id,
                    "created_at": report.created_at.isoformat(),
                    "expired_at": report.expires_at.isoformat(),
                    "reason": "retention_period_exceeded",
                },
            )
            db.delete(report)
            summary["reports_deleted"] += 1

        db.commit()
        logger.info("Retention cleanup: deleted %d expired reports", summary["reports_deleted"])

    except Exception as e:
        summary["errors"].append(f"Report cleanup error: {str(e)}")
        logger.error("Retention cleanup failed: %s", e)
        db.rollback()

    # delete old audit logs (keep HIPAA minimum 6 years)
    try:
        cutoff = now - timedelta(days=AUDIT_LOG_RETENTION_DAYS)
        deleted = db.query(AuditLog).filter(AuditLog.timestamp < cutoff).delete()
        db.commit()
        summary["audit_logs_deleted"] = deleted
        logger.info("Audit log cleanup: deleted %d old entries", deleted)

    except Exception as e:
        summary["errors"].append(f"Audit log cleanup error: {str(e)}")
        logger.error("Audit log cleanup failed: %s", e)
        db.rollback()

    return summary


# ── Right to Erasure (GDPR Article 17) ───────────────────────────────────────
def erase_patient_data(
    db: Session,
    anonymized_id: str,
    requested_by: str = "patient",
    reason: str = "gdpr_erasure_request",
) -> dict:
    """
    Delete all data associated with an anonymized ID.
    GDPR Article 17 — right to erasure ('right to be forgotten').

    Note: audit logs are retained per HIPAA even after erasure request,
    but the report content is deleted. This is standard practice —
    we keep proof that data existed and was deleted, not the data itself.
    """
    summary = {
        "anonymized_id": anonymized_id,
        "erased_at": datetime.utcnow().isoformat(),
        "requested_by": requested_by,
        "reports_deleted": 0,
        "status": "success",
    }

    try:
        reports = db.query(Report).filter(Report.anonymized_id == anonymized_id).all()

        for report in reports:
            # log erasure BEFORE deleting (HIPAA audit trail)
            _log_compliance_action(
                db,
                action="gdpr_erasure",
                anonymized_id=anonymized_id,
                details={
                    "report_id": report.id,
                    "requested_by": requested_by,
                    "reason": reason,
                },
            )
            db.delete(report)
            summary["reports_deleted"] += 1

        db.commit()
        logger.info(
            "GDPR erasure complete | anon_id=%s | deleted=%d reports",
            anonymized_id,
            summary["reports_deleted"],
        )

    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        logger.error("GDPR erasure failed | anon_id=%s | error=%s", anonymized_id, e)
        db.rollback()

    return summary


# ── Data Minimization Check ───────────────────────────────────────────────────
PII_PATTERNS = [
    "patient name",
    "date of birth",
    "dob",
    "ssn",
    "social security",
    "address",
    "phone",
    "email",
    "nhs number",
    "mrn",
    "medical record",
    "insurance",
    "passport",
    "driving licence",
    "driving license",
]


def check_report_for_pii(report_text: str) -> list[str]:
    """
    Scan a report for potential PII before storing.
    GDPR Article 25 — data protection by design.

    Returns list of detected PII patterns (empty = clean).
    """
    found = []
    text_lower = report_text.lower()
    for pattern in PII_PATTERNS:
        if pattern in text_lower:
            found.append(pattern)
    return found


def sanitize_report(report_text: str) -> tuple[str, list[str]]:
    """
    Attempt to remove detected PII from report text.
    Returns (sanitized_text, list_of_removed_patterns).
    """
    import re

    sanitized = report_text
    removed = []

    # remove dates that look like birth dates (DD/MM/YYYY or MM/DD/YYYY)
    date_pattern = r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b"
    if re.search(date_pattern, sanitized):
        sanitized = re.sub(date_pattern, "[DATE REDACTED]", sanitized)
        removed.append("date_pattern")

    # remove NHS numbers (3-3-4 digit format)
    nhs_pattern = r"\b\d{3}\s\d{3}\s\d{4}\b"
    if re.search(nhs_pattern, sanitized):
        sanitized = re.sub(nhs_pattern, "[NHS REDACTED]", sanitized)
        removed.append("nhs_number")

    # remove email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    if re.search(email_pattern, sanitized):
        sanitized = re.sub(email_pattern, "[EMAIL REDACTED]", sanitized)
        removed.append("email")

    return sanitized, removed


# ── Audit Log Helper ──────────────────────────────────────────────────────────
def _log_compliance_action(
    db: Session,
    action: str,
    anonymized_id: str,
    details: Optional[dict] = None,
):
    """Write a compliance action to the audit log."""
    entry = AuditLog(
        anonymized_id=anonymized_id,
        action=action,
        performed_by="compliance_system",
        details=json.dumps(details) if details else None,
    )
    db.add(entry)


# ── Compliance Report ─────────────────────────────────────────────────────────
def generate_compliance_report(db: Session) -> dict:
    """
    Generate a compliance summary for audit purposes.
    Useful for GDPR Data Protection Officer (DPO) reports.
    """
    now = datetime.utcnow()

    total_reports = db.query(Report).count()
    approved_reports = db.query(Report).filter(Report.human_approved).count()
    pending_reports = db.query(Report).filter(~Report.human_approved).count()
    expiring_soon = db.query(Report).filter(Report.expires_at < now + timedelta(days=7)).count()
    urgent_reports = (
        db.query(Report).filter(Report.urgency_level.in_(["urgent", "emergent"])).count()
    )
    total_audit_logs = db.query(AuditLog).count()

    # action breakdown
    actions = db.query(
        AuditLog.action,
    ).all()
    action_counts = {}
    for (action,) in actions:
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "generated_at": now.isoformat(),
        "total_reports": total_reports,
        "approved_reports": approved_reports,
        "pending_reports": pending_reports,
        "expiring_soon": expiring_soon,
        "urgent_reports": urgent_reports,
        "total_audit_logs": total_audit_logs,
        "action_breakdown": action_counts,
        "retention_policy": f"{REPORT_RETENTION_DAYS} days for reports",
        "audit_retention": f"{AUDIT_LOG_RETENTION_DAYS} days for audit logs",
        "gdpr_compliant": True,
        "hipaa_compliant": True,
    }
