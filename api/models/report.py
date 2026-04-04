import uuid
from datetime import datetime, timedelta
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, Integer, ForeignKey
from sqlalchemy.orm import relationship
from api.models.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    anonymized_id = Column(String(16), nullable=False, index=True)
    modality = Column(String(10), nullable=False)
    clinical_indication = Column(Text, default="")
    technique = Column(Text, default="")
    findings = Column(Text, default="")
    impression = Column(Text, default="")
    recommendations = Column(Text, default="")
    report_text = Column(Text, nullable=False)
    qa_score = Column(Float, default=0.0)
    qa_passed = Column(Boolean, default=False)
    urgency_level = Column(String(20), default="routine")
    human_approved = Column(Boolean, default=False)
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    pipeline_version = Column(String(20), default="0.1.0")
    retry_count = Column(Integer, default=0)
    png_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=90))
    audit_logs = relationship("AuditLog", back_populates="report")

    def __repr__(self):
        return f"<Report {self.id} | {self.modality} | {self.urgency_level}>"


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    report_id = Column(String, ForeignKey("reports.id"), nullable=True)
    anonymized_id = Column(String(16), nullable=False)
    action = Column(String(50), nullable=False)
    performed_by = Column(String(100), default="system")
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(200), nullable=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    report = relationship("Report", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.action} | {self.anonymized_id} | {self.timestamp}>"
