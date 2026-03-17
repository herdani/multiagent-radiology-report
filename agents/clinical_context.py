"""
Clinical Context Agent
-----------------------
Retrieves relevant medical context from two sources:
  1. Qdrant vector DB  — medical literature (always)
  2. MCP server        — prior patient reports (when available)

Clinical note from radiologist improves RAG retrieval quality.
"""
import asyncio
import logging
import os
from dataclasses import dataclass, field

from agents.image_analysis import ImageFindings

logger = logging.getLogger(__name__)


@dataclass
class ClinicalContext:
    anonymized_id:          str
    relevant_conditions:    list[str]
    differential_diagnosis: list[str]
    recommended_followup:   list[str]
    urgency_level:          str
    context_sources:        list[str]
    prior_reports_summary:  str = ""
    clinical_note:          str = ""


FALLBACK_KNOWLEDGE = {
    "consolidation":    {"conditions": ["Pneumonia", "Pulmonary edema"],  "differential": ["Bacterial pneumonia", "Viral pneumonia"], "followup": ["Repeat CXR in 6-8 weeks"], "urgency": "urgent"},
    "pleural effusion": {"conditions": ["Heart failure", "Malignancy"],   "differential": ["Transudative", "Exudative"],              "followup": ["Echocardiogram"],           "urgency": "urgent"},
    "pneumothorax":     {"conditions": ["Spontaneous pneumothorax"],      "differential": ["Primary", "Secondary"],                   "followup": ["Immediate assessment"],     "urgency": "emergent"},
    "normal":           {"conditions": ["No acute disease"],              "differential": ["Normal variant"],                         "followup": ["Routine follow-up"],        "urgency": "routine"},
}


def _build_query(image_findings: ImageFindings, clinical_note: str = "") -> str:
    """Build search query from findings + clinical note."""
    parts = []
    if clinical_note:
        parts.append(clinical_note)
    if image_findings.impression:
        parts.append(image_findings.impression)
    if image_findings.findings:
        parts.extend(image_findings.findings[:3])
    if image_findings.modality:
        parts.append(f"{image_findings.modality} imaging")
    return " ".join(parts)


# ── MCP client ────────────────────────────────────────────────────────────────

async def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool on the radiology MCP server via stdio."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command="python",
            args=["/home/moez/projects/radiology-ai/mcp_server/radiology_mcp.py"],
            env={
                "DATABASE_URL": os.environ.get(
                    "DATABASE_URL",
                    "sqlite:///./data/radiology.db"
                )
            },
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text if result.content else ""

    except ImportError:
        logger.warning("mcp package not installed — pip install mcp")
        return ""
    except Exception as e:
        logger.warning("MCP tool call failed: %s", e)
        return ""


def _get_prior_reports_via_mcp(anonymized_id: str) -> str:
    """
    Query prior reports via MCP server.
    MCP server handles all DB access — no direct SQLAlchemy in agents.
    Falls back gracefully if MCP server is unavailable.
    """
    try:
        result = asyncio.run(
            _call_mcp_tool(
                "get_prior_reports",
                {"anonymized_id": anonymized_id, "limit": 3},
            )
        )
        if result and "No prior reports" not in result:
            logger.info(
                "MCP prior reports retrieved | anon_id=%s",
                anonymized_id,
            )
            return result
        return ""
    except RuntimeError:
        # asyncio.run() fails if there's already an event loop running
        # this happens when called from async context — use direct DB instead
        logger.warning("asyncio.run() conflict — falling back to direct DB query")
        return _get_prior_reports_direct(anonymized_id)
    except Exception as e:
        logger.warning("MCP call failed (%s) — no prior reports", e)
        return ""


def _get_prior_reports_direct(anonymized_id: str) -> str:
    """
    Direct DB fallback when MCP async context conflicts.
    Used when called from within an existing event loop.
    """
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/radiology.db")
        connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        engine  = create_engine(DATABASE_URL, connect_args=connect_args)
        Session = sessionmaker(bind=engine)
        db      = Session()

        from api.models.report import Report
        reports = db.query(Report).filter(
            Report.anonymized_id == anonymized_id,
            Report.human_approved == True,
        ).order_by(Report.created_at.desc()).limit(3).all()
        db.close()

        if not reports:
            return ""

        summary = f"PRIOR RADIOLOGY REPORTS ({len(reports)} found):\n\n"
        for i, r in enumerate(reports, 1):
            summary += f"Report {i} — {r.created_at.strftime('%Y-%m-%d')} [{r.modality}]\n"
            if r.impression:
                summary += f"  Impression: {r.impression[:200]}\n"
            if r.findings:
                summary += f"  Findings: {r.findings[:200]}\n"
            summary += f"  Urgency: {r.urgency_level}\n\n"

        logger.info("Direct DB prior reports | anon_id=%s | found=%d", anonymized_id, len(reports))
        return summary

    except Exception as e:
        logger.warning("Direct DB query failed: %s", e)
        return ""


# ── Qdrant RAG ────────────────────────────────────────────────────────────────

def _qdrant_context(
    image_findings: ImageFindings,
    clinical_note: str = "",
) -> ClinicalContext:
    """Real RAG via Qdrant + prior reports via MCP."""
    from qdrant_client import QdrantClient

    qdrant_url      = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection_name = "medical_literature"

    client   = QdrantClient(url=qdrant_url)
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        raise RuntimeError(
            "Qdrant collection not found — run: python mlops/ingest_medical_knowledge.py"
        )

    query = _build_query(image_findings, clinical_note)
    logger.info(
        "Mode: QDRANT RAG | anon_id=%s | query='%s...'",
        image_findings.anonymized_id, query[:60],
    )

    results = client.query(
        collection_name=collection_name,
        query_text=query,
        limit=3,
    )

    if not results:
        raise RuntimeError("No Qdrant results returned")

    all_conditions   = []
    all_differential = []
    all_followup     = []
    sources          = []
    urgency          = "routine"
    urgency_rank     = {"routine": 0, "urgent": 1, "emergent": 2}

    for hit in results:
        payload     = hit.metadata if hasattr(hit, "metadata") else {}
        conditions  = payload.get("conditions", [])
        followup    = payload.get("followup", [])
        finding     = payload.get("finding", "unknown")
        hit_urgency = payload.get("urgency", "routine")
        source_id   = payload.get("id", "unknown")

        all_conditions.extend(conditions)
        all_followup.extend(followup)
        sources.append(f"qdrant:{source_id}:{finding}")

        if urgency_rank.get(hit_urgency, 0) > urgency_rank.get(urgency, 0):
            urgency = hit_urgency

        if not all_differential:
            all_differential = conditions[:2]

    # get prior reports via MCP
    prior_summary = _get_prior_reports_via_mcp(image_findings.anonymized_id)
    if prior_summary:
        sources.append("mcp:postgresql:prior_reports")

    return ClinicalContext(
        anonymized_id=image_findings.anonymized_id,
        relevant_conditions=list(dict.fromkeys(all_conditions))[:5],
        differential_diagnosis=list(dict.fromkeys(all_differential))[:3],
        recommended_followup=list(dict.fromkeys(all_followup))[:4],
        urgency_level=urgency,
        context_sources=sources,
        prior_reports_summary=prior_summary,
        clinical_note=clinical_note,
    )


def _mock_context(
    image_findings: ImageFindings,
    clinical_note: str = "",
) -> ClinicalContext:
    logger.info("Mode: MOCK clinical context | anon_id=%s", image_findings.anonymized_id)

    combined = " ".join(image_findings.findings + [image_findings.impression]).lower()
    match    = FALLBACK_KNOWLEDGE["normal"]

    for keyword, data in FALLBACK_KNOWLEDGE.items():
        if keyword in combined:
            match = data
            break

    return ClinicalContext(
        anonymized_id=image_findings.anonymized_id,
        relevant_conditions=match["conditions"],
        differential_diagnosis=match["differential"],
        recommended_followup=match["followup"],
        urgency_level=match["urgency"],
        context_sources=["fallback_knowledge_base"],
        clinical_note=clinical_note,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    image_findings: ImageFindings,
    clinical_note: str = "",
) -> ClinicalContext:
    """
    Run clinical context retrieval.
    Tries Qdrant + MCP first, falls back to mock if unavailable.
    """
    try:
        return _qdrant_context(image_findings, clinical_note)
    except Exception as e:
        logger.warning("Qdrant RAG failed (%s) — using fallback", e)
        return _mock_context(image_findings, clinical_note)
