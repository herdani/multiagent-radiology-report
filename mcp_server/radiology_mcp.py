"""
Radiology MCP Server
---------------------
Exposes the PostgreSQL reports database as MCP tools.
Agents and Claude can query prior reports naturally.

Run standalone:
  python mcp_server/radiology_mcp.py

Or via Claude Desktop config:
  {
    "mcpServers": {
      "radiology": {
        "command": "python",
        "args": ["/path/to/mcp_server/radiology_mcp.py"]
      }
    }
  }
"""

import asyncio
import logging
import os
import sys

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import Tool, TextContent  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("radiology-reports")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        Tool(
            name="get_prior_reports",
            description=(
                "Retrieve prior radiology reports for a patient by their anonymized ID. "
                "Returns the most recent reports including findings, impression, and recommendations. "
                "Use this to understand a patient's radiological history before generating a new report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "anonymized_id": {
                        "type": "string",
                        "description": "The anonymized patient ID (hashed, no real PII)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of prior reports to retrieve (default 3)",
                        "default": 3,
                    },
                },
                "required": ["anonymized_id"],
            },
        ),
        Tool(
            name="get_report_by_id",
            description="Retrieve a specific radiology report by its report ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_id": {
                        "type": "string",
                        "description": "The UUID of the report",
                    },
                },
                "required": ["report_id"],
            },
        ),
        Tool(
            name="search_reports",
            description=(
                "Search radiology reports by finding keywords or clinical terms. "
                "Use this to find reports mentioning specific conditions like "
                "'pneumonia', 'nodule', 'effusion', etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term or clinical finding to search for",
                    },
                    "modality": {
                        "type": "string",
                        "description": "Filter by modality (CR, MR, CT, etc.) — optional",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_patient_summary",
            description=(
                "Get a summary of a patient's complete radiological history. "
                "Returns count of studies, modalities used, urgency levels, "
                "and a chronological summary of findings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "anonymized_id": {
                        "type": "string",
                        "description": "The anonymized patient ID",
                    },
                },
                "required": ["anonymized_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from MCP clients."""

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/radiology.db")
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        if name == "get_prior_reports":
            return await _get_prior_reports(db, **arguments)
        elif name == "get_report_by_id":
            return await _get_report_by_id(db, **arguments)
        elif name == "search_reports":
            return await _search_reports(db, **arguments)
        elif name == "get_patient_summary":
            return await _get_patient_summary(db, **arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    finally:
        db.close()


async def _get_prior_reports(db, anonymized_id: str, limit: int = 3) -> list[TextContent]:
    """Fetch prior reports for a patient."""
    from api.models.report import Report

    reports = (
        db.query(Report)
        .filter(Report.anonymized_id == anonymized_id)
        .order_by(Report.created_at.desc())
        .limit(limit)
        .all()
    )

    if not reports:
        return [
            TextContent(type="text", text=f"No prior reports found for patient ID: {anonymized_id}")
        ]

    result = f"Found {len(reports)} prior report(s) for patient {anonymized_id}:\n\n"

    for i, report in enumerate(reports, 1):
        result += f"--- Report {i} ({report.created_at.strftime('%Y-%m-%d')}) ---\n"
        result += f"Modality: {report.modality}\n"
        result += f"Urgency: {report.urgency_level}\n"
        result += f"QA Score: {report.qa_score}\n"
        result += f"Approved: {report.human_approved}\n"
        if report.impression:
            result += f"Impression: {report.impression}\n"
        if report.findings:
            result += (
                f"Findings: {report.findings[:300]}...\n"
                if len(report.findings) > 300
                else f"Findings: {report.findings}\n"
            )
        result += f"Recommendations: {report.recommendations}\n\n"

    return [TextContent(type="text", text=result)]


async def _get_report_by_id(db, report_id: str) -> list[TextContent]:
    """Fetch a specific report by ID."""
    from api.models.report import Report

    report = db.query(Report).filter(Report.id == report_id).first()

    if not report:
        return [TextContent(type="text", text=f"Report not found: {report_id}")]

    result = f"Report ID: {report.id}\n"
    result += f"Date: {report.created_at.strftime('%Y-%m-%d %H:%M')}\n"
    result += f"Modality: {report.modality}\n"
    result += f"Urgency: {report.urgency_level}\n\n"
    result += report.report_text

    return [TextContent(type="text", text=result)]


async def _search_reports(
    db, query: str, modality: str = None, limit: int = 5
) -> list[TextContent]:
    """Search reports by keyword."""
    from api.models.report import Report
    from sqlalchemy import or_

    q = db.query(Report).filter(
        or_(
            Report.report_text.ilike(f"%{query}%"),
            Report.findings.ilike(f"%{query}%"),
            Report.impression.ilike(f"%{query}%"),
        )
    )

    if modality:
        q = q.filter(Report.modality == modality.upper())

    reports = q.order_by(Report.created_at.desc()).limit(limit).all()

    if not reports:
        return [TextContent(type="text", text=f"No reports found matching: '{query}'")]

    result = f"Found {len(reports)} report(s) matching '{query}':\n\n"
    for report in reports:
        result += f"ID: {report.id} | Date: {report.created_at.strftime('%Y-%m-%d')} | "
        result += f"Modality: {report.modality} | Urgency: {report.urgency_level}\n"
        result += f"Impression: {report.impression[:150]}...\n\n" if report.impression else "\n"

    return [TextContent(type="text", text=result)]


async def _get_patient_summary(db, anonymized_id: str) -> list[TextContent]:
    """Get complete patient radiological history summary."""
    from api.models.report import Report

    reports = (
        db.query(Report)
        .filter(Report.anonymized_id == anonymized_id)
        .order_by(Report.created_at.asc())
        .all()
    )

    if not reports:
        return [TextContent(type="text", text=f"No history found for patient: {anonymized_id}")]

    modalities = list(set(r.modality for r in reports))
    urgencies = list(set(r.urgency_level for r in reports))
    approved = sum(1 for r in reports if r.human_approved)
    first_study = reports[0].created_at.strftime("%Y-%m-%d")
    latest_study = reports[-1].created_at.strftime("%Y-%m-%d")

    result = "Patient Radiological History Summary\n"
    result += f"Patient ID: {anonymized_id}\n"
    result += f"Total studies: {len(reports)}\n"
    result += f"Modalities: {', '.join(modalities)}\n"
    result += f"Urgency levels seen: {', '.join(urgencies)}\n"
    result += f"Approved reports: {approved}/{len(reports)}\n"
    result += f"First study: {first_study}\n"
    result += f"Latest study: {latest_study}\n\n"
    result += "Chronological impressions:\n"

    for report in reports:
        result += f"  {report.created_at.strftime('%Y-%m-%d')} [{report.modality}]: "
        result += f"{report.impression[:100]}...\n" if report.impression else "No impression\n"

    return [TextContent(type="text", text=result)]


async def main():
    logger.info("Starting Radiology MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
