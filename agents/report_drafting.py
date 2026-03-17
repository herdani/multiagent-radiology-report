import httpx
"""
Report Drafting Agent
----------------------
Takes image findings + clinical context and generates
a structured radiology report in standard format.
"""
import logging
import os
from dataclasses import dataclass

from agents.image_analysis import ImageFindings
from agents.clinical_context import ClinicalContext

logger = logging.getLogger(__name__)


@dataclass
class RadiologyReport:
    anonymized_id: str
    modality: str
    clinical_indication: str
    technique: str
    findings: str
    impression: str
    recommendations: str
    urgency_level: str
    report_text: str
    draft_version: int = 1


REPORT_PROMPT = """You are an expert radiologist. Generate a formal radiology report using this exact structure:

CLINICAL INDICATION:
[brief reason for exam]

TECHNIQUE:
[imaging technique used]

FINDINGS:
[detailed findings paragraph]

IMPRESSION:
[numbered list of conclusions]

RECOMMENDATIONS:
[follow-up recommendations]

Use professional radiological language. Be concise and precise.
Do not invent findings not supported by the image analysis."""


def _parse_report_sections(raw: str) -> dict:
    """
    Parse raw LLM report text into sections.
    Handles multiline sections correctly.
    """
    sections = {
        "clinical_indication": "",
        "technique": "",
        "findings": "",
        "impression": "",
        "recommendations": "",
    }

    # map header keywords to section keys
    headers = {
        "CLINICAL INDICATION": "clinical_indication",
        "TECHNIQUE": "technique",
        "FINDINGS": "findings",
        "IMPRESSION": "impression",
        "RECOMMENDATIONS": "recommendations",
    }

    current_key = None
    buffer = []

    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper().rstrip(":")

        # check if this line is a section header
        matched_header = None
        for header, key in headers.items():
            if upper == header or stripped.upper().startswith(header + ":"):
                matched_header = key
                break

        if matched_header:
            # save previous section
            if current_key and buffer:
                sections[current_key] = "\n".join(buffer).strip()
                buffer = []
            current_key = matched_header
            # handle inline content after the header (e.g. "TECHNIQUE: MRI...")
            inline = stripped.split(":", 1)
            if len(inline) > 1 and inline[1].strip():
                buffer.append(inline[1].strip())
        elif current_key and stripped:
            buffer.append(stripped)

    # save last section
    if current_key and buffer:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def _format_mock_report(
    image_findings: ImageFindings,
    context: ClinicalContext,
) -> RadiologyReport:
    logger.info("Mode: MOCK report drafting | anon_id=%s", image_findings.anonymized_id)

    clinical_indication = f"Routine {image_findings.modality} examination."
    technique = f"Standard {image_findings.modality} acquisition without contrast."
    findings = "\n".join(f"- {f}" for f in image_findings.findings)
    impression = f"1. {image_findings.impression}\n2. No acute findings identified."
    recommendations = "\n".join(f"- {r}" for r in context.recommended_followup)

    report_text = f"""CLINICAL INDICATION:
{clinical_indication}

TECHNIQUE:
{technique}

FINDINGS:
{findings}

IMPRESSION:
{impression}

RECOMMENDATIONS:
{recommendations}"""

    return RadiologyReport(
        anonymized_id=image_findings.anonymized_id,
        modality=image_findings.modality,
        clinical_indication=clinical_indication,
        technique=technique,
        findings=findings,
        impression=impression,
        recommendations=recommendations,
        urgency_level=context.urgency_level,
        report_text=report_text,
    )


def _llm_report(
    image_findings: ImageFindings,
    context: ClinicalContext,
) -> RadiologyReport:
    from openai import OpenAI

    if os.environ.get("GROQ_API_KEY"):
        import httpx
        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        model = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    elif os.environ.get("OLLAMA_MODEL"):
            import httpx
            client = OpenAI(
                api_key="ollama",
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                timeout=httpx.Timeout(300.0, connect=60.0),
            )
            model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    else:
            import httpx
            client = OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            model = os.environ.get("LLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")

    user_prompt = f"""Generate a radiology report based on:

MODALITY: {image_findings.modality}

IMAGE FINDINGS:
{chr(10).join(f"- {f}" for f in image_findings.findings)}

IMPRESSION FROM IMAGE ANALYSIS:
{image_findings.impression}

RELEVANT CONDITIONS: {", ".join(context.relevant_conditions)}
DIFFERENTIAL DIAGNOSIS: {", ".join(context.differential_diagnosis)}
URGENCY: {context.urgency_level}

Follow the exact report structure specified."""

    logger.info("Mode: LLM report drafting | model=%s", model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REPORT_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
    )

    message = response.choices[0].message
    raw = message.content or ""
    if not raw.strip():
        reasoning = getattr(message, "reasoning", "") or ""
        if "CLINICAL INDICATION:" in reasoning:
            idx = reasoning.rfind("CLINICAL INDICATION:")
            raw = reasoning[idx:]
        else:
            raw = reasoning

    sections = _parse_report_sections(raw)

    # fallback — if parser missed sections, use mock
    empty_sections = [k for k, v in sections.items() if not v.strip()]
    if empty_sections:
        logger.warning("LLM report missing sections %s — using mock fallback", empty_sections)
        return _format_mock_report(image_findings, context)

    return RadiologyReport(
        anonymized_id=image_findings.anonymized_id,
        modality=image_findings.modality,
        clinical_indication=sections["clinical_indication"],
        technique=sections["technique"],
        findings=sections["findings"],
        impression=sections["impression"],
        recommendations=sections["recommendations"],
        urgency_level=context.urgency_level,
        report_text=raw.strip(),
    )


def run(
    image_findings: ImageFindings,
    context: ClinicalContext,
    use_mock: bool = False,
) -> RadiologyReport:
    if use_mock:
        return _format_mock_report(image_findings, context)
    try:
        return _llm_report(image_findings, context)
    except Exception as e:
        logger.warning("LLM report drafting failed (%s), using mock", e)
        return _format_mock_report(image_findings, context)
