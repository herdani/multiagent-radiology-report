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
    report_text: str              # full formatted report
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


def _format_mock_report(
    image_findings: ImageFindings,
    context: ClinicalContext,
) -> RadiologyReport:
    logger.info("Mode: MOCK report drafting | anon_id=%s", image_findings.anonymized_id)

    report_text = f"""CLINICAL INDICATION:
Routine {image_findings.modality} examination.

TECHNIQUE:
Standard {image_findings.modality} acquisition without contrast.

FINDINGS:
{chr(10).join(f"- {f}" for f in image_findings.findings)}

IMPRESSION:
1. {image_findings.impression}
2. No acute findings identified.

RECOMMENDATIONS:
{chr(10).join(f"- {r}" for r in context.recommended_followup)}"""

    return RadiologyReport(
        anonymized_id=image_findings.anonymized_id,
        modality=image_findings.modality,
        clinical_indication=f"Routine {image_findings.modality} examination.",
        technique=f"Standard {image_findings.modality} acquisition without contrast.",
        findings="\n".join(image_findings.findings),
        impression=image_findings.impression,
        recommendations="\n".join(context.recommended_followup),
        urgency_level=context.urgency_level,
        report_text=report_text,
    )


def _llm_report(
    image_findings: ImageFindings,
    context: ClinicalContext,
) -> RadiologyReport:
    from openai import OpenAI

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")

    # use OpenRouter if available
    if os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key") != "your-openrouter-key":
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        model = os.environ.get("LLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    else:
        client = OpenAI(api_key="ollama", base_url=base_url)

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

    # handle thinking models
    message = response.choices[0].message
    raw = message.content or ""
    if not raw.strip():
        reasoning = getattr(message, "reasoning", "") or ""
        if "CLINICAL INDICATION:" in reasoning:
            idx = reasoning.rfind("CLINICAL INDICATION:")
            raw = reasoning[idx:]
        else:
            raw = reasoning

    # parse sections
    sections = {
        "clinical_indication": "",
        "technique": "",
        "findings": "",
        "impression": "",
        "recommendations": "",
    }

    current_section = None
    lines = raw.strip().splitlines()
    for line in lines:
        line_upper = line.strip().upper()
        if "CLINICAL INDICATION:" in line_upper:
            current_section = "clinical_indication"
        elif "TECHNIQUE:" in line_upper:
            current_section = "technique"
        elif "FINDINGS:" in line_upper:
            current_section = "findings"
        elif "IMPRESSION:" in line_upper:
            current_section = "impression"
        elif "RECOMMENDATIONS:" in line_upper:
            current_section = "recommendations"
        elif current_section and line.strip():
            sections[current_section] += line.strip() + "\n"

    return RadiologyReport(
        anonymized_id=image_findings.anonymized_id,
        modality=image_findings.modality,
        clinical_indication=sections["clinical_indication"].strip(),
        technique=sections["technique"].strip(),
        findings=sections["findings"].strip(),
        impression=sections["impression"].strip(),
        recommendations=sections["recommendations"].strip(),
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
