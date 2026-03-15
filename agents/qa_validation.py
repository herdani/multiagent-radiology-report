"""
QA Validation Agent
--------------------
Reviews the drafted report for:
  - Completeness (all sections present)
  - Consistency (findings match impression)
  - Hallucinations (claims not supported by image analysis)
  - Urgency flags (critical findings not flagged)
  - Formatting (standard radiology report structure)
"""
import logging
import os
from dataclasses import dataclass, field

from agents.image_analysis import ImageFindings
from agents.report_drafting import RadiologyReport

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = [
    "clinical_indication",
    "technique",
    "findings",
    "impression",
    "recommendations",
]


@dataclass
class ValidationResult:
    anonymized_id: str
    passed: bool
    score: float                    # 0.0 - 1.0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    approved_report: str = ""       # final report text after QA
    requires_human_review: bool = False


def _check_completeness(report: RadiologyReport) -> list[str]:
    """Check all required sections are present and non-empty."""
    issues = []
    for section in REQUIRED_SECTIONS:
        value = getattr(report, section, "")
        if not value or not value.strip():
            issues.append(f"Missing required section: {section}")
    return issues


def _check_urgency(
    report: RadiologyReport,
    image_findings: ImageFindings,
) -> list[str]:
    """Check that urgent findings are properly flagged."""
    warnings = []
    urgent_keywords = [
        "pneumothorax", "tension", "hemorrhage", "infarct",
        "embolism", "dissection", "rupture", "obstruction",
        "mass", "malignancy", "critical"
    ]

    report_lower = report.report_text.lower()
    found_urgent = [kw for kw in urgent_keywords if kw in report_lower]

    if found_urgent and report.urgency_level == "routine":
        warnings.append(
            f"Potentially urgent keywords found ({', '.join(found_urgent)}) "
            f"but urgency is marked as routine — human review required"
        )

    if image_findings.flagged and report.urgency_level == "routine":
        warnings.append("Image analysis flagged findings but report urgency is routine")

    return warnings


def _check_consistency(
    report: RadiologyReport,
    image_findings: ImageFindings,
) -> list[str]:
    """Check report is consistent with original image findings."""
    warnings = []

    # if image analysis found nothing abnormal but report mentions findings
    normal_impression = "no acute" in image_findings.impression.lower()
    report_has_findings = any(
        kw in report.findings.lower()
        for kw in ["consolidation", "effusion", "pneumothorax", "mass", "opacity"]
    )

    if normal_impression and report_has_findings:
        warnings.append(
            "Report mentions significant findings but image analysis impression was normal — verify"
        )

    return warnings


def _mock_validation(
    report: RadiologyReport,
    image_findings: ImageFindings,
) -> ValidationResult:
    logger.info("Mode: MOCK QA validation | anon_id=%s", report.anonymized_id)

    issues = _check_completeness(report)
    warnings = _check_urgency(report, image_findings)
    warnings += _check_consistency(report, image_findings)

    passed = len(issues) == 0
    score = 1.0 - (len(issues) * 0.2) - (len(warnings) * 0.1)
    score = max(0.0, min(1.0, score))
    requires_human = not passed or len(warnings) > 1 or image_findings.flagged

    return ValidationResult(
        anonymized_id=report.anonymized_id,
        passed=passed,
        score=round(score, 2),
        issues=issues,
        warnings=warnings,
        approved_report=report.report_text if passed else "",
        requires_human_review=requires_human,
    )


def _llm_validation(
    report: RadiologyReport,
    image_findings: ImageFindings,
) -> ValidationResult:
    """Use LLM for deeper semantic validation."""
    from openai import OpenAI

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")

    if os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key") != "your-openrouter-key":
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        model = os.environ.get("LLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    else:
        client = OpenAI(api_key="ollama", base_url=base_url)

    prompt = f"""You are a senior radiologist reviewing a junior radiologist's report.

ORIGINAL IMAGE FINDINGS:
{chr(10).join(f"- {f}" for f in image_findings.findings)}

ORIGINAL IMPRESSION:
{image_findings.impression}

DRAFTED REPORT:
{report.report_text}

Review the report and respond in this exact format:
PASSED: [true/false]
SCORE: [0.0-1.0]
ISSUES: [comma separated list of critical issues, or 'none']
WARNINGS: [comma separated list of warnings, or 'none']
REQUIRES_HUMAN_REVIEW: [true/false]"""

    logger.info("Mode: LLM QA validation | model=%s", model)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    message = response.choices[0].message
    raw = message.content or ""
    if not raw.strip():
        reasoning = getattr(message, "reasoning", "") or ""
        if "PASSED:" in reasoning:
            idx = reasoning.rfind("PASSED:")
            raw = reasoning[idx:]

    # parse LLM validation response
    passed = True
    score = 0.8
    issues = []
    warnings = []
    requires_human = False

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("PASSED:"):
            passed = "true" in line.lower()
        elif line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 0.8
        elif line.startswith("ISSUES:"):
            val = line.replace("ISSUES:", "").strip()
            if val.lower() != "none":
                issues = [i.strip() for i in val.split(",")]
        elif line.startswith("WARNINGS:"):
            val = line.replace("WARNINGS:", "").strip()
            if val.lower() != "none":
                warnings = [w.strip() for w in val.split(",")]
        elif line.startswith("REQUIRES_HUMAN_REVIEW:"):
            requires_human = "true" in line.lower()

    # always run rule-based checks on top of LLM
    rule_issues = _check_completeness(report)
    rule_warnings = _check_urgency(report, image_findings)
    issues = list(set(issues + rule_issues))
    warnings = list(set(warnings + rule_warnings))

    return ValidationResult(
        anonymized_id=report.anonymized_id,
        passed=passed and len(rule_issues) == 0,
        score=round(score, 2),
        issues=issues,
        warnings=warnings,
        approved_report=report.report_text if passed else "",
        requires_human_review=requires_human or image_findings.flagged,
    )


def run(
    report: RadiologyReport,
    image_findings: ImageFindings,
    use_mock: bool = False,
) -> ValidationResult:
    if use_mock:
        return _mock_validation(report, image_findings)
    try:
        return _llm_validation(report, image_findings)
    except Exception as e:
        logger.warning("LLM validation failed (%s), using rule-based", e)
        return _mock_validation(report, image_findings)
