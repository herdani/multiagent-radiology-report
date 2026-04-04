import httpx

"""
QA Validation Agent
--------------------
Reviews the drafted report for completeness, consistency,
urgency flags, and potential hallucinations.
"""
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from agents.image_analysis import ImageFindings  # noqa: E402
from agents.report_drafting import RadiologyReport  # noqa: E402

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = [
    "clinical_indication",
    "technique",
    "findings",
    "impression",
    "recommendations",
]

# only flag if these appear WITHOUT negation nearby
URGENT_KEYWORDS = [
    "pneumothorax",
    "tension",
    "hemorrhage",
    "haemorrhage",
    "infarct",
    "embolism",
    "dissection",
    "rupture",
    "obstruction",
    "malignancy",
    "critical",
]

# negation words that indicate finding is absent
NEGATION_WORDS = [
    "no ",
    "not ",
    "without ",
    "absent ",
    "negative ",
    "clear of",
    "no evidence",
    "unremarkable",
]


@dataclass
class ValidationResult:
    anonymized_id: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    approved_report: str = ""
    requires_human_review: bool = False


def _check_completeness(report: RadiologyReport) -> list[str]:
    """Check all required sections are present and non-empty."""
    issues = []
    for section in REQUIRED_SECTIONS:
        value = getattr(report, section, "")
        if not value or not value.strip():
            issues.append(f"Missing required section: {section}")
    return issues


def _is_negated(keyword: str, text: str) -> bool:
    """Check if a keyword is negated in the surrounding context."""
    idx = text.find(keyword)
    if idx == -1:
        return False
    # look at 50 chars before the keyword for negation
    context = text[max(0, idx - 50) : idx].lower()
    return any(neg in context for neg in NEGATION_WORDS)


def _check_urgency(
    report: RadiologyReport,
    image_findings: ImageFindings,
) -> list[str]:
    """Check urgent findings are properly flagged — with negation awareness."""
    warnings = []
    report_lower = report.report_text.lower()

    # only flag keywords that are NOT negated
    found_urgent = [
        kw for kw in URGENT_KEYWORDS if kw in report_lower and not _is_negated(kw, report_lower)
    ]

    if found_urgent and report.urgency_level == "routine":
        warnings.append(
            f"Urgent findings detected ({', '.join(found_urgent)}) "
            f"but urgency marked as routine — human review required"
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
    normal_impression = "no acute" in image_findings.impression.lower()
    report_has_findings = any(
        kw in report.findings.lower() and not _is_negated(kw, report.findings.lower())
        for kw in ["consolidation", "effusion", "pneumothorax", "malignancy", "opacity"]
    )
    if normal_impression and report_has_findings:
        warnings.append(
            "Report mentions significant findings but image impression was normal — verify"
        )
    return warnings


def _clean_llm_list(raw: str) -> list[str]:
    """
    Clean LLM output that may contain template artifacts.
    Removes things like '[comma separated list...]', 'none', empty strings.
    """
    if not raw:
        return []

    # remove anything inside square brackets (template instructions)
    raw = re.sub(r"\[.*?\]", "", raw)

    items = [i.strip() for i in raw.split(",")]
    cleaned = []
    for item in items:
        item = item.strip().strip('"').strip("'")
        # skip empty, 'none', or template artifacts
        if item and item.lower() not in ("none", "n/a", "") and len(item) > 3:
            cleaned.append(item)
    return cleaned


def _mock_validation(
    report: RadiologyReport,
    image_findings: ImageFindings,
) -> ValidationResult:
    logger.info("Mode: MOCK QA validation | anon_id=%s", report.anonymized_id)

    issues = _check_completeness(report)
    warnings = _check_urgency(report, image_findings)
    warnings += _check_consistency(report, image_findings)

    passed = len(issues) == 0
    score = max(0.0, min(1.0, 1.0 - len(issues) * 0.2 - len(warnings) * 0.1))
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
    """LLM-based semantic validation."""
    from openai import OpenAI

    # priority: OpenRouter Claude → Groq → Ollama
    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        model = os.environ.get("QA_MODEL", "anthropic/claude-sonnet-4-6")
        logger.info("Mode: Claude Sonnet via OpenRouter | model=%s", model)
    elif os.environ.get("GROQ_API_KEY"):
        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        model = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        logger.info("Mode: Groq | model=%s", model)
    else:
        client = OpenAI(
            api_key="ollama",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            timeout=httpx.Timeout(300.0, connect=60.0),
        )
        model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")
        logger.info("Mode: Ollama | model=%s", model)

    prompt = f"""You are a senior radiologist reviewing a junior radiologist report.

ORIGINAL IMAGE FINDINGS:
{chr(10).join(f"- {f}" for f in image_findings.findings)}

ORIGINAL IMPRESSION:
{image_findings.impression}

DRAFTED REPORT:
{report.report_text}

Review the report carefully and respond ONLY with these exact lines, no other text:
PASSED: true
SCORE: 0.85
ISSUES: none
WARNINGS: none
REQUIRES_HUMAN_REVIEW: false

Replace the values with your actual assessment. For ISSUES and WARNINGS write specific
clinical concerns as plain text, or write 'none' if there are none."""

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

    # parse response
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
                score = float(re.search(r"[\d.]+", line.replace("SCORE:", "")).group())
            except Exception:
                score = 0.8
        elif line.startswith("ISSUES:"):
            issues = _clean_llm_list(line.replace("ISSUES:", "").strip())
        elif line.startswith("WARNINGS:"):
            warnings = _clean_llm_list(line.replace("WARNINGS:", "").strip())
        elif line.startswith("REQUIRES_HUMAN_REVIEW:"):
            requires_human = "true" in line.lower()

    # always run rule-based checks on top
    rule_issues = _check_completeness(report)
    rule_warnings = _check_urgency(report, image_findings)
    rule_warnings += _check_consistency(report, image_findings)

    issues = list(dict.fromkeys(issues + rule_issues))
    warnings = list(dict.fromkeys(warnings + rule_warnings))

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
