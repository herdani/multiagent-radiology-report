import httpx
import base64
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImageFindings:
    anonymized_id: str
    modality: str
    findings: list[str]
    impression: str
    confidence: float
    flagged: bool
    raw_response: str


SYSTEM_PROMPT = """You are an expert radiologist AI assistant.
Analyze the provided medical image and return your findings in this exact format:

FINDINGS:
- [finding 1]
- [finding 2]

IMPRESSION:
[one sentence summary]

CONFIDENCE: [0.0-1.0]
FLAGGED: [true/false]

Be precise, use standard radiological terminology.
Never invent findings."""


def _encode_image(png_path: str) -> str:
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_response(raw: str, anonymized_id: str, modality: str) -> ImageFindings:
    lines = raw.strip().splitlines()
    findings = []
    impression = ""
    confidence = 0.5
    flagged = False
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("FINDINGS:"):
            section = "findings"
        elif line.startswith("IMPRESSION:"):
            section = "impression"
            rest = line.replace("IMPRESSION:", "").strip()
            if rest:
                impression = rest
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                confidence = 0.5
        elif line.startswith("FLAGGED:"):
            flagged = "true" in line.lower()
        elif section == "findings" and line.startswith("-"):
            findings.append(line.lstrip("- ").strip())
        elif section == "impression" and line and not impression:
            impression = line

    return ImageFindings(
        anonymized_id=anonymized_id,
        modality=modality,
        findings=findings,
        impression=impression,
        confidence=confidence,
        flagged=flagged,
        raw_response=raw,
    )


def _extract_content(response) -> str:
    """
    Handle thinking models like Qwen3.5 where the final answer
    is in content, but may fall back to reasoning if content is empty.
    """
    message = response.choices[0].message
    content = message.content or ""

    if not content.strip():
        # thinking model — extract the answer from reasoning field
        reasoning = getattr(message, "reasoning", "") or ""
        # find the last structured block after all the thinking
        if "FINDINGS:" in reasoning:
            idx = reasoning.rfind("FINDINGS:")
            content = reasoning[idx:]
        else:
            content = reasoning

    return content.strip()


def _mock_analysis(anonymized_id: str, modality: str) -> ImageFindings:
    logger.info("Mode: MOCK")
    raw = """FINDINGS:
- Lungs are clear bilaterally with no focal consolidation
- No pleural effusion or pneumothorax identified
- Cardiac silhouette is within normal limits
- Mediastinum is midline and unremarkable
- Osseous structures show no acute abnormality

IMPRESSION:
No acute cardiopulmonary findings identified on this chest radiograph.

CONFIDENCE: 0.82
FLAGGED: false"""
    return _parse_response(raw, anonymized_id, modality)


def _ollama_analysis(png_path: str, anonymized_id: str, modality: str) -> ImageFindings:
    from openai import OpenAI

    model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    client = OpenAI(api_key="ollama", base_url=base_url, timeout=httpx.Timeout(300.0, connect=60.0))
    image_b64 = _encode_image(png_path)

    logger.info("Mode: OLLAMA | model=%s | anon_id=%s", model, anonymized_id)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Analyze this {modality} scan and provide your findings.",
                    },
                ],
            },
        ],
        max_tokens=4096,  # enough for thinking + response
    )

    raw = _extract_content(response)
    logger.info("Raw response length: %d chars", len(raw))
    return _parse_response(raw, anonymized_id, modality)


def _openrouter_analysis(png_path: str, anonymized_id: str, modality: str) -> ImageFindings:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    model = os.environ.get("LLM_VISION_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    image_b64 = _encode_image(png_path)
    logger.info("Mode: OPENROUTER | model=%s | anon_id=%s", model, anonymized_id)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Analyze this {modality} scan and provide your findings.",
                    },
                ],
            },
        ],
        max_tokens=4096,
    )

    raw = _extract_content(response)
    return _parse_response(raw, anonymized_id, modality)


def _groq_analysis(png_path: str, anonymized_id: str, modality: str) -> ImageFindings:
    """Cloud inference via Groq — fast, free, vision capable."""
    from openai import OpenAI
    import httpx

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        timeout=httpx.Timeout(60.0, connect=10.0),
    )

    model = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    image_b64 = _encode_image(png_path)
    logger.info("Mode: GROQ | model=%s | anon_id=%s", model, anonymized_id)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Analyze this {modality} scan and provide your findings.",
                    },
                ],
            },
        ],
        max_tokens=1024,
    )

    raw = response.choices[0].message.content or ""
    return _parse_response(raw, anonymized_id, modality)


def run(png_path: str, anonymized_id: str, modality: str = "CR") -> ImageFindings:
    """
    Mode selection priority:
      1. GROQ_API_KEY set      → Groq cloud (fast, free, vision)
      2. OLLAMA_MODEL set      → local Ollama
      3. OPENROUTER_API_KEY set → OpenRouter cloud
      4. none                  → mock mode
    """
    if os.environ.get("GROQ_API_KEY"):
        return _groq_analysis(png_path, anonymized_id, modality)
    elif os.environ.get("OLLAMA_MODEL"):
        return _ollama_analysis(png_path, anonymized_id, modality)
    elif os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key") != "your-openrouter-key":
        return _openrouter_analysis(png_path, anonymized_id, modality)
    else:
        return _mock_analysis(anonymized_id, modality)


def run_with_xai(
    png_path: str,
    anonymized_id: str,
    modality: str = "CR",
) -> tuple:
    """
    Run image analysis AND generate XAI heatmap concurrently.
    XAI runs in a background thread — never blocks the main pipeline.
    """
    import threading
    from pipeline.xai import generate_heatmap

    # start XAI in background thread immediately
    xai_result = {}

    def run_xai():
        chest_modalities = ["CR", "DX", "CT"]
        if modality.upper() in chest_modalities:
            result = generate_heatmap(png_path)
            xai_result.update(result)

    xai_thread = threading.Thread(target=run_xai, daemon=True)
    xai_thread.start()

    # run main image analysis (this is the slow Qwen call)
    findings = run(png_path, anonymized_id, modality)

    # wait for XAI to finish (max 60 seconds)
    xai_thread.join(timeout=60)

    # add pathology scores to findings if available
    if xai_result.get("pathology_scores"):
        top_findings = sorted(
            xai_result["pathology_scores"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        score_text = ", ".join(f"{k}: {v:.0%}" for k, v in top_findings if v > 0.1)
        if score_text:
            findings.findings.append(f"TorchXRayVision detection scores: {score_text}")

    return findings, xai_result
