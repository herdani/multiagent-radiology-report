"""
Clinical Context Agent
-----------------------
Takes image findings and retrieves relevant medical context.
Supports two modes:
  - mock: hardcoded knowledge base (no Qdrant needed)
  - live: queries Qdrant vector DB with real medical literature
"""
import logging
import os
from dataclasses import dataclass, field

from agents.image_analysis import ImageFindings

logger = logging.getLogger(__name__)


@dataclass
class ClinicalContext:
    anonymized_id: str
    relevant_conditions: list[str]
    differential_diagnosis: list[str]
    recommended_followup: list[str]
    urgency_level: str            # routine / urgent / emergent
    context_sources: list[str]    # for audit trail


# Minimal hardcoded knowledge base — replaced by Qdrant in production
KNOWLEDGE_BASE = {
    "consolidation": {
        "conditions": ["Pneumonia", "Pulmonary edema", "Lung contusion", "Atelectasis"],
        "differential": ["Bacterial pneumonia", "Viral pneumonia", "Aspiration pneumonia"],
        "followup": ["Clinical correlation recommended", "Repeat CXR in 4-6 weeks", "Consider sputum culture"],
        "urgency": "urgent",
    },
    "pleural effusion": {
        "conditions": ["Heart failure", "Parapneumonic effusion", "Malignancy", "Pulmonary embolism"],
        "differential": ["Transudative vs exudative effusion", "Empyema"],
        "followup": ["Lateral decubitus view", "Consider thoracentesis if large", "Echocardiogram"],
        "urgency": "urgent",
    },
    "pneumothorax": {
        "conditions": ["Spontaneous pneumothorax", "Traumatic pneumothorax", "Tension pneumothorax"],
        "differential": ["Primary vs secondary spontaneous pneumothorax"],
        "followup": ["Immediate clinical assessment", "Expiratory view", "Consider chest tube"],
        "urgency": "emergent",
    },
    "cardiomegaly": {
        "conditions": ["Congestive heart failure", "Cardiomyopathy", "Pericardial effusion"],
        "differential": ["Dilated cardiomyopathy", "Hypertensive heart disease"],
        "followup": ["Echocardiogram recommended", "BNP/NT-proBNP", "Cardiology referral"],
        "urgency": "urgent",
    },
    "normal": {
        "conditions": ["No acute cardiopulmonary disease"],
        "differential": ["Normal variant"],
        "followup": ["Routine follow-up as clinically indicated"],
        "urgency": "routine",
    },
}


def _match_findings_to_knowledge(findings: list[str], impression: str) -> dict:
    """Match findings text against knowledge base keywords."""
    combined = " ".join(findings + [impression]).lower()

    for keyword, data in KNOWLEDGE_BASE.items():
        if keyword in combined:
            return data

    # default to normal if no keywords matched
    return KNOWLEDGE_BASE["normal"]


def _mock_context(image_findings: ImageFindings) -> ClinicalContext:
    logger.info("Mode: MOCK clinical context | anon_id=%s", image_findings.anonymized_id)

    match = _match_findings_to_knowledge(
        image_findings.findings,
        image_findings.impression,
    )

    return ClinicalContext(
        anonymized_id=image_findings.anonymized_id,
        relevant_conditions=match["conditions"],
        differential_diagnosis=match["differential"],
        recommended_followup=match["followup"],
        urgency_level=match["urgency"],
        context_sources=["internal_knowledge_base_v1"],
    )


def _qdrant_context(image_findings: ImageFindings) -> ClinicalContext:
    """Live RAG via Qdrant — used in production."""
    from qdrant_client import QdrantClient
    from openai import OpenAI

    qdrant = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    # embed the findings text
    llm_client = OpenAI(
        api_key="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )

    query = f"{image_findings.impression}. {' '.join(image_findings.findings)}"

    # search vector DB
    results = qdrant.search(
        collection_name="medical_literature",
        query_text=query,
        limit=5,
    )

    # summarize retrieved context with LLM
    context_texts = [r.payload.get("text", "") for r in results]
    sources = [r.payload.get("source", "unknown") for r in results]

    prompt = f"""Based on these radiology findings:
{query}

And this retrieved medical literature:
{chr(10).join(context_texts)}

Provide:
CONDITIONS: [comma separated list]
DIFFERENTIAL: [comma separated list]
FOLLOWUP: [comma separated list]
URGENCY: [routine/urgent/emergent]"""

    response = llm_client.chat.completions.create(
        model=os.environ.get("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )

    raw = response.choices[0].message.content or ""

    # parse response
    conditions, differential, followup, urgency = [], [], [], "routine"
    for line in raw.splitlines():
        if line.startswith("CONDITIONS:"):
            conditions = [c.strip() for c in line.replace("CONDITIONS:", "").split(",")]
        elif line.startswith("DIFFERENTIAL:"):
            differential = [d.strip() for d in line.replace("DIFFERENTIAL:", "").split(",")]
        elif line.startswith("FOLLOWUP:"):
            followup = [f.strip() for f in line.replace("FOLLOWUP:", "").split(",")]
        elif line.startswith("URGENCY:"):
            urgency = line.replace("URGENCY:", "").strip().lower()

    return ClinicalContext(
        anonymized_id=image_findings.anonymized_id,
        relevant_conditions=conditions,
        differential_diagnosis=differential,
        recommended_followup=followup,
        urgency_level=urgency,
        context_sources=sources,
    )


def run(image_findings: ImageFindings) -> ClinicalContext:
    """
    Mode selection:
      QDRANT_URL set and reachable -> live RAG
      otherwise                    -> mock knowledge base
    """
    qdrant_url = os.environ.get("QDRANT_URL", "")
    use_qdrant = bool(qdrant_url) and qdrant_url != "http://localhost:6333"

    if use_qdrant:
        try:
            return _qdrant_context(image_findings)
        except Exception as e:
            logger.warning("Qdrant failed (%s), falling back to mock", e)

    return _mock_context(image_findings)
