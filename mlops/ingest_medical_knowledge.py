"""
Medical Knowledge Ingestion
-----------------------------
Loads medical radiology knowledge into Qdrant vector database.
Uses fastembed for local encoding — no API key needed.
Run once to populate Qdrant:
  python mlops/ingest_medical_knowledge.py
"""

import logging
import os
import hashlib
from dotenv import load_dotenv

load_dotenv("/home/moez/projects/radiology-ai/.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEDICAL_KNOWLEDGE = [
    # ── Chest / Pulmonary ─────────────────────────────────────────────────────
    {
        "id": "ck001",
        "category": "pulmonary",
        "finding": "consolidation",
        "text": "Pulmonary consolidation refers to replacement of air in the lung parenchyma by fluid, cells, or other material. Common causes include bacterial pneumonia, pulmonary edema, pulmonary hemorrhage, and lung contusion. On chest radiograph, consolidation appears as increased opacity with air bronchograms. Lobar consolidation suggests bacterial pneumonia. Differential diagnosis includes community-acquired pneumonia, hospital-acquired pneumonia, aspiration pneumonia, pulmonary edema, lung cancer. Recommended workup: clinical correlation, sputum culture, CBC, CRP. Follow-up chest radiograph in 6-8 weeks to confirm resolution.",
        "urgency": "urgent",
        "conditions": ["Bacterial pneumonia", "Pulmonary edema", "Lung contusion"],
        "followup": ["Sputum culture", "CBC and CRP", "Repeat CXR in 6-8 weeks"],
    },
    {
        "id": "ck002",
        "category": "pulmonary",
        "finding": "pleural effusion",
        "text": "Pleural effusion is accumulation of fluid in the pleural space. On upright chest radiograph, blunting of the costophrenic angle indicates at least 200-300ml of fluid. Causes include heart failure (most common), parapneumonic effusion, malignancy, pulmonary embolism, and hepatic cirrhosis. Transudative causes: heart failure, hepatic cirrhosis, nephrotic syndrome. Exudative causes: pneumonia, malignancy, pulmonary embolism, tuberculosis. Recommended workup: lateral decubitus radiograph, thoracentesis if large or symptomatic, echocardiogram if heart failure suspected.",
        "urgency": "urgent",
        "conditions": [
            "Heart failure",
            "Parapneumonic effusion",
            "Malignancy",
            "Pulmonary embolism",
        ],
        "followup": ["Lateral decubitus view", "Thoracentesis if large", "Echocardiogram"],
    },
    {
        "id": "ck003",
        "category": "pulmonary",
        "finding": "pneumothorax",
        "text": "Pneumothorax is presence of air in the pleural space causing lung collapse. On chest radiograph, visible pleural line with absent lung markings peripheral to the line. Tension pneumothorax is a medical emergency: tracheal deviation away from affected side, mediastinal shift, cardiovascular compromise. Primary spontaneous pneumothorax: tall thin young males, no underlying disease. Secondary spontaneous: underlying lung disease such as COPD. Traumatic: rib fractures, penetrating chest trauma. Management: small may be observed, large requires aspiration or chest tube.",
        "urgency": "emergent",
        "conditions": [
            "Spontaneous pneumothorax",
            "Tension pneumothorax",
            "Traumatic pneumothorax",
        ],
        "followup": ["Immediate clinical assessment", "Expiratory view", "Consider chest tube"],
    },
    {
        "id": "ck004",
        "category": "cardiac",
        "finding": "cardiomegaly",
        "text": "Cardiomegaly defined as cardiothoracic ratio greater than 0.5 on PA chest radiograph. Common causes: dilated cardiomyopathy, hypertensive heart disease, valvular disease, congenital heart disease, pericardial effusion. Associated findings: pulmonary vascular redistribution, Kerley B lines, pleural effusions suggest congestive heart failure. Recommended workup: echocardiogram, BNP/NT-proBNP, ECG, cardiology referral. Pericardial effusion can mimic cardiomegaly.",
        "urgency": "urgent",
        "conditions": [
            "Congestive heart failure",
            "Dilated cardiomyopathy",
            "Pericardial effusion",
        ],
        "followup": ["Echocardiogram", "BNP/NT-proBNP", "Cardiology referral"],
    },
    {
        "id": "ck005",
        "category": "pulmonary",
        "finding": "pulmonary nodule",
        "text": "Pulmonary nodule is a rounded opacity less than 3cm in diameter. Lesions greater than 3cm are termed masses and are more likely malignant. Fleischner Society guidelines govern follow-up based on size and risk factors. Low risk nodules less than 6mm: no routine follow-up needed. Intermediate nodules 6-8mm: CT follow-up at 6-12 months. High risk nodules greater than 8mm: consider PET-CT or biopsy. Benign features: calcification, stable for 2 years. Malignant features: spiculated margins, upper lobe location, growth on follow-up.",
        "urgency": "urgent",
        "conditions": ["Lung cancer", "Metastatic disease", "Granuloma", "Hamartoma"],
        "followup": [
            "CT chest follow-up per Fleischner guidelines",
            "PET-CT if >8mm",
            "Pulmonology referral",
        ],
    },
    {
        "id": "ck006",
        "category": "pulmonary",
        "finding": "interstitial lung disease",
        "text": "Interstitial lung disease encompasses heterogeneous disorders affecting lung parenchyma. Radiographic patterns include reticular pattern (fine network of lines), ground glass opacity, and honeycombing (clustered cystic airspaces indicating fibrosis). Common causes: idiopathic pulmonary fibrosis, sarcoidosis, hypersensitivity pneumonitis, connective tissue disease-associated ILD. HRCT chest is the gold standard for ILD evaluation. Multidisciplinary discussion recommended.",
        "urgency": "urgent",
        "conditions": [
            "Idiopathic pulmonary fibrosis",
            "Sarcoidosis",
            "Hypersensitivity pneumonitis",
        ],
        "followup": ["HRCT chest", "Pulmonology referral", "Multidisciplinary discussion"],
    },
    {
        "id": "ck007",
        "category": "spine",
        "finding": "disc herniation",
        "text": "Intervertebral disc herniation occurs when nucleus pulposus extrudes through annulus fibrosus. MRI is gold standard for evaluation. Types: protrusion, extrusion, sequestration. Common levels: L4-L5 and L5-S1 in lumbar spine, C5-C6 and C6-C7 in cervical spine. Clinical correlation essential as many disc herniations are asymptomatic. Central herniation may cause cauda equina syndrome which is a surgical emergency. Management: conservative treatment first, then epidural steroid injection, then surgical decompression.",
        "urgency": "routine",
        "conditions": ["Disc herniation", "Radiculopathy", "Cauda equina syndrome"],
        "followup": ["Neurosurgery referral if symptomatic", "Physiotherapy", "Pain management"],
    },
    {
        "id": "ck008",
        "category": "spine",
        "finding": "spinal cord compression",
        "text": "Spinal cord compression is a neurological emergency requiring urgent evaluation. Causes: disc herniation, epidural abscess, epidural hematoma, metastatic disease, vertebral fracture, primary spinal tumour. MRI spine with and without contrast is investigation of choice. Red flag symptoms: bilateral limb weakness, bowel or bladder dysfunction, saddle anaesthesia require emergency assessment. Metastatic spinal cord compression most common in breast, prostate, lung cancer. Time to treatment is critical for neurological outcome.",
        "urgency": "emergent",
        "conditions": ["Metastatic spinal cord compression", "Epidural abscess", "Disc herniation"],
        "followup": [
            "Urgent neurosurgery referral",
            "MRI with contrast",
            "Dexamethasone if metastatic",
        ],
    },
    {
        "id": "ck009",
        "category": "spine",
        "finding": "vertebral fracture",
        "text": "Vertebral fractures may be traumatic or insufficiency (osteoporotic). Compression fractures most common in thoracolumbar junction T12-L1. Burst fractures involve both anterior and posterior elements and are unstable. MRI distinguishes acute from chronic fractures via bone marrow oedema. Osteoporotic fractures require DEXA scan and bisphosphonate therapy. Pathological fractures require exclusion of underlying malignancy. CT myelogram or MRI if neurological compromise suspected.",
        "urgency": "urgent",
        "conditions": ["Osteoporotic fracture", "Traumatic fracture", "Pathological fracture"],
        "followup": [
            "CT for fracture characterisation",
            "DEXA scan",
            "Spine surgery referral if unstable",
        ],
    },
    {
        "id": "ck010",
        "category": "normal",
        "finding": "normal",
        "text": "No acute cardiopulmonary disease identified. Normal chest radiograph findings: clear lungs bilaterally without consolidation, effusion, or pneumothorax. Normal cardiac silhouette with cardiothoracic ratio less than 0.5. Mediastinum midline and unremarkable. No hilar enlargement. Osseous structures without acute fracture or destructive lesion. Soft tissues unremarkable. Routine clinical follow-up as indicated by clinical presentation.",
        "urgency": "routine",
        "conditions": ["No acute disease"],
        "followup": ["Routine clinical follow-up"],
    },
    {
        "id": "ck011",
        "category": "pulmonary",
        "finding": "atelectasis",
        "text": "Atelectasis refers to collapse or incomplete expansion of lung tissue. Types: compressive (pleural effusion, pneumothorax), obstructive (mucus plug, endobronchial lesion), cicatricial (fibrosis). Radiographic features: increased opacity, volume loss, displacement of fissures, mediastinal shift toward affected side. Subsegmental atelectasis is common post-operative finding, usually benign. Lobar atelectasis requires bronchoscopy to exclude endobronchial obstruction. Post-obstructive atelectasis may indicate underlying malignancy.",
        "urgency": "routine",
        "conditions": ["Post-operative atelectasis", "Mucus plugging", "Endobronchial obstruction"],
        "followup": [
            "Physiotherapy and deep breathing",
            "Bronchoscopy if lobar",
            "CT if persistent",
        ],
    },
    {
        "id": "ck012",
        "category": "vascular",
        "finding": "pulmonary embolism",
        "text": "Pulmonary embolism is occlusion of pulmonary arteries by thrombus. Chest radiograph often normal. CT pulmonary angiography is gold standard investigation. Wells score and D-dimer guide investigation pathway. Massive PE causes haemodynamic compromise and RV strain on ECG. Treatment: anticoagulation with LMWH or DOAC, thrombolysis if massive PE. Risk factors: DVT, immobility, malignancy, pregnancy, thrombophilia.",
        "urgency": "emergent",
        "conditions": ["Pulmonary embolism", "Deep vein thrombosis", "Right heart strain"],
        "followup": ["CTPA urgently", "D-dimer if low probability", "Anticoagulation"],
    },
    {
        "id": "ck013",
        "category": "pulmonary",
        "finding": "pneumonia",
        "text": "Pneumonia is infection of lung parenchyma. Community-acquired pneumonia most common in lower lobes. Radiographic features: lobar or segmental consolidation, air bronchograms, parapneumonic effusion. Atypical pneumonia (Mycoplasma, Chlamydia) shows bilateral interstitial infiltrates. Hospital-acquired pneumonia often multidrug resistant organisms. Aspiration pneumonia typically right lower lobe. Severity assessed by CURB-65 score. Treatment: antibiotics guided by organism and severity.",
        "urgency": "urgent",
        "conditions": [
            "Community-acquired pneumonia",
            "Hospital-acquired pneumonia",
            "Aspiration pneumonia",
        ],
        "followup": [
            "Sputum culture and sensitivity",
            "Blood cultures if severe",
            "Repeat CXR at 6 weeks",
        ],
    },
    {
        "id": "ck014",
        "category": "cardiac",
        "finding": "pulmonary oedema",
        "text": "Pulmonary oedema is abnormal accumulation of fluid in lung interstitium and alveoli. Cardiogenic causes: left heart failure, mitral stenosis. Non-cardiogenic: ARDS, neurogenic, high altitude. Radiographic features: bilateral perihilar haziness (bat-wing pattern), Kerley B lines, cardiomegaly, upper lobe venous diversion, pleural effusions. Treatment: diuretics for cardiogenic, treat underlying cause for non-cardiogenic. Urgent echocardiogram recommended.",
        "urgency": "urgent",
        "conditions": ["Cardiogenic pulmonary oedema", "ARDS", "Left heart failure"],
        "followup": ["Urgent echocardiogram", "BNP measurement", "Cardiology review"],
    },
    {
        "id": "ck015",
        "category": "spine",
        "finding": "prevertebral soft tissue abnormality",
        "text": "Prevertebral soft tissue abnormalities on cervical spine MRI include abscess, haematoma, lymphadenopathy, and mass lesions. Prevertebral abscess may complicate discitis osteomyelitis and requires urgent drainage. Retropharyngeal abscess can cause airway compromise. Haematoma typically post-traumatic. Lymphadenopathy may indicate metastatic disease or lymphoma. MRI with contrast essential for characterisation. T2 hyperintense lesion in prevertebral space raises concern for fluid collection or abscess.",
        "urgency": "urgent",
        "conditions": [
            "Prevertebral abscess",
            "Retropharyngeal abscess",
            "Lymphadenopathy",
            "Haematoma",
        ],
        "followup": [
            "MRI with contrast",
            "Surgical drainage if abscess",
            "ENT or spine surgery referral",
        ],
    },
    {
        "id": "ck016",
        "category": "brain",
        "finding": "intracranial haemorrhage",
        "text": "Intracranial haemorrhage is a neurological emergency. Types: epidural haematoma (biconvex, arterial), subdural haematoma (crescent, venous), subarachnoid haemorrhage (CSF spaces), intracerebral haemorrhage (parenchymal). CT head without contrast is first-line investigation. Subarachnoid haemorrhage: thunderclap headache, LP if CT negative within 6 hours. Epidural haematoma: lucid interval, middle meningeal artery injury. Anticoagulation reversal urgent if on warfarin or DOAC.",
        "urgency": "emergent",
        "conditions": ["Epidural haematoma", "Subdural haematoma", "Subarachnoid haemorrhage"],
        "followup": ["Urgent neurosurgery referral", "CT angiography", "Anticoagulation reversal"],
    },
    {
        "id": "ck017",
        "category": "abdomen",
        "finding": "bowel obstruction",
        "text": "Bowel obstruction may be small or large bowel. Small bowel obstruction: dilated loops greater than 3cm, valvulae conniventes, stepladder pattern. Large bowel obstruction: dilated colon greater than 6cm, haustral folds. Causes: adhesions (most common for small bowel), hernia, malignancy (most common for large bowel), volvulus. CT abdomen with contrast is investigation of choice. Closed loop obstruction and strangulation are surgical emergencies.",
        "urgency": "emergent",
        "conditions": ["Small bowel obstruction", "Large bowel obstruction", "Volvulus"],
        "followup": ["Urgent surgical review", "CT abdomen with contrast", "NG tube decompression"],
    },
    {
        "id": "ck018",
        "category": "musculoskeletal",
        "finding": "bone metastases",
        "text": "Bone metastases are the most common malignant bone tumours. Common primaries: breast, prostate, lung, kidney, thyroid. Lytic lesions: lung, kidney, thyroid, multiple myeloma. Sclerotic lesions: prostate, breast (mixed). Bone scan highly sensitive for detection. MRI best for marrow infiltration and cord compression. PET-CT for staging. Complications: pathological fracture, spinal cord compression, hypercalcaemia. Bisphosphonates or denosumab for bone protection.",
        "urgency": "urgent",
        "conditions": ["Bone metastases", "Pathological fracture risk", "Hypercalcaemia"],
        "followup": ["Bone scan or PET-CT", "Oncology referral", "Bisphosphonate therapy"],
    },
    {
        "id": "ck019",
        "category": "pulmonary",
        "finding": "lung mass",
        "text": "Lung mass is pulmonary opacity greater than 3cm diameter. High suspicion for primary lung malignancy especially in smokers over 40. Non-small cell lung cancer accounts for 85 percent of lung cancers. CT chest with contrast for characterisation. PET-CT for staging. CT-guided biopsy for tissue diagnosis. Staging determines resectability. Early stage (I-II): surgical resection curative intent. Advanced stage (III-IV): chemotherapy, immunotherapy, targeted therapy. Urgent respiratory medicine or thoracic surgery referral.",
        "urgency": "urgent",
        "conditions": ["Primary lung cancer", "Metastatic lung disease", "Lymphoma"],
        "followup": [
            "CT chest with contrast",
            "PET-CT staging",
            "Thoracic surgery or oncology referral",
        ],
    },
    {
        "id": "ck020",
        "category": "vascular",
        "finding": "aortic aneurysm",
        "text": "Aortic aneurysm defined as dilatation greater than 1.5 times normal diameter. Abdominal aortic aneurysm (AAA) normal diameter less than 3cm. Thoracic aortic aneurysm involves ascending or descending thoracic aorta. Rupture risk increases markedly above 5.5cm for AAA. CT angiography for planning repair. Aortic dissection: acute tearing chest or back pain, pulse deficit — emergency. Stanford Type A involves ascending aorta — surgical emergency. Type B: medical management initially.",
        "urgency": "emergent",
        "conditions": ["Aortic aneurysm", "Aortic dissection", "Aortic rupture"],
        "followup": ["CT angiography", "Vascular surgery referral", "Blood pressure control"],
    },
]


def ingest_knowledge(qdrant_url: str = "http://localhost:6333") -> bool:
    """Ingest medical knowledge into Qdrant using fastembed."""
    from qdrant_client import QdrantClient

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    collection_name = "medical_literature"

    # check if already exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info("Collection '%s' already exists — deleting and recreating", collection_name)
        client.delete_collection(collection_name)

    logger.info("Ingesting %d medical knowledge entries...", len(MEDICAL_KNOWLEDGE))

    client.add(
        collection_name=collection_name,
        documents=[entry["text"] for entry in MEDICAL_KNOWLEDGE],
        metadata=[
            {
                "id": entry["id"],
                "category": entry["category"],
                "finding": entry["finding"],
                "conditions": entry["conditions"],
                "followup": entry["followup"],
                "urgency": entry["urgency"],
                "text": entry["text"],
            }
            for entry in MEDICAL_KNOWLEDGE
        ],
        ids=[
            int(hashlib.md5(entry["id"].encode()).hexdigest()[:8], 16)
            for entry in MEDICAL_KNOWLEDGE
        ],
    )

    # verify
    info = client.get_collection(collection_name)
    logger.info(
        "Collection '%s' ready | vectors: %d",
        collection_name,
        info.points_count,
    )
    return True


if __name__ == "__main__":
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    logger.info("Ingesting medical knowledge into Qdrant at %s", qdrant_url)
    success = ingest_knowledge(qdrant_url)
    if success:
        print("\nMedical knowledge ingested successfully.")
        print(f"Total entries: {len(MEDICAL_KNOWLEDGE)}")
    else:
        print("Ingestion failed.")
