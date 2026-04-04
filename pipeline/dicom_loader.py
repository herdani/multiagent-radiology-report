import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)

# GDPR: these tags contain patient identity — we strip them on load
TAGS_TO_ANONYMIZE = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
    "StudyDate",
    "SeriesDate",
]


@dataclass
class DicomScan:
    """Represents a loaded and anonymized DICOM scan."""

    pixel_array: np.ndarray  # raw pixel data
    anonymized_id: str  # hashed patient ID — safe to log/store
    modality: str  # CT, MR, CR, DX...
    rows: int
    columns: int
    metadata: dict  # non-PII metadata only


def load_and_anonymize(path: str | Path) -> DicomScan:
    """
    Load a DICOM file, strip all PII tags (GDPR compliance),
    and return a DicomScan ready for the pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds: Dataset = pydicom.dcmread(str(path))

    # --- GDPR: create anonymous ID before wiping tags ---
    raw_id = str(getattr(ds, "PatientID", "UNKNOWN"))
    anonymized_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]

    # --- Strip all PII tags ---
    for tag in TAGS_TO_ANONYMIZE:
        if hasattr(ds, tag):
            delattr(ds, tag)

    # --- Extract safe metadata ---
    metadata = {
        "modality": getattr(ds, "Modality", "UNKNOWN"),
        "study_description": getattr(ds, "StudyDescription", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
        "body_part": getattr(ds, "BodyPartExamined", ""),
        "manufacturer": getattr(ds, "Manufacturer", ""),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
        "bits_allocated": int(getattr(ds, "BitsAllocated", 16)),
    }

    pixel_array = ds.pixel_array

    logger.info(
        "Loaded DICOM | modality=%s | shape=%s | anon_id=%s",
        metadata["modality"],
        pixel_array.shape,
        anonymized_id,
    )

    return DicomScan(
        pixel_array=pixel_array,
        anonymized_id=anonymized_id,
        modality=metadata["modality"],
        rows=metadata["rows"],
        columns=metadata["columns"],
        metadata=metadata,
    )
