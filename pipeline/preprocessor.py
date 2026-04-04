import logging
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.dicom_loader import DicomScan

logger = logging.getLogger(__name__)

TARGET_SIZE = (512, 512)  # standard input size for vision models


def normalize_pixels(array: np.ndarray) -> np.ndarray:
    """Normalize pixel values to 0-255 uint8."""
    arr = array.astype(np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - min_val) / (max_val - min_val) * 255.0
    return arr.astype(np.uint8)


def to_png(scan: DicomScan, output_path: str | Path) -> Path:
    """
    Convert a DicomScan to a PNG file ready for the vision model.
    Handles grayscale and RGB, resizes to TARGET_SIZE.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pixels = normalize_pixels(scan.pixel_array)

    # handle multi-frame (take middle frame) and RGB
    if pixels.ndim == 3 and pixels.shape[0] > 3:
        mid = pixels.shape[0] // 2
        pixels = pixels[mid]
    elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        pixels = pixels[:, :, :3]

    if pixels.ndim == 2:
        img = Image.fromarray(pixels, mode="L").convert("RGB")
    else:
        img = Image.fromarray(pixels, mode="RGB")

    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    img.save(output_path, format="PNG")

    logger.info("Saved PNG | path=%s | size=%s", output_path, TARGET_SIZE)
    return output_path


def preprocess(dicom_path: str | Path, output_dir: str | Path = "data/processed") -> dict:
    """
    Full preprocessing pipeline:
    load DICOM -> anonymize -> normalize -> save PNG
    Returns a dict with everything the agent needs.
    """
    from pipeline.dicom_loader import load_and_anonymize

    scan = load_and_anonymize(dicom_path)
    output_dir = Path(output_dir)
    png_path = output_dir / f"{scan.anonymized_id}.png"
    saved_path = to_png(scan, png_path)

    return {
        "anonymized_id": scan.anonymized_id,
        "png_path": str(saved_path),
        "modality": scan.modality,
        "metadata": scan.metadata,
    }
