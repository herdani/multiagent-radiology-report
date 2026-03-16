"""
XAI Pipeline
-------------
Generates Grad-CAM heatmaps using TorchXRayVision + medical-ai-middleware.
Uses moebouassida/medical-ai-middleware for GradCAM and visualization.

For chest modalities (CR, DX, CT):
  - TorchXRayVision DenseNet for pathology detection
  - medical-ai-middleware GradCAM for heatmap generation
  - medical-ai-middleware visualization for overlay

When MedGemma is available:
  - swap to medical-ai-middleware AttentionMap(model, model_type="medgemma")
  - works for all modalities not just chest
"""
import logging
import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# cache model — load once, reuse across calls
_model_cache = None
_model_lock  = threading.Lock()


def _get_model():
    """Load TorchXRayVision DenseNet once and cache it."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _model_lock:
        if _model_cache is not None:
            return _model_cache
        try:
            import torchxrayvision as xrv
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model.eval()
            _model_cache = model
            logger.info("TorchXRayVision DenseNet loaded and cached")
            return model
        except Exception as e:
            logger.warning("TorchXRayVision load failed: %s", e)
            return None


def _preprocess_image(png_path: str):
    """Preprocess PNG for TorchXRayVision."""
    import torchxrayvision as xrv
    import torchvision.transforms as T

    img = Image.open(png_path).convert("L")
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 255.0) * 2048 - 1024

    transform = T.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img_tensor = transform(img_np[None, ...])
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
    return img, img_tensor


def generate_heatmap(png_path: str, output_dir: str = "data/xai") -> dict:
    """
    Generate Grad-CAM heatmap using:
      - TorchXRayVision for pathology detection + target class
      - medical-ai-middleware GradCAM for gradient computation
      - medical-ai-middleware visualization for overlay rendering

    Returns dict with heatmap_b64, heatmap_path, pathology_scores, top_pathology.
    Never raises — returns empty result on any failure.
    """
    empty_result = {
        "heatmap_b64":      None,
        "heatmap_path":     None,
        "pathology_scores": {},
        "top_pathology":    "unknown",
        "xai_method":       "not_available",
    }

    try:
        from medical_middleware.xai.gradcam import GradCAM
        from medical_middleware.xai.visualization import overlay_heatmap, image_to_base64

        os.makedirs(output_dir, exist_ok=True)

        model = _get_model()
        if model is None:
            return empty_result

        img, img_tensor = _preprocess_image(png_path)

        # get pathology predictions
        with torch.no_grad():
            preds = model(img_tensor).squeeze()

        pathology_scores = {
            name: round(float(torch.sigmoid(preds[i])), 3)
            for i, name in enumerate(model.pathologies)
            if name
        }

        # find most confident pathology for Grad-CAM target
        top_idx  = int(torch.sigmoid(preds).argmax())
        top_name = model.pathologies[top_idx] or "unknown"

        logger.info(
            "TorchXRayVision | top=%s (%.1f%%) | running GradCAM...",
            top_name, pathology_scores.get(top_name, 0) * 100,
        )

        # use YOUR middleware GradCAM
        target_layer = model.features.denseblock4
        cam = GradCAM(model, target_layer=target_layer)

        result = cam.explain(
            img_tensor,
            target_class=top_idx,
            original_image=img,
            return_base64=True,
            alpha=0.5,
        )
        cam.remove_hooks()
        model.zero_grad()

        heatmap_b64 = result["heatmap_b64"]

        # save heatmap to disk
        import base64, io
        heatmap_path = os.path.join(
            output_dir, Path(png_path).stem + "_heatmap.png"
        )
        heatmap_bytes = base64.b64decode(heatmap_b64)
        with open(heatmap_path, "wb") as f:
            f.write(heatmap_bytes)

        logger.info(
            "Grad-CAM complete | top=%s | method=%s | saved=%s",
            top_name, result["method"], heatmap_path,
        )

        return {
            "heatmap_b64":      heatmap_b64,
            "heatmap_path":     heatmap_path,
            "pathology_scores": pathology_scores,
            "top_pathology":    top_name,
            "xai_method":       f"gradcam_middleware_{result['method']}",
        }

    except ImportError:
        logger.error(
            "medical-ai-middleware not installed. "
            "Run: pip install 'medical-ai-middleware[all] @ "
            "git+https://github.com/moebouassida/medical-ai-middleware.git'"
        )
        return empty_result

    except Exception as e:
        logger.error("XAI generation failed: %s", e, exc_info=True)
        return {**empty_result, "error": str(e)}


def generate_heatmap_medgemma(
    png_path: str,
    model,
    processor,
    question: str = "What findings are visible in this medical image?",
    output_dir: str = "data/xai",
) -> dict:
    """
    Generate attention map using MedGemma via medical-ai-middleware.
    Use this when MedGemma is available (RTX 2060 / Vertex AI).

    Replaces generate_heatmap() for non-chest modalities.
    """
    empty_result = {
        "heatmap_b64":       None,
        "heatmap_path":      None,
        "explanation_text":  None,
        "xai_method":        "not_available",
    }

    try:
        from medical_middleware.xai.attention import AttentionMap
        import torchvision.transforms as T
        import base64, io

        os.makedirs(output_dir, exist_ok=True)

        img = Image.open(png_path).convert("RGB")
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)

        # use YOUR middleware AttentionMap with medgemma
        attn = AttentionMap(model, model_type="medgemma")
        result = attn.explain(
            img_tensor,
            question=question,
            processor=processor,
            original_image=img,
            return_base64=True,
            alpha=0.5,
        )
        attn.remove_hooks()

        heatmap_b64  = result.get("heatmap_b64")
        heatmap_path = None

        if heatmap_b64:
            heatmap_path = os.path.join(
                output_dir, Path(png_path).stem + "_medgemma_heatmap.png"
            )
            with open(heatmap_path, "wb") as f:
                f.write(base64.b64decode(heatmap_b64))

        logger.info(
            "MedGemma attention map generated | layers=%d | saved=%s",
            result.get("layers_captured", 0), heatmap_path,
        )

        return {
            "heatmap_b64":      heatmap_b64,
            "heatmap_path":     heatmap_path,
            "explanation_text": result.get("explanation_text"),
            "xai_method":       f"attention_medgemma_{result.get('method', '')}",
        }

    except Exception as e:
        logger.error("MedGemma XAI failed: %s", e, exc_info=True)
        return {**empty_result, "error": str(e)}
