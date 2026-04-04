"""
XAI Pipeline
-------------
Grad-CAM heatmap using TorchXRayVision + medical-ai-middleware.
Clean overlay: only high-attention regions colored, same size as scan.
"""

import logging
import os
import threading
import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_model_cache = None
_model_lock = threading.Lock()


def _get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _model_lock:
        if _model_cache is not None:
            return _model_cache
        try:
            import os

            os.environ["HOME"] = "/tmp"  # redirect ~ to /tmp
            import torchxrayvision as xrv

            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model.eval()
            model = model.cpu()
            _model_cache = model
            logger.info("TorchXRayVision DenseNet loaded on CPU")
            return model
        except Exception as e:
            logger.warning("TorchXRayVision load failed: %s", e)
            return None


def _preprocess_image(png_path: str):
    import torchxrayvision as xrv
    import torchvision.transforms as T

    img = Image.open(png_path).convert("L")
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 255.0) * 2048 - 1024

    transform = T.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )
    img_tensor = transform(img_np[None, ...])
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).cpu()
    return img, img_tensor


def _clean_overlay(
    original_img: Image.Image,
    cam_np: np.ndarray,
    output_size: tuple = (512, 512),
    alpha: float = 0.45,
    threshold: float = 0.35,
) -> str:
    """
    Clean heatmap overlay:
    - Resize to output_size (matches original scan display)
    - Only regions above threshold get color
    - Below threshold: original grayscale image, no color
    - Uses jet colormap (blue→green→yellow→red) for hot regions only
    """
    # resize original to output size
    orig = original_img.convert("RGB").resize(output_size, Image.LANCZOS)
    orig_np = np.array(orig).astype(np.float32)

    # resize CAM to output size
    cam_resized = (
        np.array(
            Image.fromarray((cam_np * 255).astype(np.uint8)).resize(output_size, Image.BILINEAR)
        )
        / 255.0
    )

    # mask — only apply color where attention is above threshold
    mask = cam_resized > threshold

    # normalized attention in masked region only
    t = np.zeros_like(cam_resized)
    t[mask] = (cam_resized[mask] - threshold) / (1.0 - threshold + 1e-8)
    t = np.clip(t, 0, 1)

    # jet colormap (blue=low, cyan, green, yellow, red=high)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1) * 255
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1) * 255
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1) * 255
    jet = np.stack([r, g, b], axis=-1)

    # blend: only where mask is True
    blend = (alpha * t)[..., np.newaxis]
    blend[~mask] = 0  # zero blend outside mask → pure original

    output = orig_np * (1.0 - blend) + jet * blend
    output = np.clip(output, 0, 255).astype(np.uint8)

    result = Image.fromarray(output, mode="RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_heatmap(png_path: str, output_dir: str = "data/xai") -> dict:
    empty_result = {
        "heatmap_b64": None,
        "heatmap_path": None,
        "pathology_scores": {},
        "top_pathology": "unknown",
        "xai_method": "not_available",
    }

    try:
        from torchcam.methods import GradCAM

        os.makedirs(output_dir, exist_ok=True)

        model = _get_model()
        if model is None:
            return empty_result

        img, img_tensor = _preprocess_image(png_path)

        with torch.no_grad():
            preds = model(img_tensor).squeeze()

        pathology_scores = {
            name: round(float(torch.sigmoid(preds[i])), 3)
            for i, name in enumerate(model.pathologies)
            if name
        }

        top_idx = int(torch.sigmoid(preds).argmax())
        top_name = model.pathologies[top_idx] or "unknown"

        logger.info(
            "TorchXRayVision | top=%s (%.1f%%)", top_name, pathology_scores.get(top_name, 0) * 100
        )

        # GradCAM via torchcam — no middleware needed
        cam_extractor = GradCAM(model, target_layer="features.denseblock4")

        # need gradients for GradCAM
        img_tensor.requires_grad_(True)
        out = model(img_tensor)
        activation_map = cam_extractor(top_idx, out)[0]
        cam_extractor.remove_hooks()

        cam_np = activation_map.squeeze().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        original_png = Image.open(png_path)
        output_size = original_png.size

        heatmap_b64 = _clean_overlay(
            original_img=img,
            cam_np=cam_np,
            output_size=output_size,
            alpha=0.45,
            threshold=0.35,
        )

        heatmap_path = os.path.join(output_dir, Path(png_path).stem + "_heatmap.png")
        with open(heatmap_path, "wb") as f:
            f.write(base64.b64decode(heatmap_b64))

        logger.info("Grad-CAM complete | top=%s | saved=%s", top_name, heatmap_path)

        return {
            "heatmap_b64": heatmap_b64,
            "heatmap_path": heatmap_path,
            "pathology_scores": pathology_scores,
            "top_pathology": top_name,
            "xai_method": "gradcam_torchcam",
        }

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
    """MedGemma attention maps — for all modalities."""
    empty_result = {
        "heatmap_b64": None,
        "heatmap_path": None,
        "explanation_text": None,
        "xai_method": "not_available",
    }

    try:
        from medical_middleware.xai.attention import AttentionMap
        import torchvision.transforms as T

        os.makedirs(output_dir, exist_ok=True)

        img = Image.open(png_path).convert("RGB")
        output_size = img.size
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)

        attn = AttentionMap(model, model_type="medgemma")
        result = attn.explain(
            img_tensor,
            question=question,
            processor=processor,
            original_image=img,
            return_base64=False,
        )
        attn.remove_hooks()

        heatmap_b64 = _clean_overlay(
            original_img=img,
            cam_np=result["heatmap_raw"],
            output_size=output_size,
            alpha=0.45,
            threshold=0.25,
        )

        heatmap_path = os.path.join(output_dir, Path(png_path).stem + "_medgemma_heatmap.png")
        with open(heatmap_path, "wb") as f:
            f.write(base64.b64decode(heatmap_b64))

        return {
            "heatmap_b64": heatmap_b64,
            "heatmap_path": heatmap_path,
            "explanation_text": result.get("explanation_text"),
            "xai_method": "attention_medgemma_clean",
        }

    except Exception as e:
        logger.error("MedGemma XAI failed: %s", e, exc_info=True)
        return {**empty_result, "error": str(e)}
