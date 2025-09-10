"""Detectors module - clean, single implementation.

This module exposes:
- LegacyDetector: a test-friendly detector that prefers a deep model when
  available and falls back to lightweight features and a stable combiner.
- TorchDetector: thin wrapper around a local torch model (optional).

The file intentionally avoids duplicated definitions and keeps imports
cohesive so test collection and import-time behavior is stable.
"""

import os
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
except Exception:
    from typing import Any, Optional as _Optional
    torch: _Optional[Any] = None
    T: _Optional[Any] = None

from src.utils.ela import error_level_analysis
from src.utils.frequency import fft_highfreq_ratio, laplacian_variance, jpeg_quant_score
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def _safe_import_deep_detector():
    try:
        from src.models.deep_model_detector import DeepModelDetector

        return DeepModelDetector
    except Exception:
        return None


class LegacyDetector:
    """Compatibility detector: prefer a deep model, else fallback to simple features.

    The implementation is intentionally small and deterministic to keep tests
    fast and reliable.
    """

    def __init__(self, use_deep: bool = True) -> None:
        DeepModelDetector = _safe_import_deep_detector()
        self._deep = None
        if use_deep and DeepModelDetector is not None:
            try:
                self._deep = DeepModelDetector()
            except Exception:
                self._deep = None

    def analyze(self, im: Image.Image) -> Dict[str, Any]:
        # Prefer deep model output when available and functional
        if self._deep and getattr(self._deep, "available", False):
            try:
                pred = self._deep.predict(im)
                if pred.get("available"):
                    probs = pred.get("probs")
                    score = None
                    if isinstance(probs, list) and len(probs) > 0:
                        first = probs[0]
                        if isinstance(first, dict) and "score" in first:
                            score = float(first["score"]) if first["score"] is not None else None
                        else:
                            try:
                                score = float(first)
                            except Exception:
                                score = None
                    return {"score": score if score is not None else 0.5, "features": {"model_probs": probs}}
            except Exception:
                logger.debug("Deep model predict failed inside LegacyDetector.analyze")

        # Fallback: neutral result
        feats = self.features(im)
        return {"score": self.score(feats), "features": feats}

    def features(self, im: Image.Image) -> Dict[str, float]:
        try:
            _, ela_mean = error_level_analysis(im, quality=90)
            fft_ratio = fft_highfreq_ratio(im)
            lap_var = laplacian_variance(im)
            jq = jpeg_quant_score(im)
            return {"ela_mean": ela_mean, "fft_high_ratio": fft_ratio, "lap_var": lap_var, "jpeg_score": jq}
        except Exception:
            return {"ela_mean": 0.0, "fft_high_ratio": 0.5, "lap_var": 100.0, "jpeg_score": 0.5}

    def score(self, feats: Dict[str, float]) -> float:
        # If deep model is present, try to use it preferentially
        if self._deep and getattr(self._deep, "available", False):
            try:
                # call analyze on a small image to get deep score
                return float(self.analyze(Image.new("RGB", (1, 1)))["score"])
            except Exception:
                pass

        try:
            x = 0.0
            x += feats.get("fft_high_ratio", 0.5) * 0.9
            x += (feats.get("ela_mean", 0.0) / 50.0) * 0.6
            x += (feats.get("lap_var", 100.0) / 200.0) * 0.2
            s = 1.0 / (1.0 + np.exp(-4.0 * (x - 0.7)))
            return float(np.clip(s, 0.0, 1.0))
        except Exception:
            return 0.5


class TorchDetector:
    """Thin Torch wrapper used when a local model file is provided.

    This class lazily uses torch if it's available; otherwise it reports
    unavailable so test runs without torch still work.
    """

    def __init__(self, weights_path: str, device: Optional[str] = None, target_layer: Optional[str] = None) -> None:
        self.weights_path = weights_path
        self.available = torch is not None and os.path.exists(weights_path)
        self.model = None
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.target_layer = target_layer
        self.transform = None
        if T is not None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if self.available:
            self._load_model()

    def _load_model(self) -> None:
        try:
            if self.weights_path.endswith(".pt") and torch is not None:
                self.model = torch.jit.load(self.weights_path, map_location=self.device)
            elif torch is not None:
                self.model = torch.load(self.weights_path, map_location=self.device)
            else:
                self.model = None

            if self.model is not None:
                self.model.eval()
                logger.info(f"Torch model loaded from {self.weights_path}")
            else:
                self.available = False
        except Exception as e:
            logger.error(f"Failed to load torch model: {e}")
            self.available = False
            self.model = None

    def predict(self, im: Image.Image) -> Dict[str, Any]:
        if not self.available or self.model is None or self.transform is None:
            return {"available": False, "error": "Model not available"}
        try:
            x = self.transform(im).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].tolist()
            return {"available": True, "probs": probs}
        except Exception as e:
            logger.error(f"Torch model prediction failed: {e}")
            return {"available": False, "error": str(e)}


# Backwards-compatible alias used during migration
HeuristicDetector = LegacyDetector
