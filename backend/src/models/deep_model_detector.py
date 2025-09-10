import os
from typing import Any, Dict, Optional, List
import numpy as np
from PIL import Image

from src.config import MODEL_PROVIDER, MODEL_NAME, WEIGHTS_PATH
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class DeepModelDetector:
    """Unified interface for a deepfake detection model.

    - If MODEL_PROVIDER == 'huggingface', will try to use transformers pipeline('image-classification').
    - Otherwise will attempt to load a local torch model from WEIGHTS_PATH (keeps existing TorchDetector compatible behavior).
    """

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider or MODEL_PROVIDER
        self.model_name = model_name or MODEL_NAME
        # Backend can be a HF pipeline callable or a torch module
        self._backend: Optional[Any] = None
        # Torch module handle (may remain None if not used)
        self.torch: Optional[Any] = None
        self.available = False

        # Lazy imports to avoid heavy deps at import time
        try:
            if self.provider == 'huggingface':
                # Import transformers dynamically to avoid hard dependency at import time
                import importlib
                transformers = importlib.import_module('transformers')
                pipeline_fn = getattr(transformers, 'pipeline', None)
                if pipeline_fn is None:
                    raise RuntimeError('transformers.pipeline not available')

                logger.info(f"Initializing Hugging Face pipeline for model {self.model_name}")
                # Use image-classification pipeline which returns label/probabilities
                self._backend = pipeline_fn('image-classification', model=self.model_name)
                self.available = True

            else:
                # local torch model
                import torch
                self.torch = torch
                if os.path.exists(WEIGHTS_PATH):
                    try:
                        logger.info(f"Loading local torch model from {WEIGHTS_PATH}")
                        if WEIGHTS_PATH.endswith('.pt'):
                            self._backend = torch.jit.load(WEIGHTS_PATH, map_location='cpu')
                        else:
                            self._backend = torch.load(WEIGHTS_PATH, map_location='cpu')
                        try:
                            # some torch objects may not have eval
                            self._backend.eval()
                        except Exception:
                            pass
                        self.available = True
                    except Exception as e:
                        logger.error(f"Failed to load local torch model: {e}")
                        self.available = False
                else:
                    logger.warning(f"Local weights not found at {WEIGHTS_PATH}")
                    self.available = False

        except Exception as e:
            logger.error(f"DeepModelDetector initialization failed: {e}")
            self.available = False

    def predict(self, im: Image.Image) -> Dict[str, Any]:
        """Return a dict with availability and scores/probs.

        For HF pipeline: returns {'available': True, 'probs': [{'label':..., 'score':...}, ...]}
        For local torch: tries to mimic TorchDetector.predict existing format: {'available': True, 'probs': [p0, p1]}
        """
        if not self.available:
            return {"available": False, "error": "Model not available"}

        try:
            if self.provider == 'huggingface':
                # transformers accepts PIL images directly
                backend = self._backend
                if backend is None:
                    raise RuntimeError("Backend not initialized")
                preds = backend(im)
                # Ensure list
                if not isinstance(preds, list):
                    preds = [preds]

                # Convert HF pipeline outputs to native Python types (float, str)
                normalized: List[Dict[str, Any]] = []
                for p in preds:
                    if isinstance(p, dict):
                        lbl = p.get('label')
                        scr = p.get('score')
                        try:
                            scr_val = float(scr) if scr is not None else 0.0
                        except Exception:
                            scr_val = 0.0
                        normalized.append({
                            'label': str(lbl) if lbl is not None else '',
                            'score': scr_val
                        })
                    else:
                        # fallback: coerce primitives
                        try:
                            normalized.append({'label': str(p), 'score': float(p)})
                        except Exception:
                            normalized.append({'label': str(p), 'score': 0.0})

                return {"available": True, "probs": normalized}

            else:
                # local torch model: try to follow existing TorchDetector contract
                transform = None
                try:
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                except Exception:
                    transform = None

                # Prepare input tensor `x`
                torch_module = self.torch
                if torch_module is None:
                    try:
                        import torch as torch_module
                    except Exception:
                        torch_module = None

                if transform is None:
                    # fallback: convert to numpy and pass through
                    x_np = np.array(im.resize((224, 224))).astype(np.float32) / 255.0
                    # shape -> (1, C, H, W)
                    x_np = x_np.transpose(2, 0, 1)[None, ...]
                    if torch_module is not None:
                        x = torch_module.tensor(x_np)
                    else:
                        # final fallback: pass numpy array
                        x = x_np
                else:
                    x = transform(im).unsqueeze(0)

                # Run model forward
                # Ensure backend is callable before invoking
                if self._backend is None:
                    raise RuntimeError('Model backend is not initialized')

                if callable(self._backend):
                    if torch_module is not None and hasattr(torch_module, 'no_grad'):
                        with torch_module.no_grad():
                            logits = self._backend(x)
                    else:
                        logits = self._backend(x)
                else:
                    raise RuntimeError('Model backend is not callable')

                # handle different output shapes and types
                probs_list: List[float] = []
                try:
                    # If logits is a tuple/list (some models return (logits, ...)), extract first
                    if isinstance(logits, (list, tuple)) and len(logits) > 0:
                        candidate = logits[0]
                    else:
                        candidate = logits

                    # If this candidate behaves like a torch Tensor, use torch softmax then convert
                    if torch_module is not None and hasattr(candidate, 'detach'):
                        tensor = candidate.detach()
                        # ensure batch dim
                        if getattr(tensor, 'ndim', None) == 1:
                            tensor = tensor.unsqueeze(0)
                        try:
                            sm = torch_module.nn.functional.softmax(tensor, dim=1)
                        except Exception:
                            # fallback to elementwise exp/normalize
                            arr = tensor.cpu().numpy()
                            exp = np.exp(arr)
                            sm = exp / np.sum(exp, axis=1, keepdims=True)
                            probs_arr_np = np.asarray(sm)
                        else:
                            probs_arr_np = sm.cpu().numpy()

                        probs_list = [float(v) for v in probs_arr_np.reshape(-1).tolist()]
                    else:
                        # assume numpy-like
                        arr = np.asarray(candidate)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        try:
                            exp = np.exp(arr)
                            probs = exp / np.sum(exp, axis=1, keepdims=True)
                            probs_list = [float(v) for v in probs.reshape(-1).tolist()]
                        except Exception:
                            probs_list = [float(v) for v in arr.reshape(-1).tolist()]
                except Exception:
                    # Fallback: try to coerce to list
                    try:
                        probs_list = list(map(float, logits))
                    except Exception:
                        probs_list = []

                return {"available": True, "probs": probs_list}

        except Exception as e:
            logger.error(f"Deep model prediction failed: {e}")
            return {"available": False, "error": str(e)}

    def gradcam(self, *args, **kwargs) -> Optional[Any]:
        # Grad-CAM support is model-specific; provide None by default
        logger.debug("Grad-CAM not available from DeepModelDetector by default")
        return None
