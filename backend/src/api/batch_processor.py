import asyncio
import uuid
from typing import List, Dict, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from src.utils.logging import setup_logger
from src.models.face_detector import face_detector
from src.models.detector import LegacyDetector, TorchDetector
from src.models.deep_model_detector import DeepModelDetector
from src.config import WEIGHTS_PATH

logger = setup_logger(__name__)

class BatchProcessor:
    """Process multiple images concurrently"""
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        # detectors (may be unavailable in some environments)
        from typing import Optional
        self.deep_detector: Optional[DeepModelDetector] = None
        self.legacy_detector: Optional[LegacyDetector] = None
        self.torch_detector: Optional[TorchDetector] = None

        # Initialize detectors lazily and guard failures
        try:
            self.deep_detector = DeepModelDetector()
        except Exception:
            self.deep_detector = None

        try:
            self.legacy_detector = LegacyDetector()
        except Exception:
            self.legacy_detector = None

        try:
            self.torch_detector = TorchDetector(WEIGHTS_PATH, device="cpu")
        except Exception:
            self.torch_detector = None

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _process_single_image(self, image_data: tuple) -> Dict[str, Any]:
        """Process a single image (runs in thread)"""
        image_id, image = image_data

        try:
            # Face detection
            face_im, face_found, face_conf = face_detector.detect_largest_face(image)

            # Deep model analysis (primary)
            deep_pred = None
            try:
                if self.deep_detector is not None:
                    deep_pred = self.deep_detector.predict(face_im)
            except Exception:
                deep_pred = None
            legacy_result = {"score": 0.5, "features": {}}
            deep_score = None
            if isinstance(deep_pred, dict) and deep_pred.get('available'):
                deep_probs = deep_pred.get('probs')
                legacy_result = {"score": None, "features": {"model_probs": deep_probs}}
                # determine deep_score if possible
                if isinstance(deep_probs, list) and len(deep_probs) > 0:
                    first = deep_probs[0]
                    if isinstance(first, dict) and 'score' in first:
                        deep_score = float(first['score'])
                    else:
                        try:
                            deep_score = float(first)
                        except Exception:
                            deep_score = None

            # Torch analysis if available (legacy)
            torch_result = {"available": False, "probs": []}
            try:
                if self.torch_detector is not None:
                    torch_result = self.torch_detector.predict(face_im)
            except Exception:
                torch_result = {"available": False, "probs": []}

            canonical_deep = deep_score if deep_score is not None else legacy_result.get("score")

            return {
                "id": image_id,
                "success": True,
                "face_found": face_found,
                "face_confidence": face_conf,
                "deep_model_score": canonical_deep,
                "features": legacy_result.get("features"),
                "torch_available": torch_result.get("available", False),
                "torch_probs": torch_result.get("probs", [])
            }

        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {e}")
            return {
                "id": image_id,
                "success": False,
                "error": str(e)
            }

    async def process_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process multiple images concurrently"""
        if not images:
            return []

        logger.info(f"Starting batch processing of {len(images)} images")

        # Assign IDs to images
        image_data = [(str(uuid.uuid4())[:8], img) for img in images]

        # Process in thread pool
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._process_single_image, data)
            for data in image_data
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing exception for image {i}: {result}")
                processed_results.append({
                    "id": image_data[i][0],
                    "success": False,
                    "error": str(result)
                })
            else:
                # result should be a dict; cast for mypy
                from typing import cast, Dict, Any
                processed_results.append(cast(Dict[str, Any], result))

        successful = sum(1 for r in processed_results if r.get("success", False))
        logger.info(f"Batch processing complete: {successful}/{len(images)} successful")

        return processed_results

    def __del__(self):
        """Cleanup thread pool"""
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass

# Global batch processor instance
batch_processor = BatchProcessor()
