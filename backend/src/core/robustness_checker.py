import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Type

# Pillow Resampling fallback
try:
    Resampling: Optional[Type] = getattr(Image, 'Resampling', None)
except Exception:
    Resampling = None

LANCZOS = getattr(Resampling, 'LANCZOS', getattr(Image, 'LANCZOS', None))
from typing import Dict, List, Any
import io
from concurrent.futures import ThreadPoolExecutor
import threading

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class RobustnessChecker:
    """Test robustness against various image manipulations"""

    def __init__(self, max_workers: int = 4):
        self.logger = logger
        self.max_workers = max_workers
        self.lock = threading.Lock()

    def test_robustness(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Comprehensive robustness testing"""
        try:
            from typing import Dict, Any
            results: Dict[str, Any] = {
                "original_score": original_score,
                "tests": {},
                "stability_metrics": {},
                "overall_robustness": 0.0
            }

            # Define test suite
            test_suite = [
                ("compression_jpeg", self._test_jpeg_compression),
                ("compression_webp", self._test_webp_compression),
                ("geometric_resize", self._test_resize),
                ("geometric_crop", self._test_crop),
                ("noise_gaussian", self._test_gaussian_noise),
                ("noise_salt_pepper", self._test_salt_pepper_noise),
                ("filtering_blur", self._test_blur),
                ("filtering_sharpen", self._test_sharpen),
                ("color_brightness", self._test_brightness),
                ("color_contrast", self._test_contrast),
                ("adversarial_basic", self._test_adversarial_noise)
            ]

            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_test = {
                    executor.submit(self._run_single_test, test_name, test_func, image, detector_func, original_score): test_name
                    for test_name, test_func in test_suite
                }

                for future in future_to_test:
                    test_name = future_to_test[future]
                    try:
                        test_result = future.result(timeout=30)  # 30 second timeout per test
                        results["tests"][test_name] = test_result
                    except Exception as e:
                        self.logger.error(f"Test {test_name} failed: {e}")
                        results["tests"][test_name] = {"error": str(e), "stability_score": 0.0}

            # Compute stability metrics
            results["stability_metrics"] = self._compute_stability_metrics(results["tests"], original_score)
            results["overall_robustness"] = self._compute_overall_robustness(results["stability_metrics"])

            self.logger.info(f"Robustness testing complete: {results['overall_robustness']:.3f}")
            return results

        except Exception as e:
            self.logger.error(f"Robustness testing failed: {e}")
            return {"error": str(e), "original_score": original_score}

    def _run_single_test(self, test_name: str, test_func, image: Image.Image,
                        detector_func, original_score: float) -> Dict[str, Any]:
        """Run a single robustness test"""
        try:
            return test_func(image, detector_func, original_score)
        except Exception as e:
            self.logger.error(f"Single test {test_name} failed: {e}")
            return {"error": str(e), "stability_score": 0.0}

    def _test_jpeg_compression(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against JPEG compression"""
        try:
            qualities = [95, 85, 75, 65, 50]
            scores = []

            for quality in qualities:
                # Compress image
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                compressed_img = Image.open(buffer)

                # Get new score
                score = detector_func(compressed_img)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "compression",
                "method": "jpeg",
                "qualities": qualities,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": float(np.mean(scores)),
                "score_variance": float(np.var(scores))
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_webp_compression(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against WebP compression"""
        try:
            qualities = [95, 85, 75, 65, 50]
            scores = []

            for quality in qualities:
                try:
                    buffer = io.BytesIO()
                    image.save(buffer, format='WEBP', quality=quality)
                    buffer.seek(0)
                    compressed_img = Image.open(buffer)

                    score = detector_func(compressed_img)
                    scores.append(score)
                except Exception:
                    # WebP might not be available, use JPEG as fallback
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=quality)
                    buffer.seek(0)
                    compressed_img = Image.open(buffer)

                    score = detector_func(compressed_img)
                    scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "compression",
                "method": "webp",
                "qualities": qualities,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": float(np.mean(scores)),
                "score_variance": float(np.var(scores))
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_resize(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against resizing"""
        try:
            original_size = image.size
            scale_factors = [0.8, 0.9, 1.1, 1.2, 1.5]
            scores = []

            for scale in scale_factors:
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                resized = image.resize(new_size, LANCZOS or Image.LANCZOS)

                # Resize back to original size
                restored = resized.resize(original_size, LANCZOS or Image.LANCZOS)

                score = detector_func(restored)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "geometric",
                "method": "resize",
                "scale_factors": scale_factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": float(np.mean(scores)),
                "score_variance": float(np.var(scores))
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_crop(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against cropping"""
        try:
            w, h = image.size
            crop_ratios = [0.9, 0.8, 0.7, 0.6]
            scores = []

            for ratio in crop_ratios:
                # Center crop
                new_w, new_h = int(w * ratio), int(h * ratio)
                left = (w - new_w) // 2
                top = (h - new_h) // 2

                cropped = image.crop((left, top, left + new_w, top + new_h))

                # Resize back to original size
                restored = cropped.resize((w, h), LANCZOS or Image.LANCZOS)

                score = detector_func(restored)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "geometric",
                "method": "crop",
                "crop_ratios": crop_ratios,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": float(np.mean(scores)),
                "score_variance": float(np.var(scores))
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_gaussian_noise(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against Gaussian noise"""
        try:
            noise_levels = [5, 10, 15, 20, 25]
            scores = []

            img_array = np.array(image)

            for noise_level in noise_levels:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, img_array.shape)
                noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                noisy_img = Image.fromarray(noisy_array)

                score = detector_func(noisy_img)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "noise",
                "method": "gaussian",
                "noise_levels": noise_levels,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_salt_pepper_noise(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against salt and pepper noise"""
        try:
            noise_ratios = [0.01, 0.02, 0.03, 0.05, 0.08]
            scores = []

            img_array = np.array(image)

            for ratio in noise_ratios:
                noisy_array = img_array.copy()

                # Salt noise
                salt_coords = tuple([np.random.randint(0, i - 1, int(ratio * img_array.size / 2))
                                   for i in img_array.shape[:2]])
                noisy_array[salt_coords] = 255

                # Pepper noise
                pepper_coords = tuple([np.random.randint(0, i - 1, int(ratio * img_array.size / 2))
                                     for i in img_array.shape[:2]])
                noisy_array[pepper_coords] = 0

                noisy_img = Image.fromarray(noisy_array)
                score = detector_func(noisy_img)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "noise",
                "method": "salt_pepper",
                "noise_ratios": noise_ratios,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_blur(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against blur"""
        try:
            blur_radii = [0.5, 1.0, 1.5, 2.0, 2.5]
            scores = []

            for radius in blur_radii:
                blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
                score = detector_func(blurred)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "filtering",
                "method": "gaussian_blur",
                "blur_radii": blur_radii,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_sharpen(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against sharpening"""
        try:
            sharpen_factors = [1.2, 1.5, 2.0, 2.5, 3.0]
            scores = []

            for factor in sharpen_factors:
                enhancer = ImageEnhance.Sharpness(image)
                sharpened = enhancer.enhance(factor)
                score = detector_func(sharpened)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "filtering",
                "method": "sharpen",
                "sharpen_factors": sharpen_factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_brightness(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against brightness changes"""
        try:
            brightness_factors = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
            scores = []

            for factor in brightness_factors:
                enhancer = ImageEnhance.Brightness(image)
                adjusted = enhancer.enhance(factor)
                score = detector_func(adjusted)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "color",
                "method": "brightness",
                "brightness_factors": brightness_factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_contrast(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against contrast changes"""
        try:
            contrast_factors = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
            scores = []

            for factor in contrast_factors:
                enhancer = ImageEnhance.Contrast(image)
                adjusted = enhancer.enhance(factor)
                score = detector_func(adjusted)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "color",
                "method": "contrast",
                "contrast_factors": contrast_factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _test_adversarial_noise(self, image: Image.Image, detector_func, original_score: float) -> Dict[str, Any]:
        """Test robustness against adversarial-like noise"""
        try:
            epsilon_values = [2, 4, 8, 16, 32]
            scores = []

            img_array = np.array(image).astype(np.float32)

            for epsilon in epsilon_values:
                # Generate adversarial-like perturbation
                perturbation = np.random.uniform(-epsilon, epsilon, img_array.shape)

                # Apply perturbation
                perturbed_array = np.clip(img_array + perturbation, 0, 255).astype(np.uint8)
                perturbed_img = Image.fromarray(perturbed_array)

                score = detector_func(perturbed_img)
                scores.append(score)

            stability_score = self._compute_score_stability(original_score, scores)

            return {
                "type": "adversarial",
                "method": "uniform_noise",
                "epsilon_values": epsilon_values,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }

        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}

    def _compute_score_stability(self, original_score: float, test_scores: List[float]) -> float:
        """Compute stability score based on score variations"""
        try:
            if not test_scores:
                return 0.0

            # Compute relative deviations from original score
            deviations = [abs(score - original_score) / (original_score + 1e-8) for score in test_scores]
            mean_deviation = float(np.mean(deviations))

            # Convert to stability score (lower deviation = higher stability)
            stability = max(0.0, 1.0 - mean_deviation)

            return float(stability)

        except Exception as e:
            self.logger.error(f"Stability computation failed: {e}")
            return 0.0

    def _compute_stability_metrics(self, test_results: Dict[str, Any], original_score: float) -> Dict[str, float]:
        """Compute overall stability metrics"""
        try:
            metrics = {}

            # Group by test type
            test_types: Dict[str, List[float]] = {}
            for test_name, result in test_results.items():
                if "error" in result:
                    continue

                test_type = result.get("type", "unknown")
                if test_type not in test_types:
                    test_types[test_type] = []
                test_types[test_type].append(result.get("stability_score", 0.0))

            # Compute metrics per type
            for test_type, scores in test_types.items():
                if scores:
                    metrics[f"{test_type}_stability"] = float(np.mean(scores))
                    metrics[f"{test_type}_min_stability"] = float(np.min(scores))
                    metrics[f"{test_type}_max_stability"] = float(np.max(scores))

            # Overall metrics
            all_scores = []
            for result in test_results.values():
                if "stability_score" in result and "error" not in result:
                    all_scores.append(result["stability_score"])

            if all_scores:
                metrics["mean_stability"] = float(np.mean(all_scores))
                metrics["min_stability"] = float(np.min(all_scores))
                metrics["max_stability"] = float(np.max(all_scores))
                metrics["stability_variance"] = float(np.var(all_scores))

            return metrics

        except Exception as e:
            self.logger.error(f"Stability metrics computation failed: {e}")
            return {}

    def _compute_overall_robustness(self, stability_metrics: Dict[str, float]) -> float:
        """Compute overall robustness score"""
        try:
            if not stability_metrics:
                return 0.0

            # Weight different types of robustness
            weights = {
                "compression": 0.25,
                "geometric": 0.20,
                "noise": 0.20,
                "filtering": 0.15,
                "color": 0.15,
                "adversarial": 0.05
            }

            weighted_score = 0.0
            total_weight = 0.0

            for test_type, weight in weights.items():
                stability_key = f"{test_type}_stability"
                if stability_key in stability_metrics:
                    weighted_score += weight * stability_metrics[stability_key]
                    total_weight += weight

            if total_weight > 0:
                return float(weighted_score / total_weight)
            else:
                # Fallback to mean stability (ensure native float)
                mean_stab = stability_metrics.get("mean_stability", 0.0)
                return float(mean_stab)

        except Exception as e:
            self.logger.error(f"Overall robustness computation failed: {e}")
            return 0.0
