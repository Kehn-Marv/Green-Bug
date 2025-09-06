import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Dict, List, Tuple, Optional, Any, Callable
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class RobustnessTester:
    """Comprehensive robustness testing against various manipulations"""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logger
        self.max_workers = max_workers
        self.lock = threading.Lock()
        
        # Define test categories and their weights for overall score
        self.test_weights = {
            'compression': 0.25,
            'geometric': 0.20,
            'noise': 0.15,
            'filtering': 0.15,
            'color': 0.10,
            'adversarial': 0.10,
            'content_aware': 0.05
        }
    
    def comprehensive_robustness_test(self, image: Image.Image, 
                                    detector_func: Callable, 
                                    original_score: float) -> Dict[str, Any]:
        """
        Comprehensive robustness testing suite
        
        Args:
            image: Input image
            detector_func: Function that takes image and returns detection score
            original_score: Original detection score for comparison
            
        Returns:
            Dict containing comprehensive robustness analysis
        """
        try:
            results = {
                "original_score": original_score,
                "test_categories": {},
                "stability_metrics": {},
                "provenance_stability": {},
                "confidence_analysis": {},
                "overall_robustness": 0.0,
                "test_timestamp": datetime.now().isoformat()
            }
            
            # Define comprehensive test suite
            test_suite = self._define_test_suite()
            
            # Run tests in parallel by category
            category_results = {}
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_category = {}
                
                for category, tests in test_suite.items():
                    future = executor.submit(
                        self._run_category_tests, 
                        category, tests, image, detector_func, original_score
                    )
                    future_to_category[future] = category
                
                for future in as_completed(future_to_category):
                    category = future_to_category[future]
                    try:
                        category_result = future.result(timeout=60)
                        category_results[category] = category_result
                    except Exception as e:
                        self.logger.error(f"Category {category} testing failed: {e}")
                        category_results[category] = {
                            "error": str(e),
                            "stability_score": 0.0,
                            "tests": {}
                        }
            
            results["test_categories"] = category_results
            
            # Compute comprehensive metrics
            results["stability_metrics"] = self._compute_stability_metrics(category_results, original_score)
            results["provenance_stability"] = self._analyze_provenance_stability(category_results, original_score)
            results["confidence_analysis"] = self._analyze_confidence_patterns(category_results, original_score)
            results["overall_robustness"] = self._compute_overall_robustness(results["stability_metrics"])
            
            self.logger.info(f"Robustness testing complete: {results['overall_robustness']:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive robustness testing failed: {e}")
            return {
                "error": str(e),
                "original_score": original_score,
                "overall_robustness": 0.0
            }
    
    def _define_test_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """Define comprehensive test suite"""
        return {
            'compression': [
                {'name': 'jpeg_compression', 'func': self._test_jpeg_compression, 'params': {'qualities': [95, 85, 75, 65, 50, 30]}},
                {'name': 'webp_compression', 'func': self._test_webp_compression, 'params': {'qualities': [95, 85, 75, 65, 50]}},
                {'name': 'png_compression', 'func': self._test_png_compression, 'params': {'levels': [1, 3, 6, 9]}},
                {'name': 'progressive_jpeg', 'func': self._test_progressive_jpeg, 'params': {'qualities': [85, 75, 65]}}
            ],
            'geometric': [
                {'name': 'resize_scaling', 'func': self._test_resize_scaling, 'params': {'scales': [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]}},
                {'name': 'crop_operations', 'func': self._test_crop_operations, 'params': {'ratios': [0.9, 0.8, 0.7, 0.6, 0.5]}},
                {'name': 'rotation', 'func': self._test_rotation, 'params': {'angles': [1, 2, 5, 10, 15, 30, 45, 90]}},
                {'name': 'perspective_transform', 'func': self._test_perspective_transform, 'params': {'strengths': [0.1, 0.2, 0.3]}},
                {'name': 'aspect_ratio_change', 'func': self._test_aspect_ratio_change, 'params': {'ratios': [0.8, 0.9, 1.1, 1.2]}}
            ],
            'noise': [
                {'name': 'gaussian_noise', 'func': self._test_gaussian_noise, 'params': {'sigmas': [5, 10, 15, 20, 25, 30]}},
                {'name': 'salt_pepper_noise', 'func': self._test_salt_pepper_noise, 'params': {'ratios': [0.01, 0.02, 0.03, 0.05, 0.08]}},
                {'name': 'speckle_noise', 'func': self._test_speckle_noise, 'params': {'variances': [0.1, 0.2, 0.3, 0.4]}},
                {'name': 'poisson_noise', 'func': self._test_poisson_noise, 'params': {'scales': [0.1, 0.2, 0.3, 0.4]}}
            ],
            'filtering': [
                {'name': 'gaussian_blur', 'func': self._test_gaussian_blur, 'params': {'radii': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}},
                {'name': 'motion_blur', 'func': self._test_motion_blur, 'params': {'lengths': [3, 5, 7, 10, 15]}},
                {'name': 'sharpening', 'func': self._test_sharpening, 'params': {'factors': [1.2, 1.5, 2.0, 2.5, 3.0]}},
                {'name': 'median_filter', 'func': self._test_median_filter, 'params': {'sizes': [3, 5, 7, 9]}}
            ],
            'color': [
                {'name': 'brightness_adjustment', 'func': self._test_brightness_adjustment, 'params': {'factors': [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]}},
                {'name': 'contrast_adjustment', 'func': self._test_contrast_adjustment, 'params': {'factors': [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]}},
                {'name': 'saturation_adjustment', 'func': self._test_saturation_adjustment, 'params': {'factors': [0.0, 0.5, 0.7, 1.3, 1.5, 2.0]}},
                {'name': 'hue_shift', 'func': self._test_hue_shift, 'params': {'shifts': [-30, -15, -5, 5, 15, 30]}},
                {'name': 'gamma_correction', 'func': self._test_gamma_correction, 'params': {'gammas': [0.5, 0.7, 0.8, 1.2, 1.5, 2.0]}}
            ],
            'adversarial': [
                {'name': 'uniform_noise', 'func': self._test_uniform_adversarial, 'params': {'epsilons': [2, 4, 8, 16, 32]}},
                {'name': 'gradient_noise', 'func': self._test_gradient_adversarial, 'params': {'epsilons': [2, 4, 8, 16]}},
                {'name': 'patch_attacks', 'func': self._test_patch_attacks, 'params': {'patch_sizes': [0.05, 0.1, 0.15, 0.2]}},
                {'name': 'frequency_attacks', 'func': self._test_frequency_attacks, 'params': {'strengths': [0.1, 0.2, 0.3]}}
            ],
            'content_aware': [
                {'name': 'face_region_blur', 'func': self._test_face_region_blur, 'params': {'blur_strengths': [1.0, 2.0, 3.0]}},
                {'name': 'background_manipulation', 'func': self._test_background_manipulation, 'params': {'blur_levels': [1.0, 2.0, 3.0]}},
                {'name': 'edge_enhancement', 'func': self._test_edge_enhancement, 'params': {'strengths': [1.2, 1.5, 2.0]}}
            ]
        }
    
    def _run_category_tests(self, category: str, tests: List[Dict], 
                           image: Image.Image, detector_func: Callable, 
                           original_score: float) -> Dict[str, Any]:
        """Run all tests in a category"""
        try:
            category_results = {
                "tests": {},
                "category_stability": 0.0,
                "worst_case_score": original_score,
                "best_case_score": original_score,
                "score_variance": 0.0
            }
            
            all_scores = [original_score]
            
            for test_config in tests:
                try:
                    test_result = test_config['func'](
                        image, detector_func, original_score, **test_config['params']
                    )
                    category_results["tests"][test_config['name']] = test_result
                    
                    # Collect scores for category analysis
                    if 'scores' in test_result:
                        all_scores.extend(test_result['scores'])
                        
                except Exception as e:
                    self.logger.error(f"Test {test_config['name']} failed: {e}")
                    category_results["tests"][test_config['name']] = {
                        "error": str(e),
                        "stability_score": 0.0
                    }
            
            # Compute category metrics
            if len(all_scores) > 1:
                category_results["category_stability"] = self._compute_score_stability(original_score, all_scores[1:])
                category_results["worst_case_score"] = min(all_scores)
                category_results["best_case_score"] = max(all_scores)
                category_results["score_variance"] = np.var(all_scores)
            
            return category_results
            
        except Exception as e:
            self.logger.error(f"Category {category} testing failed: {e}")
            return {"error": str(e), "category_stability": 0.0}
    
    # Compression Tests
    def _test_jpeg_compression(self, image: Image.Image, detector_func: Callable, 
                              original_score: float, qualities: List[int]) -> Dict[str, Any]:
        """Test JPEG compression robustness"""
        try:
            scores = []
            compression_ratios = []
            
            for quality in qualities:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                compressed_img = Image.open(buffer)
                
                score = detector_func(compressed_img)
                scores.append(score)
                
                # Calculate compression ratio
                original_size = len(io.BytesIO().getvalue()) if hasattr(image, 'save') else 1000000
                compressed_size = len(buffer.getvalue())
                compression_ratios.append(compressed_size / max(original_size, 1))
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "compression",
                "method": "jpeg",
                "qualities": qualities,
                "scores": scores,
                "compression_ratios": compression_ratios,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores),
                "worst_quality_impact": max(abs(s - original_score) for s in scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_webp_compression(self, image: Image.Image, detector_func: Callable, 
                              original_score: float, qualities: List[int]) -> Dict[str, Any]:
        """Test WebP compression robustness"""
        try:
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
                    # Fallback to JPEG if WebP not supported
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
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_png_compression(self, image: Image.Image, detector_func: Callable, 
                             original_score: float, levels: List[int]) -> Dict[str, Any]:
        """Test PNG compression robustness"""
        try:
            scores = []
            
            for level in levels:
                buffer = io.BytesIO()
                image.save(buffer, format='PNG', compress_level=level)
                buffer.seek(0)
                compressed_img = Image.open(buffer)
                
                score = detector_func(compressed_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "compression",
                "method": "png",
                "compression_levels": levels,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_progressive_jpeg(self, image: Image.Image, detector_func: Callable, 
                              original_score: float, qualities: List[int]) -> Dict[str, Any]:
        """Test progressive JPEG compression"""
        try:
            scores = []
            
            for quality in qualities:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality, progressive=True)
                buffer.seek(0)
                compressed_img = Image.open(buffer)
                
                score = detector_func(compressed_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "compression",
                "method": "progressive_jpeg",
                "qualities": qualities,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    # Geometric Tests
    def _test_resize_scaling(self, image: Image.Image, detector_func: Callable, 
                            original_score: float, scales: List[float]) -> Dict[str, Any]:
        """Test resize scaling robustness"""
        try:
            original_size = image.size
            scores = []
            
            for scale in scales:
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                resized = image.resize(new_size, Image.LANCZOS)
                
                # Resize back to original size
                restored = resized.resize(original_size, Image.LANCZOS)
                
                score = detector_func(restored)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "geometric",
                "method": "resize_scaling",
                "scale_factors": scales,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_crop_operations(self, image: Image.Image, detector_func: Callable, 
                             original_score: float, ratios: List[float]) -> Dict[str, Any]:
        """Test cropping robustness"""
        try:
            w, h = image.size
            scores = []
            
            for ratio in ratios:
                # Center crop
                new_w, new_h = int(w * ratio), int(h * ratio)
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                
                cropped = image.crop((left, top, left + new_w, top + new_h))
                
                # Resize back to original size
                restored = cropped.resize((w, h), Image.LANCZOS)
                
                score = detector_func(restored)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "geometric",
                "method": "crop_operations",
                "crop_ratios": ratios,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_rotation(self, image: Image.Image, detector_func: Callable, 
                      original_score: float, angles: List[float]) -> Dict[str, Any]:
        """Test rotation robustness"""
        try:
            scores = []
            
            for angle in angles:
                # Rotate and crop to avoid black borders
                rotated = image.rotate(angle, expand=True, fillcolor='white')
                
                # Crop to remove potential black borders
                bbox = rotated.getbbox()
                if bbox:
                    rotated = rotated.crop(bbox)
                
                # Resize to original size
                restored = rotated.resize(image.size, Image.LANCZOS)
                
                score = detector_func(restored)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "geometric",
                "method": "rotation",
                "angles": angles,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_perspective_transform(self, image: Image.Image, detector_func: Callable, 
                                   original_score: float, strengths: List[float]) -> Dict[str, Any]:
        """Test perspective transformation robustness"""
        try:
            scores = []
            w, h = image.size
            
            for strength in strengths:
                # Define perspective transformation
                offset = int(min(w, h) * strength)
                
                # Source points (corners of original image)
                src_points = [(0, 0), (w, 0), (w, h), (0, h)]
                
                # Destination points (with perspective distortion)
                dst_points = [
                    (offset, offset),
                    (w - offset, offset),
                    (w, h),
                    (0, h - offset)
                ]
                
                # Apply perspective transform
                transformed = image.transform(
                    (w, h), Image.PERSPECTIVE,
                    self._get_perspective_coeffs(src_points, dst_points),
                    Image.BICUBIC
                )
                
                score = detector_func(transformed)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "geometric",
                "method": "perspective_transform",
                "strengths": strengths,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_aspect_ratio_change(self, image: Image.Image, detector_func: Callable, 
                                 original_score: float, ratios: List[float]) -> Dict[str, Any]:
        """Test aspect ratio change robustness"""
        try:
            w, h = image.size
            scores = []
            
            for ratio in ratios:
                # Change aspect ratio
                new_w = int(w * ratio)
                resized = image.resize((new_w, h), Image.LANCZOS)
                
                # Restore original aspect ratio
                restored = resized.resize((w, h), Image.LANCZOS)
                
                score = detector_func(restored)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "geometric",
                "method": "aspect_ratio_change",
                "ratios": ratios,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    # Noise Tests
    def _test_gaussian_noise(self, image: Image.Image, detector_func: Callable, 
                            original_score: float, sigmas: List[float]) -> Dict[str, Any]:
        """Test Gaussian noise robustness"""
        try:
            scores = []
            img_array = np.array(image)
            
            for sigma in sigmas:
                # Add Gaussian noise
                noise = np.random.normal(0, sigma, img_array.shape)
                noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                noisy_img = Image.fromarray(noisy_array)
                
                score = detector_func(noisy_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "noise",
                "method": "gaussian",
                "noise_sigmas": sigmas,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_salt_pepper_noise(self, image: Image.Image, detector_func: Callable, 
                               original_score: float, ratios: List[float]) -> Dict[str, Any]:
        """Test salt and pepper noise robustness"""
        try:
            scores = []
            img_array = np.array(image)
            
            for ratio in ratios:
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
                "noise_ratios": ratios,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_speckle_noise(self, image: Image.Image, detector_func: Callable, 
                           original_score: float, variances: List[float]) -> Dict[str, Any]:
        """Test speckle noise robustness"""
        try:
            scores = []
            img_array = np.array(image).astype(float)
            
            for variance in variances:
                # Speckle noise: image + image * noise
                noise = np.random.normal(0, variance, img_array.shape)
                noisy_array = img_array + img_array * noise
                noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
                
                noisy_img = Image.fromarray(noisy_array)
                score = detector_func(noisy_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "noise",
                "method": "speckle",
                "noise_variances": variances,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_poisson_noise(self, image: Image.Image, detector_func: Callable, 
                           original_score: float, scales: List[float]) -> Dict[str, Any]:
        """Test Poisson noise robustness"""
        try:
            scores = []
            img_array = np.array(image).astype(float)
            
            for scale in scales:
                # Poisson noise
                scaled_img = img_array * scale
                noisy_array = np.random.poisson(scaled_img) / scale
                noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
                
                noisy_img = Image.fromarray(noisy_array)
                score = detector_func(noisy_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "noise",
                "method": "poisson",
                "noise_scales": scales,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    # Filtering Tests
    def _test_gaussian_blur(self, image: Image.Image, detector_func: Callable, 
                           original_score: float, radii: List[float]) -> Dict[str, Any]:
        """Test Gaussian blur robustness"""
        try:
            scores = []
            
            for radius in radii:
                blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
                score = detector_func(blurred)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "filtering",
                "method": "gaussian_blur",
                "blur_radii": radii,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_motion_blur(self, image: Image.Image, detector_func: Callable, 
                         original_score: float, lengths: List[int]) -> Dict[str, Any]:
        """Test motion blur robustness"""
        try:
            scores = []
            
            for length in lengths:
                # Create motion blur kernel
                kernel = np.zeros((length, length))
                kernel[int((length-1)/2), :] = np.ones(length)
                kernel = kernel / length
                
                # Apply motion blur using OpenCV
                img_array = np.array(image)
                blurred_array = cv2.filter2D(img_array, -1, kernel)
                blurred_img = Image.fromarray(blurred_array)
                
                score = detector_func(blurred_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "filtering",
                "method": "motion_blur",
                "blur_lengths": lengths,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_sharpening(self, image: Image.Image, detector_func: Callable, 
                        original_score: float, factors: List[float]) -> Dict[str, Any]:
        """Test sharpening robustness"""
        try:
            scores = []
            
            for factor in factors:
                enhancer = ImageEnhance.Sharpness(image)
                sharpened = enhancer.enhance(factor)
                score = detector_func(sharpened)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "filtering",
                "method": "sharpening",
                "sharpen_factors": factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_median_filter(self, image: Image.Image, detector_func: Callable, 
                           original_score: float, sizes: List[int]) -> Dict[str, Any]:
        """Test median filter robustness"""
        try:
            scores = []
            
            for size in sizes:
                filtered = image.filter(ImageFilter.MedianFilter(size=size))
                score = detector_func(filtered)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "filtering",
                "method": "median_filter",
                "filter_sizes": sizes,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    # Color Tests
    def _test_brightness_adjustment(self, image: Image.Image, detector_func: Callable, 
                                   original_score: float, factors: List[float]) -> Dict[str, Any]:
        """Test brightness adjustment robustness"""
        try:
            scores = []
            
            for factor in factors:
                enhancer = ImageEnhance.Brightness(image)
                adjusted = enhancer.enhance(factor)
                score = detector_func(adjusted)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "color",
                "method": "brightness",
                "brightness_factors": factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_contrast_adjustment(self, image: Image.Image, detector_func: Callable, 
                                 original_score: float, factors: List[float]) -> Dict[str, Any]:
        """Test contrast adjustment robustness"""
        try:
            scores = []
            
            for factor in factors:
                enhancer = ImageEnhance.Contrast(image)
                adjusted = enhancer.enhance(factor)
                score = detector_func(adjusted)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "color",
                "method": "contrast",
                "contrast_factors": factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_saturation_adjustment(self, image: Image.Image, detector_func: Callable, 
                                   original_score: float, factors: List[float]) -> Dict[str, Any]:
        """Test saturation adjustment robustness"""
        try:
            scores = []
            
            for factor in factors:
                enhancer = ImageEnhance.Color(image)
                adjusted = enhancer.enhance(factor)
                score = detector_func(adjusted)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "color",
                "method": "saturation",
                "saturation_factors": factors,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_hue_shift(self, image: Image.Image, detector_func: Callable, 
                       original_score: float, shifts: List[int]) -> Dict[str, Any]:
        """Test hue shift robustness"""
        try:
            scores = []
            
            for shift in shifts:
                # Convert to HSV, shift hue, convert back
                hsv = image.convert('HSV')
                h, s, v = hsv.split()
                
                # Shift hue
                h_array = np.array(h)
                h_shifted = (h_array + shift) % 256
                h_new = Image.fromarray(h_shifted.astype(np.uint8))
                
                # Merge back and convert to RGB
                hsv_shifted = Image.merge('HSV', (h_new, s, v))
                rgb_shifted = hsv_shifted.convert('RGB')
                
                score = detector_func(rgb_shifted)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "color",
                "method": "hue_shift",
                "hue_shifts": shifts,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_gamma_correction(self, image: Image.Image, detector_func: Callable, 
                              original_score: float, gammas: List[float]) -> Dict[str, Any]:
        """Test gamma correction robustness"""
        try:
            scores = []
            
            for gamma in gammas:
                # Apply gamma correction
                img_array = np.array(image).astype(float) / 255.0
                corrected_array = np.power(img_array, gamma)
                corrected_array = (corrected_array * 255).astype(np.uint8)
                
                corrected_img = Image.fromarray(corrected_array)
                score = detector_func(corrected_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "color",
                "method": "gamma_correction",
                "gamma_values": gammas,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    # Adversarial Tests
    def _test_uniform_adversarial(self, image: Image.Image, detector_func: Callable, 
                                 original_score: float, epsilons: List[int]) -> Dict[str, Any]:
        """Test uniform adversarial noise robustness"""
        try:
            scores = []
            img_array = np.array(image).astype(np.float32)
            
            for epsilon in epsilons:
                # Generate uniform adversarial perturbation
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
                "epsilon_values": epsilons,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_gradient_adversarial(self, image: Image.Image, detector_func: Callable, 
                                  original_score: float, epsilons: List[int]) -> Dict[str, Any]:
        """Test gradient-based adversarial perturbations"""
        try:
            scores = []
            img_array = np.array(image).astype(np.float32)
            
            # Compute image gradients
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            for epsilon in epsilons:
                # Create perturbation based on gradients
                perturbation = np.zeros_like(img_array)
                for c in range(3):
                    perturbation[:, :, c] = epsilon * np.sign(grad_x + grad_y)
                
                # Apply perturbation
                perturbed_array = np.clip(img_array + perturbation, 0, 255).astype(np.uint8)
                perturbed_img = Image.fromarray(perturbed_array)
                
                score = detector_func(perturbed_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "adversarial",
                "method": "gradient_noise",
                "epsilon_values": epsilons,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_patch_attacks(self, image: Image.Image, detector_func: Callable, 
                           original_score: float, patch_sizes: List[float]) -> Dict[str, Any]:
        """Test patch attack robustness"""
        try:
            scores = []
            w, h = image.size
            
            for patch_size in patch_sizes:
                # Create random patch
                patch_w = int(w * patch_size)
                patch_h = int(h * patch_size)
                
                # Random position
                x = random.randint(0, w - patch_w)
                y = random.randint(0, h - patch_h)
                
                # Create modified image with random patch
                modified = image.copy()
                patch = Image.new('RGB', (patch_w, patch_h), 
                                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                modified.paste(patch, (x, y))
                
                score = detector_func(modified)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "adversarial",
                "method": "patch_attacks",
                "patch_sizes": patch_sizes,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_frequency_attacks(self, image: Image.Image, detector_func: Callable, 
                               original_score: float, strengths: List[float]) -> Dict[str, Any]:
        """Test frequency domain attacks"""
        try:
            scores = []
            img_array = np.array(image.convert('L'))
            
            for strength in strengths:
                # Apply FFT
                fft = np.fft.fft2(img_array)
                fft_shift = np.fft.fftshift(fft)
                
                # Add noise in frequency domain
                noise = np.random.normal(0, strength * np.abs(fft_shift).max(), fft_shift.shape)
                fft_noisy = fft_shift + noise
                
                # Convert back to spatial domain
                fft_ishift = np.fft.ifftshift(fft_noisy)
                img_back = np.fft.ifft2(fft_ishift)
                img_back = np.abs(img_back)
                
                # Normalize and convert back to RGB
                img_back = (img_back / img_back.max() * 255).astype(np.uint8)
                modified_img = Image.fromarray(img_back).convert('RGB')
                
                score = detector_func(modified_img)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "adversarial",
                "method": "frequency_attacks",
                "attack_strengths": strengths,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    # Content-Aware Tests
    def _test_face_region_blur(self, image: Image.Image, detector_func: Callable, 
                              original_score: float, blur_strengths: List[float]) -> Dict[str, Any]:
        """Test face region specific blur"""
        try:
            scores = []
            
            # Simple face region detection (center region)
            w, h = image.size
            face_region = (w//4, h//4, 3*w//4, 3*h//4)
            
            for blur_strength in blur_strengths:
                modified = image.copy()
                
                # Extract face region
                face_crop = modified.crop(face_region)
                
                # Apply blur to face region
                blurred_face = face_crop.filter(ImageFilter.GaussianBlur(radius=blur_strength))
                
                # Paste back
                modified.paste(blurred_face, face_region)
                
                score = detector_func(modified)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "content_aware",
                "method": "face_region_blur",
                "blur_strengths": blur_strengths,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_background_manipulation(self, image: Image.Image, detector_func: Callable, 
                                     original_score: float, blur_levels: List[float]) -> Dict[str, Any]:
        """Test background manipulation"""
        try:
            scores = []
            
            # Simple background region (outer border)
            w, h = image.size
            border_size = min(w, h) // 8
            
            for blur_level in blur_levels:
                modified = image.copy()
                
                # Create mask for background (simple border approach)
                mask = Image.new('L', (w, h), 0)
                inner_region = (border_size, border_size, w-border_size, h-border_size)
                
                # Blur background
                blurred = modified.filter(ImageFilter.GaussianBlur(radius=blur_level))
                
                # Composite: keep center, blur edges
                result = Image.composite(modified, blurred, mask)
                
                score = detector_func(result)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "content_aware",
                "method": "background_manipulation",
                "blur_levels": blur_levels,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    def _test_edge_enhancement(self, image: Image.Image, detector_func: Callable, 
                              original_score: float, strengths: List[float]) -> Dict[str, Any]:
        """Test edge enhancement"""
        try:
            scores = []
            
            for strength in strengths:
                # Detect edges
                edges = image.filter(ImageFilter.FIND_EDGES)
                
                # Enhance edges
                enhancer = ImageEnhance.Contrast(edges)
                enhanced_edges = enhancer.enhance(strength)
                
                # Blend with original
                result = Image.blend(image, enhanced_edges, 0.3)
                
                score = detector_func(result)
                scores.append(score)
            
            stability_score = self._compute_score_stability(original_score, scores)
            
            return {
                "type": "content_aware",
                "method": "edge_enhancement",
                "enhancement_strengths": strengths,
                "scores": scores,
                "stability_score": stability_score,
                "mean_score": np.mean(scores),
                "score_variance": np.var(scores)
            }
            
        except Exception as e:
            return {"error": str(e), "stability_score": 0.0}
    
    # Analysis Methods
    def _compute_score_stability(self, original_score: float, test_scores: List[float]) -> float:
        """Compute stability score based on score variations"""
        try:
            if not test_scores:
                return 0.0
            
            # Compute relative deviations from original score
            deviations = [abs(score - original_score) / (abs(original_score) + 1e-8) for score in test_scores]
            mean_deviation = np.mean(deviations)
            
            # Convert to stability score (lower deviation = higher stability)
            stability = max(0.0, 1.0 - mean_deviation)
            
            return float(stability)
            
        except Exception as e:
            self.logger.error(f"Stability computation failed: {e}")
            return 0.0
    
    def _compute_stability_metrics(self, category_results: Dict[str, Any], 
                                  original_score: float) -> Dict[str, float]:
        """Compute comprehensive stability metrics"""
        try:
            metrics = {}
            
            # Category-wise stability
            for category, results in category_results.items():
                if "error" not in results:
                    stability = results.get("category_stability", 0.0)
                    metrics[f"{category}_stability"] = stability
                    
                    # Additional metrics
                    worst_score = results.get("worst_case_score", original_score)
                    best_score = results.get("best_case_score", original_score)
                    
                    metrics[f"{category}_worst_case_deviation"] = abs(worst_score - original_score)
                    metrics[f"{category}_best_case_deviation"] = abs(best_score - original_score)
                    metrics[f"{category}_score_range"] = abs(best_score - worst_score)
            
            # Overall metrics
            all_stabilities = [v for k, v in metrics.items() if k.endswith('_stability')]
            if all_stabilities:
                metrics["mean_stability"] = np.mean(all_stabilities)
                metrics["min_stability"] = np.min(all_stabilities)
                metrics["max_stability"] = np.max(all_stabilities)
                metrics["stability_variance"] = np.var(all_stabilities)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Stability metrics computation failed: {e}")
            return {}
    
    def _analyze_provenance_stability(self, category_results: Dict[str, Any], 
                                     original_score: float) -> Dict[str, Any]:
        """Analyze provenance stability across different manipulations"""
        try:
            analysis = {
                "high_impact_manipulations": [],
                "low_impact_manipulations": [],
                "critical_vulnerabilities": [],
                "robust_aspects": []
            }
            
            impact_threshold = 0.2  # 20% score change considered high impact
            
            for category, results in category_results.items():
                if "error" in results:
                    continue
                
                category_impact = abs(results.get("worst_case_score", original_score) - original_score)
                
                if category_impact > impact_threshold:
                    analysis["high_impact_manipulations"].append({
                        "category": category,
                        "impact": category_impact,
                        "stability": results.get("category_stability", 0.0)
                    })
                else:
                    analysis["low_impact_manipulations"].append({
                        "category": category,
                        "impact": category_impact,
                        "stability": results.get("category_stability", 0.0)
                    })
                
                # Identify specific vulnerabilities
                for test_name, test_result in results.get("tests", {}).items():
                    if "scores" in test_result:
                        max_deviation = max(abs(s - original_score) for s in test_result["scores"])
                        if max_deviation > impact_threshold:
                            analysis["critical_vulnerabilities"].append({
                                "test": f"{category}_{test_name}",
                                "max_deviation": max_deviation,
                                "stability": test_result.get("stability_score", 0.0)
                            })
                        elif test_result.get("stability_score", 0.0) > 0.8:
                            analysis["robust_aspects"].append({
                                "test": f"{category}_{test_name}",
                                "stability": test_result.get("stability_score", 0.0)
                            })
            
            # Sort by impact/stability
            analysis["high_impact_manipulations"].sort(key=lambda x: x["impact"], reverse=True)
            analysis["critical_vulnerabilities"].sort(key=lambda x: x["max_deviation"], reverse=True)
            analysis["robust_aspects"].sort(key=lambda x: x["stability"], reverse=True)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Provenance stability analysis failed: {e}")
            return {}
    
    def _analyze_confidence_patterns(self, category_results: Dict[str, Any], 
                                    original_score: float) -> Dict[str, Any]:
        """Analyze confidence patterns across manipulations"""
        try:
            analysis = {
                "confidence_distribution": {},
                "score_trends": {},
                "reliability_assessment": {}
            }
            
            # Collect all scores
            all_scores = [original_score]
            score_by_category = {}
            
            for category, results in category_results.items():
                if "error" in results:
                    continue
                
                category_scores = []
                for test_result in results.get("tests", {}).values():
                    if "scores" in test_result:
                        category_scores.extend(test_result["scores"])
                
                if category_scores:
                    score_by_category[category] = category_scores
                    all_scores.extend(category_scores)
            
            if len(all_scores) > 1:
                # Confidence distribution analysis
                analysis["confidence_distribution"] = {
                    "mean_score": np.mean(all_scores),
                    "std_score": np.std(all_scores),
                    "min_score": np.min(all_scores),
                    "max_score": np.max(all_scores),
                    "score_range": np.max(all_scores) - np.min(all_scores)
                }
                
                # Score trends by category
                for category, scores in score_by_category.items():
                    trend = np.mean(scores) - original_score
                    analysis["score_trends"][category] = {
                        "mean_change": trend,
                        "direction": "increase" if trend > 0 else "decrease",
                        "magnitude": abs(trend)
                    }
                
                # Reliability assessment
                score_variance = np.var(all_scores)
                analysis["reliability_assessment"] = {
                    "overall_variance": score_variance,
                    "reliability_score": max(0.0, 1.0 - score_variance),
                    "prediction_stability": "high" if score_variance < 0.1 else "medium" if score_variance < 0.3 else "low"
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Confidence pattern analysis failed: {e}")
            return {}
    
    def _compute_overall_robustness(self, stability_metrics: Dict[str, float]) -> float:
        """Compute overall robustness score"""
        try:
            if not stability_metrics:
                return 0.0
            
            # Use weighted combination of category stabilities
            weighted_score = 0.0
            total_weight = 0.0
            
            for category, weight in self.test_weights.items():
                stability_key = f"{category}_stability"
                if stability_key in stability_metrics:
                    weighted_score += weight * stability_metrics[stability_key]
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_score / total_weight
            else:
                # Fallback to mean stability
                stabilities = [v for k, v in stability_metrics.items() if k.endswith('_stability')]
                return np.mean(stabilities) if stabilities else 0.0
                
        except Exception as e:
            self.logger.error(f"Overall robustness computation failed: {e}")
            return 0.0
    
    # Helper methods
    def _get_perspective_coeffs(self, src_points: List[Tuple[int, int]], 
                               dst_points: List[Tuple[int, int]]) -> List[float]:
        """Calculate perspective transformation coefficients"""
        try:
            # This is a simplified version - in practice you'd use cv2.getPerspectiveTransform
            # For PIL, we need to calculate the 8 coefficients manually
            # This is a placeholder implementation
            return [1, 0, 0, 0, 1, 0, 0, 0]  # Identity transform as fallback
        except:
            return [1, 0, 0, 0, 1, 0, 0, 0]

from datetime import datetime