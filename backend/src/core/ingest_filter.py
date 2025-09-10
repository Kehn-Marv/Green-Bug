import hashlib
from typing import Dict, List, Any
import io
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
from datetime import datetime

from src.utils.logging import setup_logger
from src.config import QUALITY_MIN_SIDE, FACE_CONFIDENCE_THRESHOLD

logger = setup_logger(__name__)

class IngestFilter:
    """Advanced ingest and filtering system for deepfake detection"""

    def __init__(self):
        self.logger = logger
        self.quality_thresholds = {
            'min_resolution': QUALITY_MIN_SIDE,
            'face_confidence': FACE_CONFIDENCE_THRESHOLD,
            'blur_threshold': 100.0,
            'noise_threshold': 0.8,
            'compression_quality': 0.3
        }

    def process_upload(self, image_data: bytes, filename: str,
                      strip_exif: bool = True) -> Dict[str, Any]:
        """
        Complete ingest processing pipeline
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            strip_exif: Whether to strip EXIF data
            
        Returns:
            Dict containing processed image, metadata, and quality assessment
        """
        try:
            # Generate file hash for integrity
            file_hash = hashlib.sha256(image_data).hexdigest()

            # Load image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            original_size = image.size

            # Extract EXIF before potential stripping
            exif_data = self._extract_exif_metadata(image)

            # Strip EXIF if requested
            if strip_exif:
                image = self._strip_exif(image)
                logger.info(f"EXIF data stripped from {filename}")

            # Compute comprehensive quality score
            quality_assessment = self._compute_quality_score(image, exif_data)

            # Determine if suitable for training
            training_candidate = self._assess_training_suitability(
                quality_assessment, image, exif_data
            )

            result = {
                'processed_image': image,
                'file_hash': file_hash,
                'original_filename': filename,
                'original_size': original_size,
                'file_size': len(image_data),
                'exif_data': exif_data if not strip_exif else {},
                'quality_assessment': quality_assessment,
                'training_candidate': training_candidate,
                'processing_timestamp': datetime.now().isoformat(),
                'exif_stripped': strip_exif
            }

            logger.info(f"Processed upload: {filename}, quality_score: {quality_assessment['overall_score']:.3f}")
            return result

        except Exception as e:
            logger.error(f"Upload processing failed for {filename}: {e}")
            raise

    def _extract_exif_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extract comprehensive EXIF metadata"""
        try:
            exif_dict: Dict[str, Any] = {}
            exif_data = image.getexif()

            if exif_data:
                for tag_id, value in exif_data.items():
                    # Ensure tag is a string key
                    tag = TAGS.get(tag_id, tag_id)
                    tag_key = str(tag) if tag is not None else str(tag_id)

                    # Skip binary data
                    if isinstance(value, bytes):
                        continue

                    # Handle special cases
                    if tag_key == 'DateTime':
                        try:
                            exif_dict['capture_datetime'] = datetime.strptime(
                                str(value), '%Y:%m:%d %H:%M:%S'
                            ).isoformat()
                        except:
                            exif_dict['capture_datetime'] = str(value)
                    elif tag in ['Make', 'Model', 'Software']:
                        key = tag_key.lower()
                        exif_dict[key] = str(value)
                    elif tag in ['ExifImageWidth', 'ExifImageHeight']:
                        key = tag_key.lower()
                        try:
                            exif_dict[key] = int(value)
                        except Exception:
                            exif_dict[key] = value
                    elif tag == 'Orientation':
                        exif_dict['orientation'] = int(value)
                    else:
                        exif_dict[tag_key] = value

            # Add derived metadata
            exif_dict['has_camera_info'] = any(
                key in exif_dict for key in ['make', 'model']
            )
            exif_dict['has_software_info'] = 'software' in exif_dict
            exif_dict['metadata_richness'] = len(exif_dict)

            return exif_dict

        except Exception as e:
            logger.warning(f"EXIF extraction failed: {e}")
            return {}

    def _strip_exif(self, image: Image.Image) -> Image.Image:
        """Remove EXIF data from image"""
        try:
            # Create new image without EXIF
            data = list(image.getdata())
            clean_image = Image.new(image.mode, image.size)
            clean_image.putdata(data)
            return clean_image
        except Exception as e:
            logger.warning(f"EXIF stripping failed: {e}")
            return image

    def _compute_quality_score(self, image: Image.Image,
                              exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive quality assessment"""
        try:
            scores = {}

            # Resolution quality
            width, height = image.size
            min_side = min(width, height)
            scores['resolution'] = min(1.0, min_side / self.quality_thresholds['min_resolution'])

            # Blur detection using Laplacian variance
            gray = np.array(image.convert('L'))
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores['sharpness'] = min(1.0, laplacian_var / self.quality_thresholds['blur_threshold'])

            # Noise assessment
            noise_score = self._assess_noise_level(gray)
            scores['noise'] = 1.0 - min(1.0, noise_score / self.quality_thresholds['noise_threshold'])

            # Compression artifacts
            compression_score = self._assess_compression_artifacts(image)
            scores['compression'] = compression_score

            # Color distribution quality
            color_score = self._assess_color_quality(image)
            scores['color_distribution'] = color_score

            # EXIF authenticity indicators
            exif_score = self._assess_exif_authenticity(exif_data)
            scores['exif_authenticity'] = exif_score

            # Compute weighted overall score
            weights = {
                'resolution': 0.25,
                'sharpness': 0.20,
                'noise': 0.15,
                'compression': 0.15,
                'color_distribution': 0.15,
                'exif_authenticity': 0.10
            }

            overall_score = sum(scores[key] * weights[key] for key in weights.keys())

            return {
                'individual_scores': scores,
                'overall_score': overall_score,
                'weights_used': weights,
                'quality_flags': self._generate_quality_flags(scores),
                'assessment_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'individual_scores': {},
                'overall_score': 0.5,
                'error': str(e)
            }

    def _assess_noise_level(self, gray_image: np.ndarray) -> float:
        """Assess noise level in grayscale image"""
        try:
            # Use median filter to estimate noise
            median_filtered = cv2.medianBlur(gray_image, 5)
            noise = gray_image.astype(float) - median_filtered.astype(float)
            return float(np.std(noise))
        except:
            return 0.5

    def _assess_compression_artifacts(self, image: Image.Image) -> float:
        """Assess JPEG compression artifacts"""
        try:
            import io

            # Recompress at high quality and measure difference
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            recompressed = Image.open(buffer)

            # Calculate difference
            orig_array = np.array(image)
            recomp_array = np.array(recompressed)

            if orig_array.shape == recomp_array.shape:
                diff = np.mean(np.abs(orig_array.astype(float) - recomp_array.astype(float)))
                # Lower difference indicates less compression artifacts
                return max(0.0, 1.0 - diff / 50.0)

            return 0.5

        except Exception:
            return 0.5

    def _assess_color_quality(self, image: Image.Image) -> float:
        """Assess color distribution quality"""
        try:
            # Convert to HSV for better color analysis
            hsv = image.convert('HSV')
            h, s, v = hsv.split()

            # Check for color diversity
            h_hist = np.histogram(np.array(h), bins=36)[0]
            s_hist = np.histogram(np.array(s), bins=32)[0]
            v_hist = np.histogram(np.array(v), bins=32)[0]

            # Calculate entropy (higher entropy = better color distribution)
            h_entropy = self._calculate_entropy(h_hist)
            s_entropy = self._calculate_entropy(s_hist)
            v_entropy = self._calculate_entropy(v_hist)

            # Normalize and combine
            max_entropy = np.log2(36)  # Maximum possible entropy for hue
            color_score = (h_entropy + s_entropy + v_entropy) / (3 * max_entropy)

            return min(1.0, color_score)

        except Exception:
            return 0.5

    def _assess_exif_authenticity(self, exif_data: Dict[str, Any]) -> float:
        """Assess EXIF data authenticity indicators"""
        try:
            score = 0.0

            # Presence of camera information
            if exif_data.get('has_camera_info', False):
                score += 0.3

            # Presence of capture datetime
            if 'capture_datetime' in exif_data:
                score += 0.2

            # Reasonable metadata richness
            richness = exif_data.get('metadata_richness', 0)
            if richness > 10:
                score += 0.3
            elif richness > 5:
                score += 0.2

            # Software information (can be suspicious if AI-related)
            software_raw = exif_data.get('software', '')
            # Ensure we treat software metadata as string for analysis
            software = str(software_raw).lower() if software_raw is not None else ''
            if software:
                if any(term in software for term in ['photoshop', 'gimp', 'ai', 'generated']):
                    score += 0.1  # Suspicious but not necessarily fake
                else:
                    score += 0.2

            return min(1.0, score)

        except Exception:
            return 0.5

    def _calculate_entropy(self, histogram: np.ndarray) -> float:
        """Calculate entropy of histogram"""
        try:
            # Normalize histogram
            hist_norm = histogram / (np.sum(histogram) + 1e-8)
            # Remove zeros
            hist_norm = hist_norm[hist_norm > 0]
            # Calculate entropy
            return -np.sum(hist_norm * np.log2(hist_norm))
        except:
            return 0.0

    def _generate_quality_flags(self, scores: Dict[str, float]) -> List[str]:
        """Generate quality warning flags"""
        flags = []

        if scores.get('resolution', 1.0) < 0.7:
            flags.append('low_resolution')

        if scores.get('sharpness', 1.0) < 0.5:
            flags.append('blurry_image')

        if scores.get('noise', 1.0) < 0.6:
            flags.append('high_noise')

        if scores.get('compression', 1.0) < 0.4:
            flags.append('heavy_compression')

        if scores.get('color_distribution', 1.0) < 0.4:
            flags.append('poor_color_distribution')

        if scores.get('exif_authenticity', 1.0) < 0.3:
            flags.append('suspicious_metadata')

        return flags

    def _assess_training_suitability(self, quality_assessment: Dict[str, Any],
                                   image: Image.Image,
                                   exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if image is suitable for training"""
        try:
            overall_score = quality_assessment.get('overall_score', 0.0)
            quality_flags = quality_assessment.get('quality_flags', [])

            # Base suitability on overall quality
            suitable = overall_score >= 0.7

            # Additional checks
            reasons = []

            if overall_score < 0.7:
                reasons.append(f'low_quality_score_{overall_score:.2f}')

            if 'low_resolution' in quality_flags:
                suitable = False
                reasons.append('insufficient_resolution')

            if 'blurry_image' in quality_flags:
                suitable = False
                reasons.append('image_too_blurry')

            # Check for potential synthetic indicators in EXIF
            software_raw = exif_data.get('software', '')
            software = str(software_raw).lower() if software_raw is not None else ''
            if any(term in software for term in ['ai', 'generated', 'synthetic']):
                reasons.append('suspicious_software_metadata')

            confidence = overall_score if suitable else 0.0

            return {
                'suitable_for_training': suitable,
                'confidence': confidence,
                'quality_score': overall_score,
                'rejection_reasons': reasons,
                'assessment_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Training suitability assessment failed: {e}")
            return {
                'suitable_for_training': False,
                'confidence': 0.0,
                'error': str(e)
            }

import cv2
import io
