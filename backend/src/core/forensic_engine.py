import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any
from skimage import feature, filters

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class ForensicFeatureExtractor:
    """Advanced forensic feature extraction for deepfake detection"""

    def __init__(self):
        self.logger = logger
        # helpers for safe numeric ops

    def _safe_div(self, num, den, eps: float = 1e-8):
        try:
            den_arr = np.array(den)
            if np.all(np.abs(den_arr) <= eps):
                return 0.0
            return float(num / (den + eps))
        except Exception:
            try:
                return float(num / (den + eps))
            except Exception:
                return 0.0

    def _safe_corrcoef(self, a, b):
        try:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            if a_arr.size == 0 or b_arr.size == 0:
                return 0.0
            if np.nanstd(a_arr) == 0 or np.nanstd(b_arr) == 0:
                return 0.0
            corr = np.corrcoef(a_arr, b_arr)[0, 1]
            if np.isnan(corr) or np.isinf(corr):
                return 0.0
            return float(corr)
        except Exception:
            return 0.0

    def extract_all_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract comprehensive forensic features"""
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            features = {}

            # 1. Residual/Noise Features
            features.update(self._extract_noise_features(img_array, gray))

            # 2. Spectral Features (FFT/DCT)
            features.update(self._extract_spectral_features(gray))

            # 3. Color Statistics
            features.update(self._extract_color_statistics(img_array))

            # 4. Texture Features
            features.update(self._extract_texture_features(gray))

            # 5. Compression Artifacts
            features.update(self._extract_compression_features(image))

            # 6. Geometric Features
            features.update(self._extract_geometric_features(gray))

            # 7. Pixel-level Statistics
            features.update(self._extract_pixel_statistics(img_array))

            self.logger.debug(f"Extracted {len(features)} forensic features")
            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {}

    def _extract_noise_features(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Extract noise and residual features"""
        features = {}

        try:
            # Noise residual using median filter
            median_filtered = cv2.medianBlur(gray, 5)
            noise_residual = gray.astype(float) - median_filtered.astype(float)

            features['noise_variance'] = float(np.var(noise_residual))
            features['noise_mean'] = float(np.mean(np.abs(noise_residual)))
            features['noise_skewness'] = float(self._skewness(noise_residual.flatten()))
            features['noise_kurtosis'] = float(self._kurtosis(noise_residual.flatten()))

            # High-pass filter residual
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_pass = cv2.filter2D(gray, -1, kernel)
            features['highpass_energy'] = float(np.sum(high_pass ** 2))

            # Prewitt filter for edge residuals
            prewitt_x = cv2.filter2D(gray, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
            prewitt_y = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
            features['prewitt_magnitude'] = float(np.mean(np.sqrt(prewitt_x**2 + prewitt_y**2)))

        except Exception as e:
            self.logger.warning(f"Noise feature extraction failed: {e}")

        return features

    def _extract_spectral_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        features = {}

        try:
            # FFT features
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)

            # Frequency energy distribution
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2

            # Low, mid, high frequency energy
            low_freq_mask = self._create_circular_mask(h, w, center_y, center_x, min(h, w) // 6)
            mid_freq_mask = self._create_circular_mask(h, w, center_y, center_x, min(h, w) // 3) & ~low_freq_mask
            high_freq_mask = ~self._create_circular_mask(h, w, center_y, center_x, min(h, w) // 3)

            total_energy = np.sum(magnitude_spectrum ** 2)
            features['fft_low_freq_ratio'] = float(self._safe_div(np.sum((magnitude_spectrum * low_freq_mask) ** 2), total_energy))
            features['fft_mid_freq_ratio'] = float(self._safe_div(np.sum((magnitude_spectrum * mid_freq_mask) ** 2), total_energy))
            features['fft_high_freq_ratio'] = float(self._safe_div(np.sum((magnitude_spectrum * high_freq_mask) ** 2), total_energy))

            # DCT features
            dct = cv2.dct(gray.astype(np.float32))
            features['dct_energy'] = float(np.sum(dct ** 2))
            features['dct_ac_energy'] = float(np.sum(dct[1:, 1:] ** 2))
            features['dct_dc_ac_ratio'] = float(self._safe_div(dct[0, 0] ** 2, features['dct_ac_energy']))

        except Exception as e:
            self.logger.warning(f"Spectral feature extraction failed: {e}")

        return features

    def _extract_color_statistics(self, img: np.ndarray) -> Dict[str, float]:
        """Extract color-based statistical features"""
        features = {}

        try:
            # RGB channel statistics
            for i, channel in enumerate(['r', 'g', 'b']):
                ch = img[:, :, i]
                features[f'{channel}_mean'] = float(np.mean(ch))
                features[f'{channel}_std'] = float(np.std(ch))
                features[f'{channel}_skewness'] = float(self._skewness(ch.flatten()))
                features[f'{channel}_kurtosis'] = float(self._kurtosis(ch.flatten()))

            # Cross-channel correlations (safe)
            features['rg_correlation'] = float(self._safe_corrcoef(img[:, :, 0].flatten(), img[:, :, 1].flatten()))
            features['rb_correlation'] = float(self._safe_corrcoef(img[:, :, 0].flatten(), img[:, :, 2].flatten()))
            features['gb_correlation'] = float(self._safe_corrcoef(img[:, :, 1].flatten(), img[:, :, 2].flatten()))

            # HSV features
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            features['hue_circular_mean'] = float(np.mean(np.cos(hsv[:, :, 0] * 2 * np.pi / 180)))
            features['saturation_mean'] = float(np.mean(hsv[:, :, 1]))
            features['value_mean'] = float(np.mean(hsv[:, :, 2]))

        except Exception as e:
            self.logger.warning(f"Color statistics extraction failed: {e}")

        return features

    def _extract_texture_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features"""
        features = {}

        try:
            # Local Binary Pattern
            lbp = feature.local_binary_pattern(gray, 24, 3, method='uniform')
            features['lbp_uniformity'] = float(self._safe_div(np.sum(lbp <= 24), lbp.size))

            # Gray Level Co-occurrence Matrix features
            glcm = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
            features['glcm_contrast'] = float(np.mean(feature.graycoprops(glcm, 'contrast')))
            features['glcm_dissimilarity'] = float(np.mean(feature.graycoprops(glcm, 'dissimilarity')))
            features['glcm_homogeneity'] = float(np.mean(feature.graycoprops(glcm, 'homogeneity')))
            features['glcm_energy'] = float(np.mean(feature.graycoprops(glcm, 'energy')))

            # Gabor filter responses
            gabor_responses = []
            for theta in [0, 45, 90, 135]:
                real, _ = filters.gabor(gray, frequency=0.1, theta=np.deg2rad(theta))
                gabor_responses.append(np.mean(np.abs(real)))

            features['gabor_mean_response'] = float(np.mean(gabor_responses))
            features['gabor_std_response'] = float(np.std(gabor_responses))

        except Exception as e:
            self.logger.warning(f"Texture feature extraction failed: {e}")

        return features

    def _extract_compression_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract JPEG compression artifacts"""
        features = {}

        try:
            from src.utils.ela import error_level_analysis

            # Error Level Analysis at different qualities
            for quality in [70, 85, 95]:
                ela_img, ela_mean = error_level_analysis(image, quality=quality)
                features[f'ela_mean_q{quality}'] = ela_mean

            # Double JPEG compression detection
            features.update(self._detect_double_jpeg(image))

        except Exception as e:
            self.logger.warning(f"Compression feature extraction failed: {e}")

        return features

    def _extract_geometric_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract geometric and structural features"""
        features = {}

        try:
            # Edge density and distribution
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)

            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                features['contour_count'] = len(contours)
                features['contour_mean_area'] = float(np.mean(areas))
                features['contour_area_std'] = float(np.std(areas))

            # Symmetry features
            features['horizontal_symmetry'] = float(self._compute_symmetry(gray, axis=0))
            features['vertical_symmetry'] = float(self._compute_symmetry(gray, axis=1))

        except Exception as e:
            self.logger.warning(f"Geometric feature extraction failed: {e}")

        return features

    def _extract_pixel_statistics(self, img: np.ndarray) -> Dict[str, float]:
        """Extract pixel-level statistical features"""
        features = {}

        try:
            # Pixel value distributions
            flat = img.flatten()
            features['pixel_entropy'] = float(self._entropy(flat))
            features['pixel_range'] = float(np.ptp(flat))

            # Gradient statistics
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            features['gradient_mean'] = float(np.mean(gradient_magnitude))
            features['gradient_std'] = float(np.std(gradient_magnitude))

        except Exception as e:
            self.logger.warning(f"Pixel statistics extraction failed: {e}")

        return features

    def _create_circular_mask(self, h: int, w: int, center_y: int, center_x: int, radius: int) -> np.ndarray:
        """Create circular mask for frequency analysis"""
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        return mask

    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data"""
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _detect_double_jpeg(self, image: Image.Image) -> Dict[str, float]:
        """Detect double JPEG compression artifacts"""
        features = {}
        try:
            # This is a simplified version - in practice, you'd analyze DCT coefficients
            import io

            # Compress at different qualities and measure artifacts
            qualities = [75, 85, 95]
            artifacts = []

            for q in qualities:
                buf = io.BytesIO()
                image.save(buf, format='JPEG', quality=q)
                buf.seek(0)
                recompressed = Image.open(buf)

                # Measure difference
                diff = np.array(image).astype(float) - np.array(recompressed).astype(float)
                artifacts.append(np.mean(np.abs(diff)))

            features['double_jpeg_artifact'] = float(np.std(artifacts))

        except Exception as e:
            self.logger.warning(f"Double JPEG detection failed: {e}")

        return features

    def _compute_symmetry(self, img: np.ndarray, axis: int) -> float:
        """Compute symmetry along specified axis"""
        try:
            if axis == 0:  # horizontal symmetry
                top_half = img[:img.shape[0]//2, :]
                bottom_half = np.flipud(img[img.shape[0]//2:, :])
            else:  # vertical symmetry
                left_half = img[:, :img.shape[1]//2]
                right_half = np.fliplr(img[:, img.shape[1]//2:])

            # Ensure same dimensions
            min_dim = min(top_half.shape[0] if axis == 0 else left_half.shape[1],
                         bottom_half.shape[0] if axis == 0 else right_half.shape[1])

            if axis == 0:
                top_half = top_half[:min_dim, :]
                bottom_half = bottom_half[:min_dim, :]
                correlation = self._safe_corrcoef(top_half.flatten(), bottom_half.flatten())
            else:
                left_half = left_half[:, :min_dim]
                right_half = right_half[:, :min_dim]
                correlation = self._safe_corrcoef(left_half.flatten(), right_half.flatten())

            return correlation if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0
