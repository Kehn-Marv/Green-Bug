import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy import ndimage
from skimage import feature, filters
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class ForensicFeatureExtractor:
    """Comprehensive forensic feature extraction for deepfake detection"""

    def __init__(self):
        self.logger = logger
        self.scaler = StandardScaler()

        # Initialize CNN feature extractor
        self.cnn_extractor = self._initialize_cnn_extractor()

    def _initialize_cnn_extractor(self):
        """Initialize CNN for high-level feature extraction"""
        try:
            # Use a pre-trained ResNet for feature extraction
            import torchvision.models as models

            model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            logger.info("CNN feature extractor initialized successfully")
            return model

        except Exception as e:
            logger.warning(f"CNN feature extractor initialization failed: {e}")
            return None

    def extract_comprehensive_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract all forensic features from image
        
        Returns:
            Dict containing all extracted features organized by category
        """
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            features: Dict[str, Any] = {
                'residual_noise': self._extract_residual_noise_features(img_array, gray),
                'spectral': self._extract_spectral_features(gray),
                'color_statistics': self._extract_color_statistics(img_array),
                'texture': self._extract_texture_features(gray),
                'compression': self._extract_compression_features(image),
                'geometric': self._extract_geometric_features(gray),
                'cnn_embeddings': self._extract_cnn_features(image),
                'metadata': {
                    'image_size': image.size,
                    'channels': len(img_array.shape),
                    'dtype': str(img_array.dtype),
                    'extraction_timestamp': datetime.now().isoformat()
                }
            }

            # Flatten features for compatibility
            flattened_features = self._flatten_features(features)

            logger.info(f"Extracted {len(flattened_features)} forensic features")

            return {
                'structured_features': features,
                'flattened_features': flattened_features,
                'feature_count': len(flattened_features)
            }

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                'structured_features': {},
                'flattened_features': {},
                'error': str(e)
            }

    # Utility helpers to avoid divide-by-zero and NaN propagation
    def _safe_div(self, num, den, eps: float = 1e-8):
        """Safely divide num by den and return 0.0 if denominator is (near) zero."""
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
        """Compute correlation safely; return 0.0 if either input has zero variance or result is NaN."""
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

    def _extract_residual_noise_features(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Extract residual and noise-based features"""
        features: Dict[str, float] = {}

        try:
            # 1. Median filter residual
            median_filtered = cv2.medianBlur(gray, 5)
            median_residual = gray.astype(float) - median_filtered.astype(float)

            features['median_residual_mean'] = float(np.mean(np.abs(median_residual)))
            features['median_residual_std'] = float(np.std(median_residual))
            features['median_residual_skew'] = float(self._skewness(median_residual.flatten()))
            features['median_residual_kurtosis'] = float(self._kurtosis(median_residual.flatten()))

            # 2. Gaussian filter residual
            gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 1.0)
            gaussian_residual = gray.astype(float) - gaussian_filtered.astype(float)

            features['gaussian_residual_mean'] = float(np.mean(np.abs(gaussian_residual)))
            features['gaussian_residual_std'] = float(np.std(gaussian_residual))

            # 3. High-pass filter responses
            kernels = {
                'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
                'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                'prewitt_x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                'prewitt_y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            }

            for kernel_name, kernel in kernels.items():
                filtered = cv2.filter2D(gray, -1, kernel)
                features[f'{kernel_name}_energy'] = float(np.sum(filtered ** 2))
                features[f'{kernel_name}_mean'] = float(np.mean(np.abs(filtered)))

            # 4. Wavelet-based noise estimation
            try:
                import pywt
                coeffs = pywt.dwt2(gray, 'db4')
                _, (lh, hl, hh) = coeffs

                # Estimate noise from high-frequency subbands
                sigma_lh = np.median(np.abs(lh)) / 0.6745
                sigma_hl = np.median(np.abs(hl)) / 0.6745
                sigma_hh = np.median(np.abs(hh)) / 0.6745

                features['wavelet_noise_lh'] = float(sigma_lh)
                features['wavelet_noise_hl'] = float(sigma_hl)
                features['wavelet_noise_hh'] = float(sigma_hh)
                features['wavelet_noise_avg'] = float((sigma_lh + sigma_hl + sigma_hh) / 3)

            except ImportError:
                logger.warning("PyWavelets not available, skipping wavelet features")

            # 5. Local variance analysis
            local_var = ndimage.generic_filter(gray.astype(float), np.var, size=5)
            features['local_variance_mean'] = float(np.mean(local_var))
            features['local_variance_std'] = float(np.std(local_var))

        except Exception as e:
            logger.warning(f"Residual noise feature extraction failed: {e}")

        return features

    def _extract_spectral_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        features: Dict[str, float] = {}

        try:
            # 1. FFT-based features
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            phase_spectrum = np.angle(fft_shift)

            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2

            # Create frequency masks
            y, x = np.ogrid[:h, :w]
            center_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Low, mid, high frequency regions
            max_dist = min(h, w) // 2
            low_mask = center_dist <= max_dist * 0.2
            mid_mask = (center_dist > max_dist * 0.2) & (center_dist <= max_dist * 0.6)
            high_mask = center_dist > max_dist * 0.6

            total_energy = np.sum(magnitude_spectrum ** 2)
            features['fft_low_energy_ratio'] = float(self._safe_div(np.sum((magnitude_spectrum * low_mask) ** 2), total_energy))
            features['fft_mid_energy_ratio'] = float(self._safe_div(np.sum((magnitude_spectrum * mid_mask) ** 2), total_energy))
            features['fft_high_energy_ratio'] = float(self._safe_div(np.sum((magnitude_spectrum * high_mask) ** 2), total_energy))

            # Spectral entropy
            magnitude_norm = magnitude_spectrum / (np.sum(magnitude_spectrum) + 1e-8)
            magnitude_norm = magnitude_norm[magnitude_norm > 0]
            features['spectral_entropy'] = float(-np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-8)))

            # Phase coherence
            features['phase_coherence'] = float(np.std(phase_spectrum))

            # 2. DCT-based features
            dct = cv2.dct(gray.astype(np.float32))

            # DC component
            features['dct_dc_energy'] = float(dct[0, 0] ** 2)

            # AC energy in different regions
            ac_energy = np.sum(dct[1:, 1:] ** 2)
            features['dct_ac_energy'] = float(ac_energy)
            features['dct_dc_ac_ratio'] = float(self._safe_div(features['dct_dc_energy'], ac_energy))

            # Low frequency AC components
            low_freq_ac = np.sum(dct[1:8, 1:8] ** 2)
            features['dct_low_freq_ac'] = float(low_freq_ac)

            # 3. Power spectral density features
            psd = np.abs(fft) ** 2
            features['psd_mean'] = float(np.mean(psd))
            features['psd_std'] = float(np.std(psd))
            features['psd_max'] = float(np.max(psd))

            # Radial power spectrum
            radial_profile = self._compute_radial_profile(magnitude_spectrum, center_x, center_y)
            features['radial_spectrum_slope'] = float(self._compute_spectrum_slope(radial_profile))

        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")

        return features

    def _extract_color_statistics(self, img: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive color-based features"""
        features: Dict[str, float] = {}

        try:
            # 1. RGB channel statistics
            for i, channel in enumerate(['r', 'g', 'b']):
                ch = img[:, :, i]
                features[f'{channel}_mean'] = float(np.mean(ch))
                features[f'{channel}_std'] = float(np.std(ch))
                features[f'{channel}_skew'] = float(self._skewness(ch.flatten()))
                features[f'{channel}_kurtosis'] = float(self._kurtosis(ch.flatten()))
                features[f'{channel}_entropy'] = float(self._calculate_entropy(
                    np.histogram(ch, bins=256, range=(0, 256))[0]
                ))

            # 2. Cross-channel correlations (safe)
            r, g, b = img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()
            features['rg_correlation'] = float(self._safe_corrcoef(r, g))
            features['rb_correlation'] = float(self._safe_corrcoef(r, b))
            features['gb_correlation'] = float(self._safe_corrcoef(g, b))

            # 3. HSV color space features
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            for i, channel in enumerate(['h', 's', 'v']):
                ch = hsv[:, :, i]
                features[f'hsv_{channel}_mean'] = float(np.mean(ch))
                features[f'hsv_{channel}_std'] = float(np.std(ch))

            # Hue circular statistics
            hue_rad = hsv[:, :, 0] * 2 * np.pi / 180
            features['hue_circular_mean'] = float(np.arctan2(np.mean(np.sin(hue_rad)), np.mean(np.cos(hue_rad))))
            features['hue_circular_variance'] = float(1 - np.sqrt(np.mean(np.cos(hue_rad))**2 + np.mean(np.sin(hue_rad))**2))

            # 4. Lab color space features
            try:
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                for i, channel in enumerate(['l', 'a', 'b']):
                    ch = lab[:, :, i]
                    features[f'lab_{channel}_mean'] = float(np.mean(ch))
                    features[f'lab_{channel}_std'] = float(np.std(ch))
            except:
                pass

            # 5. Color distribution features
            # Color diversity (number of unique colors)
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            total_pixels = img.shape[0] * img.shape[1]
            features['color_diversity'] = float(self._safe_div(unique_colors, total_pixels))

            # Dominant color analysis
            pixels = img.reshape(-1, 3)
            from sklearn.cluster import KMeans
            try:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                kmeans.fit(pixels)

                # Color cluster statistics
                features['dominant_color_variance'] = float(np.var(kmeans.cluster_centers_))
                features['color_cluster_inertia'] = float(kmeans.inertia_)

            except:
                pass

        except Exception as e:
            logger.warning(f"Color statistics extraction failed: {e}")

        return features

    def _extract_texture_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features"""
        features: Dict[str, float] = {}

        try:
            # 1. Local Binary Pattern (LBP)
            lbp = feature.local_binary_pattern(gray, 24, 3, method='uniform')
            lbp_hist = np.histogram(lbp, bins=26)[0]
            lbp_sum = np.sum(lbp_hist)
            features['lbp_uniformity'] = float(self._safe_div(np.sum(lbp_hist[:25]), lbp_sum))
            features['lbp_entropy'] = float(self._calculate_entropy(lbp_hist))

            # 2. Gray Level Co-occurrence Matrix (GLCM)
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

            glcm_features = []
            for distance in distances:
                for angle in angles:
                    try:
                        glcm = feature.graycomatrix(gray, [distance], [angle],
                                                  levels=256, symmetric=True, normed=True)

                        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
                        dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
                        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
                        energy = feature.graycoprops(glcm, 'energy')[0, 0]
                        correlation = feature.graycoprops(glcm, 'correlation')[0, 0]

                        glcm_features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
                    except:
                        continue

            if glcm_features:
                features['glcm_contrast_mean'] = float(np.mean([f for i, f in enumerate(glcm_features) if i % 5 == 0]))
                features['glcm_dissimilarity_mean'] = float(np.mean([f for i, f in enumerate(glcm_features) if i % 5 == 1]))
                features['glcm_homogeneity_mean'] = float(np.mean([f for i, f in enumerate(glcm_features) if i % 5 == 2]))
                features['glcm_energy_mean'] = float(np.mean([f for i, f in enumerate(glcm_features) if i % 5 == 3]))
                features['glcm_correlation_mean'] = float(np.mean([f for i, f in enumerate(glcm_features) if i % 5 == 4]))

            # 3. Gabor filter responses
            gabor_responses = []
            for theta in [0, 45, 90, 135]:
                for frequency in [0.1, 0.3, 0.5]:
                    try:
                        real, _ = filters.gabor(gray, frequency=frequency, theta=np.deg2rad(theta))
                        gabor_responses.append(np.mean(np.abs(real)))
                        gabor_responses.append(np.std(real))
                    except:
                        continue

            if gabor_responses:
                features['gabor_mean_response'] = float(np.mean(gabor_responses))
                features['gabor_std_response'] = float(np.std(gabor_responses))
                features['gabor_max_response'] = float(np.max(gabor_responses))

            # 4. Haralick texture features
            try:
                from mahotas import features as mh_features
                haralick = mh_features.haralick(gray)
                if haralick.size > 0:
                    features['haralick_mean'] = float(np.mean(haralick))
                    features['haralick_std'] = float(np.std(haralick))
            except ImportError:
                pass

            # 5. Fractal dimension
            try:
                fractal_dim = self._calculate_fractal_dimension(gray)
                features['fractal_dimension'] = float(fractal_dim)
            except:
                pass

        except Exception as e:
            logger.warning(f"Texture feature extraction failed: {e}")

        return features

    def _extract_compression_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract compression artifact features"""
        features: Dict[str, float] = {}

        try:
            # 1. JPEG compression analysis
            from src.utils.ela import error_level_analysis

            # ELA at different quality levels
            for quality in [70, 85, 95]:
                try:
                    ela_img, ela_mean = error_level_analysis(image, quality=quality)
                    features[f'ela_mean_q{quality}'] = ela_mean

                    # ELA statistics
                    ela_array = np.array(ela_img)
                    features[f'ela_std_q{quality}'] = float(np.std(ela_array))
                    features[f'ela_max_q{quality}'] = float(np.max(ela_array))
                except:
                    continue

            # 2. Double JPEG compression detection
            double_jpeg_score = self._detect_double_jpeg_compression(image)
            features['double_jpeg_score'] = double_jpeg_score

            # 3. Block artifact detection
            block_artifacts = self._detect_block_artifacts(image)
            features.update(block_artifacts)

            # 4. Quantization table analysis
            quant_features = self._analyze_quantization_tables(image)
            features.update(quant_features)

        except Exception as e:
            logger.warning(f"Compression feature extraction failed: {e}")

        return features

    def _extract_geometric_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract geometric and structural features"""
        features: Dict[str, float] = {}

        try:
            # 1. Edge analysis
            edges_canny = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges_canny > 0) / edges_canny.size)

            # Edge direction histogram
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_angles = np.arctan2(grad_y, grad_x)
            edge_hist = np.histogram(edge_angles, bins=8)[0]
            features['edge_direction_entropy'] = float(self._calculate_entropy(edge_hist))

            # 2. Contour analysis
            contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
                if areas:
                    features['contour_count'] = len(areas)
                    features['contour_mean_area'] = float(np.mean(areas))
                    features['contour_area_std'] = float(np.std(areas))
                    features['contour_area_max'] = float(np.max(areas))

            # 3. Symmetry analysis
            features['horizontal_symmetry'] = float(self._compute_symmetry(gray, axis=0))
            features['vertical_symmetry'] = float(self._compute_symmetry(gray, axis=1))

            # 4. Structural similarity patterns
            # Divide image into blocks and analyze similarity
            block_similarities = self._analyze_block_similarities(gray)
            features.update(block_similarities)

        except Exception as e:
            logger.warning(f"Geometric feature extraction failed: {e}")

        return features

    def _extract_cnn_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract high-level CNN features"""
        features: Dict[str, float] = {}

        try:
            if self.cnn_extractor is None:
                return features

            # Prepare image tensor
            input_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                cnn_features = self.cnn_extractor(input_tensor)

            # The extractor may return a Tensor or an ndarray-like object
            if isinstance(cnn_features, torch.Tensor):
                cnn_features = cnn_features.squeeze().cpu().numpy()
            else:
                cnn_features = np.asarray(cnn_features).squeeze()

            flat = np.asarray(cnn_features).ravel()
            if flat.size == 0:
                return features

            # Statistical summary of CNN features
            features['cnn_mean'] = float(np.mean(flat))
            features['cnn_std'] = float(np.std(flat))
            features['cnn_max'] = float(np.max(flat))
            features['cnn_min'] = float(np.min(flat))
            features['cnn_l2_norm'] = float(np.linalg.norm(flat))

            # Sparsity measure
            features['cnn_sparsity'] = float(np.sum(flat == 0) / len(flat))

            # Energy in different percentiles
            sorted_features = np.sort(np.abs(flat))[::-1]
            total_energy = float(np.sum(sorted_features ** 2))
            if total_energy > 0:
                for percentile in [10, 25, 50, 75, 90]:
                    idx = max(1, int(len(sorted_features) * percentile / 100))
                    energy = float(np.sum(sorted_features[:idx] ** 2))
                    features[f'cnn_energy_top_{percentile}pct'] = float(self._safe_div(energy, total_energy))

        except Exception as e:
            logger.warning(f"CNN feature extraction failed: {e}")

        return features

    def _flatten_features(self, structured_features: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested feature dictionary"""
        flattened = {}

        def _flatten_dict(d, prefix=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    _flatten_dict(value, f"{prefix}{key}_" if prefix else f"{key}_")
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    flattened[f"{prefix}{key}"] = float(value)

        # Skip metadata when flattening
        for category, features in structured_features.items():
            if category != 'metadata' and isinstance(features, dict):
                _flatten_dict(features, f"{category}_")

        return flattened

    # Helper methods
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_entropy(self, histogram: np.ndarray) -> float:
        """Calculate entropy of histogram"""
        hist_norm = histogram / (np.sum(histogram) + 1e-8)
        hist_norm = hist_norm[hist_norm > 0]
        return -np.sum(hist_norm * np.log2(hist_norm + 1e-8))

    def _compute_radial_profile(self, image: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        """Compute radial profile of image"""
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)

        tbin = np.bincount(r.ravel(), image.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / (nr + 1e-8)

        return radialprofile

    def _compute_spectrum_slope(self, radial_profile: np.ndarray) -> float:
        """Compute slope of power spectrum"""
        try:
            # Fit line to log-log plot
            x = np.arange(1, len(radial_profile))
            y = radial_profile[1:]

            # Remove zeros and take log
            valid_idx = y > 0
            if np.sum(valid_idx) < 2:
                return 0.0
            # Clip to avoid log(0) or negative values which can cause warnings
            eps = 1e-8
            log_x = np.log(np.clip(x[valid_idx], eps, None))
            log_y = np.log(np.clip(y[valid_idx], eps, None))

            # Linear regression
            slope = np.polyfit(log_x, log_y, 1)[0]
            return slope

        except:
            return 0.0

    def _compute_symmetry(self, img: np.ndarray, axis: int) -> float:
        """Compute symmetry along axis"""
        try:
            if axis == 0:  # horizontal
                top = img[:img.shape[0]//2, :]
                bottom = np.flipud(img[img.shape[0]//2:, :])
            else:  # vertical
                left = img[:, :img.shape[1]//2]
                right = np.fliplr(img[:, img.shape[1]//2:])

            # Ensure same dimensions
            min_dim = min(top.shape[0] if axis == 0 else left.shape[1],
                         bottom.shape[0] if axis == 0 else right.shape[1])

            if axis == 0:
                top = top[:min_dim, :]
                bottom = bottom[:min_dim, :]
                correlation = self._safe_corrcoef(top.flatten(), bottom.flatten())
            else:
                left = left[:, :min_dim]
                right = right[:, :min_dim]
                correlation = self._safe_corrcoef(left.flatten(), right.flatten())

            return correlation if not np.isnan(correlation) else 0.0

        except:
            return 0.0

    def _detect_double_jpeg_compression(self, image: Image.Image) -> float:
        """Detect double JPEG compression artifacts"""
        try:
            import io

            # Compress at different qualities and analyze DCT coefficients
            qualities = [75, 85, 95]
            dct_histograms = []

            for quality in qualities:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                compressed = Image.open(buffer)

                # Convert to grayscale and compute DCT
                gray = np.array(compressed.convert('L'))
                dct = cv2.dct(gray.astype(np.float32))

                # Analyze DCT coefficient histogram
                hist = np.histogram(dct.flatten(), bins=50)[0]
                dct_histograms.append(hist)

            # Compare histograms to detect double compression
            if len(dct_histograms) >= 2:
                # Calculate histogram differences
                diff1 = np.sum(np.abs(dct_histograms[0] - dct_histograms[1]))
                diff2 = np.sum(np.abs(dct_histograms[1] - dct_histograms[2]))

                # Double compression typically shows specific patterns
                double_jpeg_indicator = abs(diff1 - diff2) / (diff1 + diff2 + 1e-8)
                return float(double_jpeg_indicator)

            return 0.0

        except:
            return 0.0

    def _detect_block_artifacts(self, image: Image.Image) -> Dict[str, float]:
        """Detect JPEG block artifacts"""
        features: Dict[str, float] = {}

        try:
            gray = np.array(image.convert('L'))

            # Analyze 8x8 block boundaries (JPEG standard)
            block_size = 8
            h, w = gray.shape

            # Horizontal block boundaries
            h_boundaries = []
            for i in range(block_size, h, block_size):
                if i < h:
                    boundary_diff = np.mean(np.abs(gray[i, :] - gray[i-1, :]))
                    h_boundaries.append(boundary_diff)

            # Vertical block boundaries
            v_boundaries = []
            for j in range(block_size, w, block_size):
                if j < w:
                    boundary_diff = np.mean(np.abs(gray[:, j] - gray[:, j-1]))
                    v_boundaries.append(boundary_diff)

            if h_boundaries:
                features['horizontal_block_artifacts'] = float(np.mean(h_boundaries))
                features['horizontal_block_artifacts_std'] = float(np.std(h_boundaries))

            if v_boundaries:
                features['vertical_block_artifacts'] = float(np.mean(v_boundaries))
                features['vertical_block_artifacts_std'] = float(np.std(v_boundaries))

        except:
            pass

        return features

    def _analyze_quantization_tables(self, image: Image.Image) -> Dict[str, float]:
        """Analyze JPEG quantization tables if available"""
        features: Dict[str, float] = {}

        try:
            import io

            # Recompress and analyze quality indicators
            for quality in [50, 75, 95]:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality)
                size = len(buffer.getvalue())
                features[f'compressed_size_q{quality}'] = float(size)

            # Calculate compression ratios
            original_buffer = io.BytesIO()
            image.save(original_buffer, format='PNG')
            original_size = len(original_buffer.getvalue())

            for quality in [50, 75, 95]:
                ratio = features.get(f'compressed_size_q{quality}', 0.0) / (original_size + 1e-8)
                features[f'compression_ratio_q{quality}'] = float(ratio)

        except Exception:
            pass

        return features

    def _analyze_block_similarities(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze similarities between image blocks"""
        features: Dict[str, float] = {}

        try:
            block_size = 16
            h, w = gray.shape

            similarities = []

            # Compare adjacent blocks
            for i in range(0, max(0, h - block_size), block_size):
                for j in range(0, max(0, w - block_size), block_size):
                    block1 = gray[i:i+block_size, j:j+block_size]

                    # Compare with right neighbor
                    if j + 2*block_size < w:
                        block2 = gray[i:i+block_size, j+block_size:j+2*block_size]
                        similarity = self._safe_corrcoef(block1.flatten(), block2.flatten())
                        similarities.append(similarity)

                    # Compare with bottom neighbor
                    if i + 2*block_size < h:
                        block2 = gray[i+block_size:i+2*block_size, j:j+block_size]
                        similarity = self._safe_corrcoef(block1.flatten(), block2.flatten())
                        similarities.append(similarity)

            if similarities:
                features['block_similarity_mean'] = float(np.mean(similarities))
                features['block_similarity_std'] = float(np.std(similarities))
                features['block_similarity_max'] = float(np.max(similarities))

                # High similarity might indicate copy-paste or synthetic generation
                high_similarity_count = int(np.sum(np.array(similarities) > 0.8))
                features['high_similarity_block_ratio'] = float(self._safe_div(high_similarity_count, len(similarities)))

        except Exception:
            pass

        return features

    def _calculate_fractal_dimension(self, image: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            # Threshold image
            threshold = np.mean(image)
            binary = image > threshold

            # Box-counting
            sizes = [2, 4, 8, 16, 32]
            counts = []

            for size in sizes:
                # Count boxes containing edge pixels
                h, w = binary.shape
                count = 0

                for i in range(0, h, size):
                    for j in range(0, w, size):
                        box = binary[i:min(i+size, h), j:min(j+size, w)]
                        if np.any(box):
                            count += 1

                counts.append(count)

            # Fit line to log-log plot (clip to avoid log(0))
            eps = 1e-8
            log_sizes = np.log(np.clip(sizes, eps, None))
            log_counts = np.log(np.clip(counts, eps, None))

            # Linear regression
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            fractal_dim = -slope

            return fractal_dim

        except:
            return 1.5  # Default fractal dimension

# datetime already imported at top of file
