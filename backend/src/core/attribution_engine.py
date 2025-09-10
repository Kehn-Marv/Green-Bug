import numpy as np
import json
import os
from typing import Dict, List, Optional, Any, cast
from datetime import datetime
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class AttributionEngine:
    """Advanced attribution engine with ensemble methods"""

    def __init__(self, fingerprints_path: str, embeddings_path: str = None):
        self.fingerprints_path = fingerprints_path
        self.embeddings_path = (
            embeddings_path or fingerprints_path.replace('.json', '_embeddings.pkl')
        )
        self.logger = logger

        # Explicitly type the fingerprints_db to avoid mypy inferring 'object' later
        from typing import Any, Dict
        self.fingerprints_db: Dict[str, Any] = {}

        # Initialize components
        self.closed_set_classifier = ClosedSetClassifier()
        # EmbeddingIndex will manage its own typed attributes
        self.embedding_index = EmbeddingIndex(self.embeddings_path)
        self.open_set_detector = OpenSetDetector()

        # Load or create fingerprint DB
        self._load_fingerprints()

    def _load_fingerprints(self) -> None:
        """Load fingerprint database"""
        try:
            if os.path.exists(self.fingerprints_path):
                with open(self.fingerprints_path, 'r') as f:
                    self.fingerprints_db = json.load(f)
            else:
                self.fingerprints_db = self._create_default_db()
                self._save_fingerprints()

            self.logger.info(
                f"Loaded {len(self.fingerprints_db.get('families', []))} attribution families"
            )

        except Exception as e:
            self.logger.error(f"Failed to load fingerprints: {e}")
            # Ensure we always have a dict with families list
            self.fingerprints_db = self._create_default_db()

    def _create_default_db(self) -> Dict:
        """Create default fingerprints database with advanced families"""
        return {
            "version": 2,
            "created": datetime.now().isoformat(),
            "families": [
                {
                    "name": "stylegan2_ffhq",
                    "type": "gan",
                    "description": "StyleGAN2 trained on FFHQ dataset",
                    "features_mean": {
                        "fft_high_freq_ratio": 0.75,
                        "noise_variance": 45.2,
                        "r_mean": 128.5,
                        "glcm_contrast": 0.65,
                        "ela_mean_q85": 12.3
                    },
                    "features_std": {
                        "fft_high_freq_ratio": 0.05,
                        "noise_variance": 8.1,
                        "r_mean": 15.2,
                        "glcm_contrast": 0.12,
                        "ela_mean_q85": 3.4
                    },
                    "sample_count": 0,
                    "confidence_threshold": 0.85,
                    "last_updated": None
                },
                {
                    "name": "deepfakes_autoencoder",
                    "type": "autoencoder",
                    "description": "Classic DeepFakes autoencoder architecture",
                    "features_mean": {
                        "fft_high_freq_ratio": 0.62,
                        "noise_variance": 38.7,
                        "r_mean": 125.8,
                        "glcm_contrast": 0.58,
                        "ela_mean_q85": 18.5
                    },
                    "features_std": {
                        "fft_high_freq_ratio": 0.08,
                        "noise_variance": 12.3,
                        "r_mean": 18.7,
                        "glcm_contrast": 0.15,
                        "ela_mean_q85": 5.2
                    },
                    "sample_count": 0,
                    "confidence_threshold": 0.80,
                    "last_updated": None
                },
                {
                    "name": "first_order_motion",
                    "type": "motion_transfer",
                    "description": "First Order Motion Model for face animation",
                    "features_mean": {
                        "fft_high_freq_ratio": 0.68,
                        "noise_variance": 42.1,
                        "r_mean": 130.2,
                        "glcm_contrast": 0.61,
                        "ela_mean_q85": 15.7
                    },
                    "features_std": {
                        "fft_high_freq_ratio": 0.06,
                        "noise_variance": 9.8,
                        "r_mean": 16.4,
                        "glcm_contrast": 0.13,
                        "ela_mean_q85": 4.1
                    },
                    "sample_count": 0,
                    "confidence_threshold": 0.82,
                    "last_updated": None
                },
                {
                    "name": "diffusion_inpainting",
                    "type": "diffusion",
                    "description": "Diffusion model based face inpainting",
                    "features_mean": {
                        "fft_high_freq_ratio": 0.71,
                        "noise_variance": 35.4,
                        "r_mean": 132.1,
                        "glcm_contrast": 0.59,
                        "ela_mean_q85": 11.2
                    },
                    "features_std": {
                        "fft_high_freq_ratio": 0.04,
                        "noise_variance": 7.6,
                        "r_mean": 14.8,
                        "glcm_contrast": 0.11,
                        "ela_mean_q85": 2.9
                    },
                    "sample_count": 0,
                    "confidence_threshold": 0.88,
                    "last_updated": None
                }
            ]
        }

    def analyze_attribution(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Comprehensive attribution analysis using ensemble methods"""
        try:
            # Explicitly type results and methods so mypy allows indexed assignments
            methods: Dict[str, Any] = {}
            results: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "feature_count": len(features),
                "methods": methods
            }

            # 1. Closed-set classification
            closed_set_result = self.closed_set_classifier.classify(features, self.fingerprints_db)
            results["methods"]["closed_set"] = closed_set_result

            # 2. Embedding-based matching
            embedding_result = self.embedding_index.find_matches(features)
            results["methods"]["embedding_match"] = embedding_result

            # 3. Open-set detection
            open_set_result = self.open_set_detector.detect_novelty(features, self.fingerprints_db)
            results["methods"]["open_set"] = open_set_result

            # 4. Ensemble decision
            ensemble_result = self._ensemble_decision(closed_set_result, embedding_result, open_set_result)
            results["ensemble"] = ensemble_result

            self.logger.debug(f"Attribution analysis complete: {ensemble_result['predicted_family']}")
            return results

        except Exception as e:
            self.logger.error(f"Attribution analysis failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _ensemble_decision(self, closed_set: Dict, embedding: Dict, open_set: Dict) -> Dict[str, Any]:
        """Combine results from all attribution methods"""
        try:
            # Weight the different methods
            weights = {
                "closed_set": 0.4,
                "embedding": 0.4,
                "open_set": 0.2
            }

            # Aggregate confidence scores
            family_scores: Dict[str, float] = {}

            # From closed-set classifier
            if closed_set.get("predictions"):
                for pred in closed_set["predictions"]:
                    family = pred["family"]
                    confidence = pred["confidence"]
                    if family not in family_scores:
                        family_scores[family] = 0
                    family_scores[family] += weights["closed_set"] * confidence

            # From embedding matches
            if embedding.get("matches"):
                for match in embedding["matches"]:
                    family = match["family"]
                    similarity = match["similarity"]
                    if family not in family_scores:
                        family_scores[family] = 0
                    family_scores[family] += weights["embedding"] * similarity

            # Adjust for open-set detection
            novelty_score = open_set.get("novelty_score", 0.5)
            if novelty_score > 0.7:  # High novelty - reduce all scores
                family_scores = {k: v * (1 - novelty_score * 0.5) for k, v in family_scores.items()}

            # Determine final prediction
            if family_scores:
                # Use explicit key lambda so mypy can reason about types
                best_family = max(family_scores.keys(), key=lambda k: family_scores[k])
                best_score = family_scores[best_family]

                return {
                    "predicted_family": best_family,
                    "confidence": float(best_score),
                    "all_scores": family_scores,
                    "novelty_detected": novelty_score > 0.7,
                    "novelty_score": float(novelty_score),
                    "method_weights": weights
                }
            else:
                return {
                    "predicted_family": "unknown",
                    "confidence": 0.0,
                    "all_scores": {},
                    "novelty_detected": True,
                    "novelty_score": float(novelty_score),
                    "method_weights": weights
                }

        except Exception as e:
            self.logger.error(f"Ensemble decision failed: {e}")
            return {"predicted_family": "error", "confidence": 0.0, "error": str(e)}

    def update_fingerprints(self, family_name: str, features: Dict[str, float],
                          learning_rate: float = 0.1) -> bool:
        """Update fingerprints with new sample using adaptive learning"""
        try:
            family: Optional[Dict[str, Any]] = None
            fingerprints: Dict[str, Any] = cast(Dict[str, Any], self.fingerprints_db)
            for f in fingerprints.get("families", []):
                # Ensure mypy understands the family is a dict so indexed assignment is allowed
                if f["name"] == family_name:
                    family = cast(Dict[str, Any], f)
                    break

            if family is None:
                # Create new family
                family = {
                    "name": family_name,
                    "type": "unknown",
                    "description": f"Auto-discovered family: {family_name}",
                    "features_mean": {},
                    "features_std": {},
                    "sample_count": 0,
                    "confidence_threshold": 0.75,
                    "last_updated": None
                }
                fingerprints.setdefault("families", []).append(family)
                self.fingerprints_db = fingerprints

            # Update statistics with adaptive learning rate
            n = family["sample_count"]
            adaptive_lr = learning_rate / (1 + n * 0.01)  # Decrease learning rate over time

            for feature_name, value in features.items():
                if feature_name in family["features_mean"]:
                    # Update mean with adaptive learning
                    old_mean = family["features_mean"][feature_name]
                    new_mean = old_mean + adaptive_lr * (value - old_mean)
                    family["features_mean"][feature_name] = new_mean

                    # Update standard deviation
                    if feature_name in family["features_std"]:
                        old_std = family["features_std"][feature_name]
                        diff_sq = (value - new_mean) ** 2
                        new_var = old_std ** 2 + adaptive_lr * (diff_sq - old_std ** 2)
                        family["features_std"][feature_name] = np.sqrt(max(new_var, 0.01))
                    else:
                        family["features_std"][feature_name] = 1.0
                else:
                    # New feature
                    family["features_mean"][feature_name] = value
                    family["features_std"][feature_name] = 1.0

            family["sample_count"] = n + 1
            family["last_updated"] = datetime.now().isoformat()

            # Update embedding index
            self.embedding_index.add_sample(family_name, features)

            self._save_fingerprints()
            self.logger.info(f"Updated fingerprints for family {family_name}: {n + 1} samples")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update fingerprints: {e}")
            return False

    def _save_fingerprints(self):
        """Save fingerprints database"""
        try:
            os.makedirs(os.path.dirname(self.fingerprints_path), exist_ok=True)
            with open(self.fingerprints_path, 'w') as f:
                json.dump(self.fingerprints_db, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save fingerprints: {e}")


class ClosedSetClassifier:
    """Closed-set classifier for known deepfake families"""

    def __init__(self):
        self.logger = logger

    def classify(self, features: Dict[str, float], fingerprints_db: Dict) -> Dict[str, Any]:
        """Classify against known families using statistical distance"""
        try:
            predictions = []

            for family in fingerprints_db.get("families", []):
                confidence = self._compute_likelihood(features, family)
                predictions.append({
                    "family": family["name"],
                    "type": family.get("type", "unknown"),
                    "confidence": confidence,
                    "threshold": family.get("confidence_threshold", 0.8)
                })

            # Sort by confidence
            predictions.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "method": "closed_set_statistical",
                "predictions": predictions[:5],  # Top 5
                "best_match": predictions[0] if predictions else None
            }

        except Exception as e:
            self.logger.error(f"Closed-set classification failed: {e}")
            return {"method": "closed_set_statistical", "error": str(e)}

    def _compute_likelihood(self, features: Dict[str, float], family: Dict) -> float:
        """Compute likelihood using Mahalanobis-like distance"""
        try:
            family_mean = family.get("features_mean", {})
            family_std = family.get("features_std", {})

            if not family_mean:
                return 0.0

            # Find common features
            common_features = set(features.keys()) & set(family_mean.keys())
            if not common_features:
                return 0.0

            # Compute normalized distance
            distances = []
            for feature in common_features:
                mean_val = family_mean[feature]
                std_val = family_std.get(feature, 1.0)
                observed_val = features[feature]

                # Normalized distance
                norm_dist = abs(observed_val - mean_val) / (std_val + 1e-8)
                distances.append(norm_dist)

            # Convert distance to likelihood (higher distance = lower likelihood)
            avg_distance = np.mean(distances)
            likelihood = np.exp(-avg_distance)  # Exponential decay

            return float(likelihood)

        except Exception as e:
            self.logger.warning(f"Likelihood computation failed: {e}")
            return 0.0


class EmbeddingIndex:
    """FAISS-based embedding index for fast similarity search"""
    def __init__(self, embeddings_path: str):
        self.embeddings_path = embeddings_path
        self.logger = logger
        # FAISS index (or None when not available)
        self.index: Optional[Any] = None
        # Metadata
        self.family_labels: List[str] = []
        self.feature_names: List[str] = []
        self.scaler = StandardScaler()
        self.faiss_available = FAISS_AVAILABLE

        # Load or initialize index
        self._load_index()

    def _load_index(self) -> None:
        """Load an existing index from disk, or initialize empty metadata."""
        try:
            if os.path.exists(self.embeddings_path):
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.index = data.get('index')
                    self.family_labels = data.get('family_labels', [])
                    self.feature_names = data.get('feature_names', [])
                    self.scaler = data.get('scaler', StandardScaler())

                self.logger.info(f"Loaded embedding index with {len(self.family_labels)} samples")
            else:
                self._create_empty_index()
        except Exception as e:
            self.logger.warning(f"Failed to load embedding index: {e}")
            self._create_empty_index()

    def _create_empty_index(self) -> None:
        """Reset index metadata to empty defaults."""
        self.index = None
        self.family_labels = []
        self.feature_names = []
        self.scaler = StandardScaler()

    def find_matches(self, features: Dict[str, float], k: int = 5) -> Dict[str, Any]:
        """Find similar samples using FAISS (if available). Returns a dict with matches."""
        try:
            if not self.faiss_available:
                return {"method": "embedding_faiss", "matches": [], "note": "FAISS not available"}

            if self.index is None or len(self.family_labels) == 0:
                return {"method": "embedding_faiss", "matches": [], "note": "Empty index"}

            query_vector = self._features_to_vector(features)
            if query_vector is None:
                return {"method": "embedding_faiss", "matches": [], "error": "Feature mismatch"}

            similarities, indices = self.index.search(query_vector.reshape(1, -1), min(k, len(self.family_labels)))

            matches: List[Dict[str, Any]] = []
            for sim, idx in zip(similarities[0], indices[0]):
                if 0 <= int(idx) < len(self.family_labels):
                    matches.append({
                        "family": self.family_labels[int(idx)],
                        "similarity": float(sim),
                        "index": int(idx)
                    })

            return {
                "method": "embedding_faiss",
                "matches": matches,
                "query_dimension": len(query_vector)
            }
        except Exception as e:
            self.logger.error(f"Embedding search failed: {e}")
            return {"method": "embedding_faiss", "error": str(e)}

    def add_sample(self, family_name: str, features: Dict[str, float]) -> bool:
        """Add new sample to the index (if FAISS is available)."""
        try:
            if not self.faiss_available:
                return False

            if not self.feature_names:
                self.feature_names = sorted(features.keys())
                self._rebuild_index()

            vector = self._features_to_vector(features)
            if vector is None:
                return False

            if self.index is None:
                dimension = len(vector)
                # type: ignore[attr-defined]
                self.index = faiss.IndexFlatIP(dimension)

            vector_norm = vector / (np.linalg.norm(vector) + 1e-8)
            self.index.add(vector_norm.reshape(1, -1).astype('float32'))
            self.family_labels.append(family_name)

            self._save_index()
            return True
        except Exception as e:
            self.logger.error(f"Failed to add sample to index: {e}")
            return False

    def _features_to_vector(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Convert ordered feature dict to numpy vector matching feature_names."""
        try:
            if not self.feature_names:
                return None

            vector = [features.get(name, 0.0) for name in self.feature_names]
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Feature vectorization failed: {e}")
            return None

    def _rebuild_index(self) -> None:
        """Reset the in-memory index when features change."""
        self.index = None
        self.family_labels = []

    def _save_index(self) -> None:
        """Persist index metadata to disk (index object may or may not be serializable)."""
        try:
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            data = {
                'index': self.index,
                'family_labels': self.family_labels,
                'feature_names': self.feature_names,
                'scaler': self.scaler
            }
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save embedding index: {e}")


class OpenSetDetector:
    """Open-set detector for unknown/novel deepfake families"""

    def __init__(self):
        self.logger = logger
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False

    def detect_novelty(self, features: Dict[str, float], fingerprints_db: Dict) -> Dict[str, Any]:
        """Detect if sample belongs to unknown family"""
        try:
            # Fit model if not already fitted
            if not self.is_fitted:
                self._fit_model(fingerprints_db)

            # Convert features to vector
            feature_vector = self._prepare_features(features, fingerprints_db)
            if feature_vector is None:
                return {"method": "open_set_isolation", "novelty_score": 0.5, "error": "Feature preparation failed"}

            # Predict novelty
            novelty_score = self.isolation_forest.decision_function([feature_vector])[0]
            is_outlier = self.isolation_forest.predict([feature_vector])[0] == -1

            # Normalize score to [0, 1] where 1 = high novelty
            normalized_score = max(0, min(1, (0.5 - novelty_score) / 1.0))

            return {
                "method": "open_set_isolation",
                "novelty_score": float(normalized_score),
                "is_outlier": bool(is_outlier),
                "raw_score": float(novelty_score),
                "threshold": 0.7
            }

        except Exception as e:
            self.logger.error(f"Novelty detection failed: {e}")
            return {"method": "open_set_isolation", "error": str(e), "novelty_score": 0.5}

    def _fit_model(self, fingerprints_db: Dict):
        """Fit isolation forest on known family features"""
        try:
            training_data = []

            for family in fingerprints_db.get("families", []):
                features_mean = family.get("features_mean", {})
                if features_mean:
                    # Create synthetic samples around the mean
                    for _ in range(10):  # Generate multiple samples per family
                        sample = []
                        for feature_name in sorted(features_mean.keys()):
                            mean_val = features_mean[feature_name]
                            std_val = family.get("features_std", {}).get(feature_name, mean_val * 0.1)
                            # Add noise
                            noisy_val = mean_val + np.random.normal(0, std_val * 0.1)
                            sample.append(noisy_val)
                        training_data.append(sample)

            if training_data:
                self.isolation_forest.fit(training_data)
                self.is_fitted = True
                self.logger.info(f"Fitted open-set detector on {len(training_data)} synthetic samples")

        except Exception as e:
            self.logger.error(f"Failed to fit open-set detector: {e}")

    def _prepare_features(self, features: Dict[str, float], fingerprints_db: Dict) -> Optional[List[float]]:
        """Prepare features for novelty detection"""
        try:
            # Get all known feature names
            all_features = set()
            for family in fingerprints_db.get("families", []):
                all_features.update(family.get("features_mean", {}).keys())

            if not all_features:
                return None

            # Create feature vector
            feature_vector = []
            for feature_name in sorted(all_features):
                feature_vector.append(features.get(feature_name, 0.0))

            return feature_vector

        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return None
