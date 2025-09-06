import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
from enum import Enum
import hashlib

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class LearningStrategy(Enum):
    UNCERTAINTY_SAMPLING = "uncertainty"
    DIVERSITY_SAMPLING = "diversity"
    HYBRID_SAMPLING = "hybrid"

@dataclass
class LearningCandidate:
    """Candidate sample for active learning"""
    image_hash: str
    features: Dict[str, float]
    prediction_confidence: float
    uncertainty_score: float
    diversity_score: float
    quality_score: float
    timestamp: str
    user_consent: bool = False
    human_label: Optional[str] = None
    metadata: Dict[str, Any] = None

class SelfLearningEngine:
    """Advanced self-learning system with active learning and safeguards"""
    
    def __init__(self, learning_db_path: str, max_candidates: int = 1000):
        self.learning_db_path = learning_db_path
        self.max_candidates = max_candidates
        self.logger = logger
        
        # Learning parameters
        self.uncertainty_threshold = 0.3  # High uncertainty samples
        self.diversity_threshold = 0.7    # Diverse samples
        self.quality_threshold = 0.8      # High quality samples
        self.consent_required = True      # Require user consent
        
        # Load learning database
        self.learning_db = self._load_learning_db()
        
        # Initialize active learning components
        self.candidate_selector = CandidateSelector()
        self.quality_assessor = QualityAssessor()
        self.consent_manager = ConsentManager()
        
    def _load_learning_db(self) -> Dict[str, Any]:
        """Load learning database"""
        try:
            if os.path.exists(self.learning_db_path):
                with open(self.learning_db_path, 'r') as f:
                    db = json.load(f)
                self.logger.info(f"Loaded learning database with {len(db.get('candidates', []))} candidates")
                return db
            else:
                return self._create_empty_db()
                
        except Exception as e:
            self.logger.error(f"Failed to load learning database: {e}")
            return self._create_empty_db()
    
    def _create_empty_db(self) -> Dict[str, Any]:
        """Create empty learning database"""
        return {
            "version": 1,
            "created": datetime.now().isoformat(),
            "candidates": [],
            "learned_samples": [],
            "statistics": {
                "total_candidates": 0,
                "total_learned": 0,
                "consent_rate": 0.0,
                "quality_rate": 0.0
            },
            "settings": {
                "uncertainty_threshold": self.uncertainty_threshold,
                "diversity_threshold": self.diversity_threshold,
                "quality_threshold": self.quality_threshold,
                "consent_required": self.consent_required
            }
        }
    
    def evaluate_learning_candidate(self, image_data: bytes, features: Dict[str, float], 
                                  prediction_result: Dict[str, Any], 
                                  user_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate if sample should be considered for learning"""
        try:
            # Generate image hash for deduplication
            image_hash = hashlib.sha256(image_data).hexdigest()
            
            # Check if already processed
            if self._is_duplicate(image_hash):
                return {
                    "selected": False,
                    "reason": "duplicate",
                    "image_hash": image_hash
                }
            
            # Assess quality
            quality_score = self.quality_assessor.assess_quality(features, prediction_result)
            
            if quality_score < self.quality_threshold:
                return {
                    "selected": False,
                    "reason": "low_quality",
                    "quality_score": quality_score,
                    "image_hash": image_hash
                }
            
            # Calculate uncertainty score
            uncertainty_score = self._calculate_uncertainty(prediction_result)
            
            # Calculate diversity score
            diversity_score = self._calculate_diversity(features)
            
            # Determine if candidate should be selected
            selected = self.candidate_selector.should_select(
                uncertainty_score, diversity_score, quality_score
            )
            
            if selected:
                # Create learning candidate
                candidate = LearningCandidate(
                    image_hash=image_hash,
                    features=features,
                    prediction_confidence=prediction_result.get("confidence", 0.0),
                    uncertainty_score=uncertainty_score,
                    diversity_score=diversity_score,
                    quality_score=quality_score,
                    timestamp=datetime.now().isoformat(),
                    user_consent=False,
                    metadata=user_metadata or {}
                )
                
                # Add to candidates
                self._add_candidate(candidate)
                
                self.logger.info(f"Selected learning candidate: {image_hash[:8]}")
                
                return {
                    "selected": True,
                    "candidate_id": image_hash,
                    "uncertainty_score": uncertainty_score,
                    "diversity_score": diversity_score,
                    "quality_score": quality_score,
                    "requires_consent": self.consent_required
                }
            else:
                return {
                    "selected": False,
                    "reason": "selection_criteria_not_met",
                    "scores": {
                        "uncertainty": uncertainty_score,
                        "diversity": diversity_score,
                        "quality": quality_score
                    },
                    "image_hash": image_hash
                }
                
        except Exception as e:
            self.logger.error(f"Learning candidate evaluation failed: {e}")
            return {"selected": False, "error": str(e)}
    
    def request_user_consent(self, candidate_id: str, consent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user consent for learning"""
        try:
            candidate = self._find_candidate(candidate_id)
            if not candidate:
                return {"success": False, "error": "Candidate not found"}
            
            # Validate consent data
            consent_valid = self.consent_manager.validate_consent(consent_data)
            if not consent_valid:
                return {"success": False, "error": "Invalid consent data"}
            
            # Update candidate with consent
            candidate["user_consent"] = True
            candidate["consent_data"] = consent_data
            candidate["consent_timestamp"] = datetime.now().isoformat()
            
            # Add human label if provided
            if "human_label" in consent_data:
                candidate["human_label"] = consent_data["human_label"]
            
            self._save_learning_db()
            
            self.logger.info(f"User consent received for candidate: {candidate_id[:8]}")
            
            return {
                "success": True,
                "candidate_id": candidate_id,
                "consent_timestamp": candidate["consent_timestamp"]
            }
            
        except Exception as e:
            self.logger.error(f"Consent processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_learning_batch(self, batch_size: int = 10, 
                          strategy: LearningStrategy = LearningStrategy.HYBRID_SAMPLING) -> List[Dict[str, Any]]:
        """Get batch of samples for retraining"""
        try:
            # Filter consented candidates
            consented_candidates = [
                c for c in self.learning_db["candidates"] 
                if c.get("user_consent", False) and not c.get("used_for_training", False)
            ]
            
            if not consented_candidates:
                return []
            
            # Select batch based on strategy
            if strategy == LearningStrategy.UNCERTAINTY_SAMPLING:
                batch = self._select_by_uncertainty(consented_candidates, batch_size)
            elif strategy == LearningStrategy.DIVERSITY_SAMPLING:
                batch = self._select_by_diversity(consented_candidates, batch_size)
            else:  # HYBRID_SAMPLING
                batch = self._select_hybrid(consented_candidates, batch_size)
            
            # Mark as used for training
            for candidate in batch:
                candidate["used_for_training"] = True
                candidate["training_timestamp"] = datetime.now().isoformat()
            
            self._save_learning_db()
            
            self.logger.info(f"Generated learning batch: {len(batch)} samples")
            return batch
            
        except Exception as e:
            self.logger.error(f"Learning batch generation failed: {e}")
            return []
    
    def update_learning_statistics(self, training_results: Dict[str, Any]):
        """Update learning statistics after training"""
        try:
            stats = self.learning_db["statistics"]
            
            # Update counters
            stats["total_learned"] += training_results.get("samples_trained", 0)
            
            # Update rates
            total_candidates = len(self.learning_db["candidates"])
            consented_candidates = len([c for c in self.learning_db["candidates"] if c.get("user_consent", False)])
            
            if total_candidates > 0:
                stats["consent_rate"] = consented_candidates / total_candidates
            
            # Update quality metrics
            if "quality_metrics" in training_results:
                stats.update(training_results["quality_metrics"])
            
            # Add training result
            self.learning_db["learned_samples"].append({
                "timestamp": datetime.now().isoformat(),
                "results": training_results
            })
            
            self._save_learning_db()
            
            self.logger.info("Learning statistics updated")
            
        except Exception as e:
            self.logger.error(f"Statistics update failed: {e}")
    
    def cleanup_old_candidates(self, days_old: int = 30):
        """Remove old candidates to manage storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            original_count = len(self.learning_db["candidates"])
            
            # Keep candidates that are recent or have consent
            self.learning_db["candidates"] = [
                c for c in self.learning_db["candidates"]
                if (datetime.fromisoformat(c["timestamp"]) > cutoff_date or 
                    c.get("user_consent", False))
            ]
            
            removed_count = original_count - len(self.learning_db["candidates"])
            
            if removed_count > 0:
                self._save_learning_db()
                self.logger.info(f"Cleaned up {removed_count} old candidates")
            
        except Exception as e:
            self.logger.error(f"Candidate cleanup failed: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        try:
            stats = self.learning_db["statistics"].copy()
            
            # Add current counts
            candidates = self.learning_db["candidates"]
            stats["current_candidates"] = len(candidates)
            stats["consented_candidates"] = len([c for c in candidates if c.get("user_consent", False)])
            stats["pending_candidates"] = len([c for c in candidates if not c.get("user_consent", False)])
            stats["trained_candidates"] = len([c for c in candidates if c.get("used_for_training", False)])
            
            # Add recent activity
            recent_date = datetime.now() - timedelta(days=7)
            recent_candidates = [
                c for c in candidates 
                if datetime.fromisoformat(c["timestamp"]) > recent_date
            ]
            stats["recent_candidates"] = len(recent_candidates)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistics retrieval failed: {e}")
            return {}
    
    def _calculate_uncertainty(self, prediction_result: Dict[str, Any]) -> float:
        """Calculate uncertainty score from prediction"""
        try:
            confidence = prediction_result.get("confidence", 0.5)
            
            # High uncertainty when confidence is close to 0.5 (decision boundary)
            uncertainty = 1.0 - abs(confidence - 0.5) * 2
            
            return float(uncertainty)
            
        except Exception:
            return 0.5
    
    def _calculate_diversity(self, features: Dict[str, float]) -> float:
        """Calculate diversity score compared to existing samples"""
        try:
            if not self.learning_db["candidates"]:
                return 1.0  # First sample is maximally diverse
            
            # Compare with existing candidates
            existing_features = [c["features"] for c in self.learning_db["candidates"]]
            
            # Calculate minimum distance to existing samples
            min_distance = float('inf')
            
            for existing in existing_features:
                distance = self._feature_distance(features, existing)
                min_distance = min(min_distance, distance)
            
            # Normalize distance to [0, 1] range
            diversity_score = min(1.0, min_distance / 10.0)  # Adjust scaling as needed
            
            return diversity_score
            
        except Exception:
            return 0.5
    
    def _feature_distance(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate distance between feature vectors"""
        try:
            common_keys = set(features1.keys()) & set(features2.keys())
            if not common_keys:
                return 1.0
            
            # Euclidean distance
            distance = 0.0
            for key in common_keys:
                distance += (features1[key] - features2[key]) ** 2
            
            return np.sqrt(distance / len(common_keys))
            
        except Exception:
            return 1.0
    
    def _is_duplicate(self, image_hash: str) -> bool:
        """Check if image hash already exists"""
        return any(c["image_hash"] == image_hash for c in self.learning_db["candidates"])
    
    def _add_candidate(self, candidate: LearningCandidate):
        """Add candidate to database"""
        candidate_dict = {
            "image_hash": candidate.image_hash,
            "features": candidate.features,
            "prediction_confidence": candidate.prediction_confidence,
            "uncertainty_score": candidate.uncertainty_score,
            "diversity_score": candidate.diversity_score,
            "quality_score": candidate.quality_score,
            "timestamp": candidate.timestamp,
            "user_consent": candidate.user_consent,
            "human_label": candidate.human_label,
            "metadata": candidate.metadata or {}
        }
        
        self.learning_db["candidates"].append(candidate_dict)
        
        # Maintain max candidates limit
        if len(self.learning_db["candidates"]) > self.max_candidates:
            # Remove oldest candidates without consent
            self.learning_db["candidates"] = sorted(
                self.learning_db["candidates"],
                key=lambda x: (x.get("user_consent", False), x["timestamp"]),
                reverse=True
            )[:self.max_candidates]
        
        self._save_learning_db()
    
    def _find_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Find candidate by ID"""
        for candidate in self.learning_db["candidates"]:
            if candidate["image_hash"] == candidate_id:
                return candidate
        return None
    
    def _select_by_uncertainty(self, candidates: List[Dict], batch_size: int) -> List[Dict]:
        """Select candidates with highest uncertainty"""
        return sorted(candidates, key=lambda x: x["uncertainty_score"], reverse=True)[:batch_size]
    
    def _select_by_diversity(self, candidates: List[Dict], batch_size: int) -> List[Dict]:
        """Select most diverse candidates"""
        return sorted(candidates, key=lambda x: x["diversity_score"], reverse=True)[:batch_size]
    
    def _select_hybrid(self, candidates: List[Dict], batch_size: int) -> List[Dict]:
        """Select candidates using hybrid strategy"""
        # Combine uncertainty and diversity scores
        for candidate in candidates:
            candidate["hybrid_score"] = (
                0.6 * candidate["uncertainty_score"] + 
                0.4 * candidate["diversity_score"]
            )
        
        return sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)[:batch_size]
    
    def _save_learning_db(self):
        """Save learning database"""
        try:
            os.makedirs(os.path.dirname(self.learning_db_path), exist_ok=True)
            with open(self.learning_db_path, 'w') as f:
                json.dump(self.learning_db, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save learning database: {e}")


class CandidateSelector:
    """Select candidates for active learning"""
    
    def should_select(self, uncertainty_score: float, diversity_score: float, 
                     quality_score: float) -> bool:
        """Determine if candidate should be selected"""
        # Weighted selection criteria
        selection_score = (
            0.4 * uncertainty_score +
            0.3 * diversity_score +
            0.3 * quality_score
        )
        
        return selection_score > 0.6


class QualityAssessor:
    """Assess quality of learning candidates"""
    
    def assess_quality(self, features: Dict[str, float], 
                      prediction_result: Dict[str, Any]) -> float:
        """Assess overall quality of the sample"""
        try:
            quality_factors = []
            
            # Feature completeness
            expected_features = 20  # Adjust based on your feature set
            completeness = len(features) / expected_features
            quality_factors.append(min(1.0, completeness))
            
            # Feature validity (no NaN, reasonable ranges)
            valid_features = sum(1 for v in features.values() 
                               if not np.isnan(v) and np.isfinite(v))
            validity = valid_features / len(features) if features else 0
            quality_factors.append(validity)
            
            # Prediction stability (if available)
            if "robustness" in prediction_result:
                stability = prediction_result["robustness"].get("overall_robustness", 0.5)
                quality_factors.append(stability)
            
            # Face detection quality (if available)
            if "face" in prediction_result:
                face_conf = prediction_result["face"].get("confidence", 0.0)
                quality_factors.append(face_conf)
            
            return np.mean(quality_factors)
            
        except Exception:
            return 0.5


class ConsentManager:
    """Manage user consent for learning"""
    
    def validate_consent(self, consent_data: Dict[str, Any]) -> bool:
        """Validate consent data"""
        try:
            required_fields = ["consent_given", "timestamp", "user_id"]
            
            # Check required fields
            for field in required_fields:
                if field not in consent_data:
                    return False
            
            # Validate consent is explicitly given
            if not consent_data.get("consent_given", False):
                return False
            
            # Validate timestamp is recent (within 1 hour)
            consent_time = datetime.fromisoformat(consent_data["timestamp"])
            if datetime.now() - consent_time > timedelta(hours=1):
                return False
            
            return True
            
        except Exception:
            return False