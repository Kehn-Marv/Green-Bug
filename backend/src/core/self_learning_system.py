import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import os
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class LearningStrategy(Enum):
    UNCERTAINTY_SAMPLING = "uncertainty"
    DIVERSITY_SAMPLING = "diversity"
    HYBRID_SAMPLING = "hybrid"
    QUALITY_BASED = "quality"

class ConsentStatus(Enum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"

@dataclass
class LearningCandidate:
    """Candidate sample for active learning"""
    id: str
    image_hash: str
    features: Dict[str, float]
    prediction_confidence: float
    uncertainty_score: float
    diversity_score: float
    quality_score: float
    timestamp: str
    consent_status: ConsentStatus = ConsentStatus.PENDING
    human_label: Optional[str] = None
    metadata: Dict[str, Any] = None
    user_id: Optional[str] = None
    consent_timestamp: Optional[str] = None
    used_for_training: bool = False
    training_timestamp: Optional[str] = None

@dataclass
class ConsentRequest:
    """User consent request"""
    candidate_id: str
    user_id: str
    consent_given: bool
    human_label: Optional[str] = None
    timestamp: str = None
    additional_data: Dict[str, Any] = None

class SelfLearningSystem:
    """Advanced self-learning system with active learning and comprehensive safeguards"""
    
    def __init__(self, learning_db_path: str, max_candidates: int = 1000):
        self.learning_db_path = learning_db_path
        self.max_candidates = max_candidates
        self.logger = logger
        self.lock = threading.Lock()
        
        # Learning parameters
        self.uncertainty_threshold = 0.3
        self.diversity_threshold = 0.7
        self.quality_threshold = 0.8
        self.consent_timeout_hours = 24
        
        # Safeguards
        self.max_daily_candidates = 50
        self.min_consent_rate = 0.3
        self.max_training_frequency_hours = 6
        
        # Load learning database
        self.learning_db = self._load_learning_db()
        
        # Initialize components
        self.candidate_selector = CandidateSelector()
        self.quality_assessor = QualityAssessor()
        self.consent_manager = ConsentManager()
        self.diversity_analyzer = DiversityAnalyzer()
        self.safeguard_monitor = SafeguardMonitor()
        
    def _load_learning_db(self) -> Dict[str, Any]:
        """Load learning database with migration support"""
        try:
            if os.path.exists(self.learning_db_path):
                with open(self.learning_db_path, 'r') as f:
                    db = json.load(f)
                
                # Migrate old format if needed
                db = self._migrate_database(db)
                
                self.logger.info(f"Loaded learning database with {len(db.get('candidates', []))} candidates")
                return db
            else:
                return self._create_empty_db()
                
        except Exception as e:
            self.logger.error(f"Failed to load learning database: {e}")
            return self._create_empty_db()
    
    def _create_empty_db(self) -> Dict[str, Any]:
        """Create empty learning database with comprehensive structure"""
        return {
            "version": 2,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "candidates": [],
            "learned_samples": [],
            "consent_requests": [],
            "training_sessions": [],
            "statistics": {
                "total_candidates": 0,
                "total_learned": 0,
                "consent_rate": 0.0,
                "quality_rate": 0.0,
                "daily_candidates": 0,
                "last_daily_reset": datetime.now().date().isoformat()
            },
            "settings": {
                "uncertainty_threshold": self.uncertainty_threshold,
                "diversity_threshold": self.diversity_threshold,
                "quality_threshold": self.quality_threshold,
                "consent_timeout_hours": self.consent_timeout_hours,
                "max_daily_candidates": self.max_daily_candidates,
                "min_consent_rate": self.min_consent_rate
            },
            "safeguards": {
                "enabled": True,
                "violations": [],
                "last_check": datetime.now().isoformat()
            }
        }
    
    def _migrate_database(self, db: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate database to current version"""
        try:
            version = db.get("version", 1)
            
            if version < 2:
                # Migrate to version 2
                db["version"] = 2
                db["consent_requests"] = []
                db["training_sessions"] = []
                
                # Migrate candidates to new format
                for candidate in db.get("candidates", []):
                    if "consent_status" not in candidate:
                        candidate["consent_status"] = ConsentStatus.PENDING.value
                    if "id" not in candidate:
                        candidate["id"] = str(uuid.uuid4())
                
                # Add safeguards section
                if "safeguards" not in db:
                    db["safeguards"] = {
                        "enabled": True,
                        "violations": [],
                        "last_check": datetime.now().isoformat()
                    }
                
                self.logger.info("Migrated database to version 2")
            
            return db
            
        except Exception as e:
            self.logger.error(f"Database migration failed: {e}")
            return db
    
    def evaluate_learning_candidate(self, image_data: bytes, features: Dict[str, float], 
                                  prediction_result: Dict[str, Any], 
                                  user_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of learning candidate with safeguards
        
        Args:
            image_data: Raw image bytes
            features: Extracted forensic features
            prediction_result: Detection results
            user_metadata: Additional metadata
            
        Returns:
            Dict containing evaluation results and next steps
        """
        try:
            with self.lock:
                # Check daily limits
                if not self._check_daily_limits():
                    return {
                        "selected": False,
                        "reason": "daily_limit_exceeded",
                        "daily_limit": self.max_daily_candidates
                    }
                
                # Generate unique identifiers
                image_hash = hashlib.sha256(image_data).hexdigest()
                candidate_id = str(uuid.uuid4())
                
                # Check for duplicates
                if self._is_duplicate(image_hash):
                    return {
                        "selected": False,
                        "reason": "duplicate_image",
                        "image_hash": image_hash
                    }
                
                # Comprehensive quality assessment
                quality_assessment = self.quality_assessor.comprehensive_assessment(
                    features, prediction_result, user_metadata
                )
                
                if quality_assessment["overall_score"] < self.quality_threshold:
                    return {
                        "selected": False,
                        "reason": "insufficient_quality",
                        "quality_score": quality_assessment["overall_score"],
                        "quality_details": quality_assessment,
                        "image_hash": image_hash
                    }
                
                # Calculate uncertainty and diversity scores
                uncertainty_score = self._calculate_uncertainty(prediction_result)
                diversity_score = self.diversity_analyzer.calculate_diversity(features, self.learning_db)
                
                # Selection decision
                selection_result = self.candidate_selector.evaluate_candidate(
                    uncertainty_score, diversity_score, quality_assessment["overall_score"]
                )
                
                if not selection_result["selected"]:
                    return {
                        "selected": False,
                        "reason": "selection_criteria_not_met",
                        "scores": {
                            "uncertainty": uncertainty_score,
                            "diversity": diversity_score,
                            "quality": quality_assessment["overall_score"]
                        },
                        "selection_details": selection_result,
                        "image_hash": image_hash
                    }
                
                # Create learning candidate
                candidate = LearningCandidate(
                    id=candidate_id,
                    image_hash=image_hash,
                    features=features,
                    prediction_confidence=prediction_result.get("confidence", 0.0),
                    uncertainty_score=uncertainty_score,
                    diversity_score=diversity_score,
                    quality_score=quality_assessment["overall_score"],
                    timestamp=datetime.now().isoformat(),
                    metadata=user_metadata or {}
                )
                
                # Add to database
                self._add_candidate(candidate)
                
                # Update statistics
                self._update_daily_statistics()
                
                self.logger.info(f"Selected learning candidate: {candidate_id}")
                
                return {
                    "selected": True,
                    "candidate_id": candidate_id,
                    "image_hash": image_hash,
                    "scores": {
                        "uncertainty": uncertainty_score,
                        "diversity": diversity_score,
                        "quality": quality_assessment["overall_score"]
                    },
                    "quality_assessment": quality_assessment,
                    "selection_details": selection_result,
                    "consent_required": True,
                    "consent_timeout_hours": self.consent_timeout_hours
                }
                
        except Exception as e:
            self.logger.error(f"Learning candidate evaluation failed: {e}")
            return {"selected": False, "error": str(e)}
    
    def process_consent_request(self, consent_request: ConsentRequest) -> Dict[str, Any]:
        """Process user consent with comprehensive validation"""
        try:
            with self.lock:
                # Validate consent request
                validation_result = self.consent_manager.validate_consent_request(consent_request)
                if not validation_result["valid"]:
                    return {
                        "success": False,
                        "error": "invalid_consent_request",
                        "details": validation_result
                    }
                
                # Find candidate
                candidate = self._find_candidate(consent_request.candidate_id)
                if not candidate:
                    return {
                        "success": False,
                        "error": "candidate_not_found",
                        "candidate_id": consent_request.candidate_id
                    }
                
                # Check if consent is still valid (not expired)
                candidate_time = datetime.fromisoformat(candidate["timestamp"])
                if datetime.now() - candidate_time > timedelta(hours=self.consent_timeout_hours):
                    # Mark as expired
                    candidate["consent_status"] = ConsentStatus.EXPIRED.value
                    self._save_learning_db()
                    
                    return {
                        "success": False,
                        "error": "consent_expired",
                        "candidate_id": consent_request.candidate_id,
                        "timeout_hours": self.consent_timeout_hours
                    }
                
                # Process consent
                if consent_request.consent_given:
                    candidate["consent_status"] = ConsentStatus.GRANTED.value
                    candidate["user_id"] = consent_request.user_id
                    candidate["consent_timestamp"] = datetime.now().isoformat()
                    
                    if consent_request.human_label:
                        candidate["human_label"] = consent_request.human_label
                    
                    if consent_request.additional_data:
                        candidate["metadata"].update(consent_request.additional_data)
                    
                    self.logger.info(f"Consent granted for candidate: {consent_request.candidate_id}")
                    
                else:
                    candidate["consent_status"] = ConsentStatus.DENIED.value
                    candidate["user_id"] = consent_request.user_id
                    candidate["consent_timestamp"] = datetime.now().isoformat()
                    
                    self.logger.info(f"Consent denied for candidate: {consent_request.candidate_id}")
                
                # Record consent request
                self.learning_db["consent_requests"].append({
                    "candidate_id": consent_request.candidate_id,
                    "user_id": consent_request.user_id,
                    "consent_given": consent_request.consent_given,
                    "timestamp": datetime.now().isoformat(),
                    "human_label": consent_request.human_label,
                    "additional_data": consent_request.additional_data
                })
                
                self._save_learning_db()
                
                return {
                    "success": True,
                    "candidate_id": consent_request.candidate_id,
                    "consent_status": candidate["consent_status"],
                    "consent_timestamp": candidate["consent_timestamp"]
                }
                
        except Exception as e:
            self.logger.error(f"Consent processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_training_batch(self, batch_size: int = 10, 
                          strategy: LearningStrategy = LearningStrategy.HYBRID_SAMPLING) -> Dict[str, Any]:
        """Get curated batch for retraining with comprehensive safeguards"""
        try:
            with self.lock:
                # Check safeguards
                safeguard_check = self.safeguard_monitor.check_training_safeguards(self.learning_db)
                if not safeguard_check["safe_to_train"]:
                    return {
                        "success": False,
                        "error": "safeguard_violation",
                        "details": safeguard_check,
                        "batch": []
                    }
                
                # Get consented candidates
                consented_candidates = [
                    c for c in self.learning_db["candidates"] 
                    if (c.get("consent_status") == ConsentStatus.GRANTED.value and 
                        not c.get("used_for_training", False))
                ]
                
                if not consented_candidates:
                    return {
                        "success": True,
                        "batch": [],
                        "message": "no_consented_candidates_available"
                    }
                
                # Apply selection strategy
                selected_batch = self._apply_selection_strategy(
                    consented_candidates, batch_size, strategy
                )
                
                # Mark as used for training
                training_session_id = str(uuid.uuid4())
                for candidate in selected_batch:
                    candidate["used_for_training"] = True
                    candidate["training_timestamp"] = datetime.now().isoformat()
                    candidate["training_session_id"] = training_session_id
                
                # Record training session
                training_session = {
                    "session_id": training_session_id,
                    "timestamp": datetime.now().isoformat(),
                    "batch_size": len(selected_batch),
                    "strategy": strategy.value,
                    "candidate_ids": [c["id"] for c in selected_batch],
                    "safeguard_check": safeguard_check
                }
                
                self.learning_db["training_sessions"].append(training_session)
                self._save_learning_db()
                
                self.logger.info(f"Generated training batch: {len(selected_batch)} samples, strategy: {strategy.value}")
                
                return {
                    "success": True,
                    "batch": selected_batch,
                    "training_session_id": training_session_id,
                    "strategy": strategy.value,
                    "safeguard_check": safeguard_check
                }
                
        except Exception as e:
            self.logger.error(f"Training batch generation failed: {e}")
            return {"success": False, "error": str(e), "batch": []}
    
    def update_training_results(self, training_session_id: str, 
                               training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update learning statistics after training"""
        try:
            with self.lock:
                # Find training session
                training_session = None
                for session in self.learning_db["training_sessions"]:
                    if session["session_id"] == training_session_id:
                        training_session = session
                        break
                
                if not training_session:
                    return {
                        "success": False,
                        "error": "training_session_not_found",
                        "session_id": training_session_id
                    }
                
                # Update session with results
                training_session["results"] = training_results
                training_session["completed_timestamp"] = datetime.now().isoformat()
                
                # Update global statistics
                stats = self.learning_db["statistics"]
                stats["total_learned"] += training_results.get("samples_trained", 0)
                
                # Update quality metrics
                if "quality_metrics" in training_results:
                    stats.update(training_results["quality_metrics"])
                
                # Update consent rate
                total_candidates = len(self.learning_db["candidates"])
                consented_candidates = len([
                    c for c in self.learning_db["candidates"] 
                    if c.get("consent_status") == ConsentStatus.GRANTED.value
                ])
                
                if total_candidates > 0:
                    stats["consent_rate"] = consented_candidates / total_candidates
                
                # Update quality rate
                high_quality_candidates = len([
                    c for c in self.learning_db["candidates"] 
                    if c.get("quality_score", 0) >= self.quality_threshold
                ])
                
                if total_candidates > 0:
                    stats["quality_rate"] = high_quality_candidates / total_candidates
                
                self._save_learning_db()
                
                self.logger.info(f"Updated training results for session: {training_session_id}")
                
                return {
                    "success": True,
                    "session_id": training_session_id,
                    "updated_statistics": stats
                }
                
        except Exception as e:
            self.logger.error(f"Training results update failed: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_expired_candidates(self) -> Dict[str, Any]:
        """Clean up expired and old candidates"""
        try:
            with self.lock:
                original_count = len(self.learning_db["candidates"])
                current_time = datetime.now()
                
                # Remove expired candidates
                self.learning_db["candidates"] = [
                    c for c in self.learning_db["candidates"]
                    if not self._is_candidate_expired(c, current_time)
                ]
                
                # Remove old denied candidates (older than 7 days)
                cutoff_time = current_time - timedelta(days=7)
                self.learning_db["candidates"] = [
                    c for c in self.learning_db["candidates"]
                    if not (c.get("consent_status") == ConsentStatus.DENIED.value and 
                           datetime.fromisoformat(c["timestamp"]) < cutoff_time)
                ]
                
                # Keep only recent consent requests (last 30 days)
                consent_cutoff = current_time - timedelta(days=30)
                self.learning_db["consent_requests"] = [
                    r for r in self.learning_db["consent_requests"]
                    if datetime.fromisoformat(r["timestamp"]) > consent_cutoff
                ]
                
                removed_count = original_count - len(self.learning_db["candidates"])
                
                if removed_count > 0:
                    self._save_learning_db()
                    self.logger.info(f"Cleaned up {removed_count} expired/old candidates")
                
                return {
                    "success": True,
                    "removed_candidates": removed_count,
                    "remaining_candidates": len(self.learning_db["candidates"])
                }
                
        except Exception as e:
            self.logger.error(f"Candidate cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning system statistics"""
        try:
            with self.lock:
                stats = self.learning_db["statistics"].copy()
                candidates = self.learning_db["candidates"]
                
                # Current counts
                stats["current_candidates"] = len(candidates)
                stats["pending_consent"] = len([c for c in candidates if c.get("consent_status") == ConsentStatus.PENDING.value])
                stats["consented_candidates"] = len([c for c in candidates if c.get("consent_status") == ConsentStatus.GRANTED.value])
                stats["denied_candidates"] = len([c for c in candidates if c.get("consent_status") == ConsentStatus.DENIED.value])
                stats["expired_candidates"] = len([c for c in candidates if c.get("consent_status") == ConsentStatus.EXPIRED.value])
                stats["trained_candidates"] = len([c for c in candidates if c.get("used_for_training", False)])
                
                # Quality distribution
                quality_scores = [c.get("quality_score", 0) for c in candidates]
                if quality_scores:
                    stats["quality_distribution"] = {
                        "mean": np.mean(quality_scores),
                        "std": np.std(quality_scores),
                        "min": np.min(quality_scores),
                        "max": np.max(quality_scores)
                    }
                
                # Uncertainty distribution
                uncertainty_scores = [c.get("uncertainty_score", 0) for c in candidates]
                if uncertainty_scores:
                    stats["uncertainty_distribution"] = {
                        "mean": np.mean(uncertainty_scores),
                        "std": np.std(uncertainty_scores),
                        "min": np.min(uncertainty_scores),
                        "max": np.max(uncertainty_scores)
                    }
                
                # Diversity distribution
                diversity_scores = [c.get("diversity_score", 0) for c in candidates]
                if diversity_scores:
                    stats["diversity_distribution"] = {
                        "mean": np.mean(diversity_scores),
                        "std": np.std(diversity_scores),
                        "min": np.min(diversity_scores),
                        "max": np.max(diversity_scores)
                    }
                
                # Recent activity
                recent_time = datetime.now() - timedelta(days=7)
                recent_candidates = [
                    c for c in candidates 
                    if datetime.fromisoformat(c["timestamp"]) > recent_time
                ]
                stats["recent_activity"] = {
                    "candidates_last_7_days": len(recent_candidates),
                    "consent_requests_last_7_days": len([
                        r for r in self.learning_db["consent_requests"]
                        if datetime.fromisoformat(r["timestamp"]) > recent_time
                    ]),
                    "training_sessions_last_7_days": len([
                        s for s in self.learning_db["training_sessions"]
                        if datetime.fromisoformat(s["timestamp"]) > recent_time
                    ])
                }
                
                # Safeguard status
                stats["safeguards"] = self.learning_db["safeguards"]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Statistics retrieval failed: {e}")
            return {"error": str(e)}
    
    # Helper methods
    def _check_daily_limits(self) -> bool:
        """Check if daily candidate limits are exceeded"""
        try:
            stats = self.learning_db["statistics"]
            today = datetime.now().date().isoformat()
            
            # Reset daily counter if new day
            if stats.get("last_daily_reset") != today:
                stats["daily_candidates"] = 0
                stats["last_daily_reset"] = today
            
            return stats["daily_candidates"] < self.max_daily_candidates
            
        except Exception:
            return True
    
    def _update_daily_statistics(self):
        """Update daily statistics"""
        try:
            stats = self.learning_db["statistics"]
            stats["daily_candidates"] = stats.get("daily_candidates", 0) + 1
            stats["total_candidates"] = len(self.learning_db["candidates"])
        except Exception as e:
            self.logger.error(f"Daily statistics update failed: {e}")
    
    def _is_duplicate(self, image_hash: str) -> bool:
        """Check if image hash already exists"""
        return any(c["image_hash"] == image_hash for c in self.learning_db["candidates"])
    
    def _calculate_uncertainty(self, prediction_result: Dict[str, Any]) -> float:
        """Calculate uncertainty score from prediction"""
        try:
            confidence = prediction_result.get("confidence", 0.5)
            
            # High uncertainty when confidence is close to 0.5 (decision boundary)
            uncertainty = 1.0 - abs(confidence - 0.5) * 2
            
            # Factor in robustness if available
            if "robustness" in prediction_result:
                robustness = prediction_result["robustness"].get("overall_robustness", 1.0)
                uncertainty = uncertainty * (1.0 - robustness * 0.5)
            
            return float(uncertainty)
            
        except Exception:
            return 0.5
    
    def _add_candidate(self, candidate: LearningCandidate):
        """Add candidate to database with size management"""
        candidate_dict = asdict(candidate)
        candidate_dict["consent_status"] = candidate.consent_status.value
        
        self.learning_db["candidates"].append(candidate_dict)
        
        # Maintain max candidates limit
        if len(self.learning_db["candidates"]) > self.max_candidates:
            # Remove oldest candidates without consent first
            self.learning_db["candidates"] = sorted(
                self.learning_db["candidates"],
                key=lambda x: (
                    x.get("consent_status") == ConsentStatus.GRANTED.value,
                    x.get("quality_score", 0),
                    x["timestamp"]
                ),
                reverse=True
            )[:self.max_candidates]
        
        self._save_learning_db()
    
    def _find_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Find candidate by ID"""
        for candidate in self.learning_db["candidates"]:
            if candidate["id"] == candidate_id:
                return candidate
        return None
    
    def _is_candidate_expired(self, candidate: Dict[str, Any], current_time: datetime) -> bool:
        """Check if candidate is expired"""
        try:
            if candidate.get("consent_status") != ConsentStatus.PENDING.value:
                return False
            
            candidate_time = datetime.fromisoformat(candidate["timestamp"])
            return current_time - candidate_time > timedelta(hours=self.consent_timeout_hours)
            
        except Exception:
            return False
    
    def _apply_selection_strategy(self, candidates: List[Dict], batch_size: int, 
                                 strategy: LearningStrategy) -> List[Dict]:
        """Apply selection strategy to get training batch"""
        try:
            if len(candidates) <= batch_size:
                return candidates
            
            if strategy == LearningStrategy.UNCERTAINTY_SAMPLING:
                return sorted(candidates, key=lambda x: x["uncertainty_score"], reverse=True)[:batch_size]
            
            elif strategy == LearningStrategy.DIVERSITY_SAMPLING:
                return sorted(candidates, key=lambda x: x["diversity_score"], reverse=True)[:batch_size]
            
            elif strategy == LearningStrategy.QUALITY_BASED:
                return sorted(candidates, key=lambda x: x["quality_score"], reverse=True)[:batch_size]
            
            else:  # HYBRID_SAMPLING
                # Combine all scores with weights
                for candidate in candidates:
                    candidate["hybrid_score"] = (
                        0.4 * candidate["uncertainty_score"] +
                        0.3 * candidate["diversity_score"] +
                        0.3 * candidate["quality_score"]
                    )
                
                return sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)[:batch_size]
                
        except Exception as e:
            self.logger.error(f"Selection strategy application failed: {e}")
            return candidates[:batch_size]
    
    def _save_learning_db(self):
        """Save learning database"""
        try:
            self.learning_db["last_updated"] = datetime.now().isoformat()
            os.makedirs(os.path.dirname(self.learning_db_path), exist_ok=True)
            with open(self.learning_db_path, 'w') as f:
                json.dump(self.learning_db, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save learning database: {e}")


class CandidateSelector:
    """Advanced candidate selection with multiple criteria"""
    
    def __init__(self):
        self.selection_weights = {
            "uncertainty": 0.4,
            "diversity": 0.3,
            "quality": 0.3
        }
    
    def evaluate_candidate(self, uncertainty_score: float, diversity_score: float, 
                          quality_score: float) -> Dict[str, Any]:
        """Comprehensive candidate evaluation"""
        try:
            # Weighted selection score
            selection_score = (
                self.selection_weights["uncertainty"] * uncertainty_score +
                self.selection_weights["diversity"] * diversity_score +
                self.selection_weights["quality"] * quality_score
            )
            
            # Dynamic threshold based on score distribution
            base_threshold = 0.6
            
            # Additional criteria
            criteria_met = {
                "uncertainty_sufficient": uncertainty_score > 0.3,
                "diversity_sufficient": diversity_score > 0.4,
                "quality_sufficient": quality_score > 0.7,
                "overall_score_sufficient": selection_score > base_threshold
            }
            
            selected = all(criteria_met.values())
            
            return {
                "selected": selected,
                "selection_score": selection_score,
                "criteria_met": criteria_met,
                "weights_used": self.selection_weights,
                "threshold": base_threshold
            }
            
        except Exception as e:
            return {
                "selected": False,
                "error": str(e),
                "selection_score": 0.0
            }


class QualityAssessor:
    """Comprehensive quality assessment for learning candidates"""
    
    def __init__(self):
        self.quality_weights = {
            "feature_completeness": 0.25,
            "feature_validity": 0.25,
            "prediction_stability": 0.20,
            "face_detection_quality": 0.15,
            "metadata_richness": 0.10,
            "robustness_score": 0.05
        }
    
    def comprehensive_assessment(self, features: Dict[str, float], 
                               prediction_result: Dict[str, Any],
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive quality assessment"""
        try:
            quality_factors = {}
            
            # Feature completeness
            expected_features = 50  # Adjust based on your feature set
            completeness = min(1.0, len(features) / expected_features)
            quality_factors["feature_completeness"] = completeness
            
            # Feature validity (no NaN, reasonable ranges)
            valid_features = sum(1 for v in features.values() 
                               if not np.isnan(v) and np.isfinite(v) and -1000 <= v <= 1000)
            validity = valid_features / len(features) if features else 0
            quality_factors["feature_validity"] = validity
            
            # Prediction stability
            stability = prediction_result.get("robustness", {}).get("overall_robustness", 0.5)
            quality_factors["prediction_stability"] = stability
            
            # Face detection quality
            face_info = prediction_result.get("face", {})
            face_quality = face_info.get("confidence", 0.0) if face_info.get("found", False) else 0.0
            quality_factors["face_detection_quality"] = face_quality
            
            # Metadata richness
            metadata_score = 0.0
            if metadata:
                metadata_score = min(1.0, len(metadata) / 10)  # Normalize to 10 expected fields
            quality_factors["metadata_richness"] = metadata_score
            
            # Robustness score
            robustness_score = prediction_result.get("robustness", {}).get("overall_robustness", 0.5)
            quality_factors["robustness_score"] = robustness_score
            
            # Compute weighted overall score
            overall_score = sum(
                quality_factors[factor] * self.quality_weights[factor]
                for factor in self.quality_weights.keys()
                if factor in quality_factors
            )
            
            # Quality flags
            quality_flags = self._generate_quality_flags(quality_factors)
            
            return {
                "overall_score": overall_score,
                "quality_factors": quality_factors,
                "quality_flags": quality_flags,
                "weights_used": self.quality_weights,
                "assessment_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "overall_score": 0.0,
                "error": str(e),
                "quality_factors": {},
                "quality_flags": ["assessment_failed"]
            }
    
    def _generate_quality_flags(self, quality_factors: Dict[str, float]) -> List[str]:
        """Generate quality warning flags"""
        flags = []
        
        if quality_factors.get("feature_completeness", 1.0) < 0.7:
            flags.append("incomplete_features")
        
        if quality_factors.get("feature_validity", 1.0) < 0.9:
            flags.append("invalid_features")
        
        if quality_factors.get("prediction_stability", 1.0) < 0.6:
            flags.append("unstable_prediction")
        
        if quality_factors.get("face_detection_quality", 1.0) < 0.8:
            flags.append("poor_face_detection")
        
        if quality_factors.get("metadata_richness", 1.0) < 0.3:
            flags.append("sparse_metadata")
        
        return flags


class ConsentManager:
    """Manage user consent with comprehensive validation"""
    
    def __init__(self):
        self.required_fields = ["candidate_id", "user_id", "consent_given"]
    
    def validate_consent_request(self, consent_request: ConsentRequest) -> Dict[str, Any]:
        """Validate consent request"""
        try:
            validation_errors = []
            
            # Check required fields
            if not consent_request.candidate_id:
                validation_errors.append("missing_candidate_id")
            
            if not consent_request.user_id:
                validation_errors.append("missing_user_id")
            
            if consent_request.consent_given is None:
                validation_errors.append("missing_consent_decision")
            
            # Validate timestamp if provided
            if consent_request.timestamp:
                try:
                    request_time = datetime.fromisoformat(consent_request.timestamp)
                    if datetime.now() - request_time > timedelta(hours=1):
                        validation_errors.append("consent_request_too_old")
                except ValueError:
                    validation_errors.append("invalid_timestamp_format")
            
            # Validate human label if provided
            if consent_request.human_label:
                valid_labels = ["real", "fake", "uncertain", "poor_quality"]
                if consent_request.human_label not in valid_labels:
                    validation_errors.append("invalid_human_label")
            
            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"validation_exception: {str(e)}"],
                "validation_timestamp": datetime.now().isoformat()
            }


class DiversityAnalyzer:
    """Analyze diversity of candidates for optimal learning"""
    
    def __init__(self):
        self.feature_weights = {}  # Will be computed dynamically
    
    def calculate_diversity(self, features: Dict[str, float], 
                          learning_db: Dict[str, Any]) -> float:
        """Calculate diversity score compared to existing samples"""
        try:
            existing_candidates = learning_db.get("candidates", [])
            
            if not existing_candidates:
                return 1.0  # First sample is maximally diverse
            
            # Extract features from existing candidates
            existing_features = [c.get("features", {}) for c in existing_candidates]
            
            # Calculate minimum distance to existing samples
            min_distance = float('inf')
            
            for existing in existing_features:
                distance = self._calculate_feature_distance(features, existing)
                min_distance = min(min_distance, distance)
            
            # Normalize distance to [0, 1] range
            diversity_score = min(1.0, min_distance / 10.0)  # Adjust scaling as needed
            
            return diversity_score
            
        except Exception as e:
            logger.error(f"Diversity calculation failed: {e}")
            return 0.5
    
    def _calculate_feature_distance(self, features1: Dict[str, float], 
                                   features2: Dict[str, float]) -> float:
        """Calculate distance between feature vectors"""
        try:
            common_keys = set(features1.keys()) & set(features2.keys())
            if not common_keys:
                return 1.0
            
            # Euclidean distance with normalization
            distance = 0.0
            for key in common_keys:
                # Normalize by feature range (simple approach)
                val1 = features1[key]
                val2 = features2[key]
                
                # Simple normalization
                normalized_diff = (val1 - val2) / (abs(val1) + abs(val2) + 1e-8)
                distance += normalized_diff ** 2
            
            return np.sqrt(distance / len(common_keys))
            
        except Exception:
            return 1.0


class SafeguardMonitor:
    """Monitor and enforce learning safeguards"""
    
    def __init__(self):
        self.safeguard_checks = [
            self._check_consent_rate,
            self._check_training_frequency,
            self._check_data_quality,
            self._check_user_diversity,
            self._check_system_resources
        ]
    
    def check_training_safeguards(self, learning_db: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive safeguard checking"""
        try:
            safeguard_results = {
                "safe_to_train": True,
                "checks_passed": [],
                "checks_failed": [],
                "warnings": [],
                "check_timestamp": datetime.now().isoformat()
            }
            
            # Run all safeguard checks
            for check_func in self.safeguard_checks:
                try:
                    check_result = check_func(learning_db)
                    
                    if check_result["passed"]:
                        safeguard_results["checks_passed"].append(check_result)
                    else:
                        safeguard_results["checks_failed"].append(check_result)
                        safeguard_results["safe_to_train"] = False
                    
                    if check_result.get("warnings"):
                        safeguard_results["warnings"].extend(check_result["warnings"])
                        
                except Exception as e:
                    safeguard_results["checks_failed"].append({
                        "check": check_func.__name__,
                        "passed": False,
                        "error": str(e)
                    })
                    safeguard_results["safe_to_train"] = False
            
            # Record safeguard check
            learning_db["safeguards"]["last_check"] = datetime.now().isoformat()
            if not safeguard_results["safe_to_train"]:
                learning_db["safeguards"]["violations"].append({
                    "timestamp": datetime.now().isoformat(),
                    "failed_checks": safeguard_results["checks_failed"]
                })
            
            return safeguard_results
            
        except Exception as e:
            return {
                "safe_to_train": False,
                "error": str(e),
                "check_timestamp": datetime.now().isoformat()
            }
    
    def _check_consent_rate(self, learning_db: Dict[str, Any]) -> Dict[str, Any]:
        """Check if consent rate is acceptable"""
        try:
            candidates = learning_db.get("candidates", [])
            if not candidates:
                return {"check": "consent_rate", "passed": True, "message": "no_candidates"}
            
            consented = len([c for c in candidates if c.get("consent_status") == ConsentStatus.GRANTED.value])
            consent_rate = consented / len(candidates)
            
            min_rate = learning_db.get("settings", {}).get("min_consent_rate", 0.3)
            
            return {
                "check": "consent_rate",
                "passed": consent_rate >= min_rate,
                "consent_rate": consent_rate,
                "min_required": min_rate,
                "consented_count": consented,
                "total_candidates": len(candidates)
            }
            
        except Exception as e:
            return {"check": "consent_rate", "passed": False, "error": str(e)}
    
    def _check_training_frequency(self, learning_db: Dict[str, Any]) -> Dict[str, Any]:
        """Check training frequency limits"""
        try:
            training_sessions = learning_db.get("training_sessions", [])
            if not training_sessions:
                return {"check": "training_frequency", "passed": True, "message": "no_training_sessions"}
            
            # Check last training session
            recent_sessions = [
                s for s in training_sessions
                if datetime.fromisoformat(s["timestamp"]) > datetime.now() - timedelta(hours=6)
            ]
            
            max_frequency_hours = learning_db.get("settings", {}).get("max_training_frequency_hours", 6)
            
            return {
                "check": "training_frequency",
                "passed": len(recent_sessions) == 0,
                "recent_sessions": len(recent_sessions),
                "max_frequency_hours": max_frequency_hours,
                "warnings": ["frequent_training"] if len(recent_sessions) > 0 else []
            }
            
        except Exception as e:
            return {"check": "training_frequency", "passed": False, "error": str(e)}
    
    def _check_data_quality(self, learning_db: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall data quality"""
        try:
            candidates = learning_db.get("candidates", [])
            if not candidates:
                return {"check": "data_quality", "passed": True, "message": "no_candidates"}
            
            # Check quality scores
            quality_scores = [c.get("quality_score", 0) for c in candidates]
            mean_quality = np.mean(quality_scores)
            
            min_quality = 0.6
            
            return {
                "check": "data_quality",
                "passed": mean_quality >= min_quality,
                "mean_quality": mean_quality,
                "min_required": min_quality,
                "quality_distribution": {
                    "mean": mean_quality,
                    "std": np.std(quality_scores),
                    "min": np.min(quality_scores),
                    "max": np.max(quality_scores)
                }
            }
            
        except Exception as e:
            return {"check": "data_quality", "passed": False, "error": str(e)}
    
    def _check_user_diversity(self, learning_db: Dict[str, Any]) -> Dict[str, Any]:
        """Check user diversity in consented samples"""
        try:
            consented_candidates = [
                c for c in learning_db.get("candidates", [])
                if c.get("consent_status") == ConsentStatus.GRANTED.value and c.get("user_id")
            ]
            
            if not consented_candidates:
                return {"check": "user_diversity", "passed": True, "message": "no_consented_candidates"}
            
            unique_users = len(set(c["user_id"] for c in consented_candidates))
            total_consented = len(consented_candidates)
            
            # Require at least 3 different users or 50% diversity
            min_users = max(3, total_consented * 0.5)
            
            return {
                "check": "user_diversity",
                "passed": unique_users >= min_users,
                "unique_users": unique_users,
                "total_consented": total_consented,
                "diversity_ratio": unique_users / total_consented,
                "min_required_users": min_users
            }
            
        except Exception as e:
            return {"check": "user_diversity", "passed": False, "error": str(e)}
    
    def _check_system_resources(self, learning_db: Dict[str, Any]) -> Dict[str, Any]:
        """Check system resource availability"""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            memory_ok = memory.percent < 85
            disk_ok = disk.percent < 90
            
            warnings = []
            if memory.percent > 75:
                warnings.append("high_memory_usage")
            if disk.percent > 80:
                warnings.append("high_disk_usage")
            
            return {
                "check": "system_resources",
                "passed": memory_ok and disk_ok,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": disk.percent,
                "warnings": warnings
            }
            
        except Exception as e:
            return {"check": "system_resources", "passed": True, "error": str(e), "message": "psutil_unavailable"}