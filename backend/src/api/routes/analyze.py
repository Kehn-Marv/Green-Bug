import os
import uuid
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from PIL import Image
import numpy as np
import cv2

from src.config import OUTPUT_DIR, FINGERPRINTS_PATH, WEIGHTS_PATH
from src.utils.img import load_image_from_bytes, ensure_min_size
from src.utils.exif_utils import extract_exif_summary
from src.utils.validation import validate_file_upload, validate_image_dimensions, sanitize_filename
from src.utils.logging import setup_logger, log_analysis_request, log_analysis_result, log_error
from src.ingest.filtering import accept_image
from src.trace.attribution import AttributionIndex
from src.models.detector import HeuristicDetector, TorchDetector
from src.models.face_detector import face_detector
from src.api.rate_limiter import rate_limiter
from src.core.ingest_filter import IngestFilter
from src.core.forensic_feature_extractor import ForensicFeatureExtractor
from src.core.attribution_engine import AttributionEngine
from src.core.robustness_tester import RobustnessTester
from src.core.self_learning_system import SelfLearningSystem
from src.core.report_generator import ForensicReportGenerator

logger = setup_logger(__name__)
router = APIRouter()

# Initialize core components
ingest_filter = IngestFilter()
feature_extractor = ForensicFeatureExtractor()
attribution_engine = AttributionEngine(FINGERPRINTS_PATH, FINGERPRINTS_PATH.replace('.json', '_embeddings.pkl'))
robustness_tester = RobustnessTester()
self_learning_system = SelfLearningSystem(FINGERPRINTS_PATH.replace('.json', '_learning.json'))
report_generator = ForensicReportGenerator(OUTPUT_DIR)

def check_rate_limit(request: Request):
    """Dependency for rate limiting"""
    return rate_limiter.check_rate_limit(request)

def _create_fallback_heatmap(face_im: Image.Image) -> np.ndarray:
    """Create forensic saliency map when Grad-CAM is not available"""
    try:
        arr = np.array(face_im.convert("L"))
        
        # Combine multiple forensic indicators
        edges = cv2.Canny(arr, 80, 200).astype("float32")
        
        # Add Laplacian for texture analysis
        laplacian = cv2.Laplacian(arr, cv2.CV_64F)
        laplacian_norm = np.abs(laplacian).astype("float32")
        laplacian_norm = (laplacian_norm / (laplacian_norm.max() + 1e-8)) * 255
        
        # Combine edge and texture information
        combined = 0.7 * edges + 0.3 * laplacian_norm
        combined = np.clip(combined, 0, 255)
        
        return combined / 255.0
        
    except Exception as e:
        logger.error(f"Fallback heatmap creation failed: {e}")
        # Return uniform heatmap as last resort
        return np.ones((face_im.height, face_im.width), dtype=np.float32) * 0.5

def _overlay_heatmap(base_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Create overlay of heatmap on original image"""
    try:
        # Normalize heatmap
        heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_uint8 = (255 * heat_norm).astype("uint8")
        
        # Apply colormap
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        
        # Resize to match base image
        if heat_color.shape[:2] != base_rgb.shape[:2]:
            heat_color = cv2.resize(heat_color, (base_rgb.shape[1], base_rgb.shape[0]))
        
        # Blend
        overlay = (alpha * heat_color + (1 - alpha) * base_rgb).astype("uint8")
        return overlay
        
    except Exception as e:
        logger.error(f"Heatmap overlay creation failed: {e}")
        return base_rgb  # Return original if overlay fails

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_layer: Optional[str] = "layer4.1.conv2",
    strip_exif: bool = True,
    enable_learning: bool = True,
    generate_report: bool = False,
    _: bool = Depends(check_rate_limit)
):
    """
    Comprehensive deepfake analysis with full pipeline:
    - Ingest & filtering with quality assessment
    - Forensic feature extraction (residual/noise, spectral, color, texture, CNN)
    - Attribution engine with ensemble methods
    - Robustness testing against various manipulations
    - Self-learning candidate evaluation
    - Optional forensic report generation
    """
    
    # Validate file upload
    file_size = 0
    data = await file.read()
    file_size = len(data)
    
    safe_filename = sanitize_filename(file.filename or "unknown.jpg")
    
    try:
        validate_file_upload(safe_filename, file_size)
        log_analysis_request(logger, safe_filename, file_size)
        
        # 1. INGEST & FILTER - Process upload with quality assessment
        ingest_result = ingest_filter.process_upload(data, safe_filename, strip_exif=strip_exif)
        processed_image = ingest_result['processed_image']
        quality_assessment = ingest_result['quality_assessment']
        
        validate_image_dimensions(processed_image)
        processed_image = ensure_min_size(processed_image, 256)
        
        # Face detection with enhanced processing
        face_im, face_found, face_conf = face_detector.detect_largest_face(processed_image)
        
        # Quality assessment
        accept, qflags = accept_image(face_found, face_conf, face_im.width, face_im.height)
        
        # 2. FORENSIC FEATURE EXTRACTION - Comprehensive feature extraction
        feature_extraction_result = feature_extractor.extract_comprehensive_features(face_im)
        comprehensive_features = feature_extraction_result['structured_features']
        flattened_features = feature_extraction_result['flattened_features']
        
        # Legacy heuristic analysis for compatibility
        heur_detector = HeuristicDetector()
        heur_result = heur_detector.analyze(face_im)
        
        # Deep model analysis
        torch_detector = TorchDetector(WEIGHTS_PATH, device="cpu", target_layer=target_layer)
        torch_pred = torch_detector.predict(face_im)
        
        # Calculate deep model score
        deep_score = None
        if torch_pred.get("available"):
            probs = torch_pred["probs"]
            if len(probs) == 1:
                deep_score = float(probs[0])
            elif len(probs) >= 2:
                deep_score = float(1.0 - probs[1])  # Assuming [real, fake] order
        
        # 3. ATTRIBUTION ENGINE - Ensemble attribution analysis
        attribution_result = attribution_engine.analyze_attribution(flattened_features)
        
        # Legacy attribution for compatibility
        try:
            attribution_idx = AttributionIndex(FINGERPRINTS_PATH)
            topk_matches = attribution_idx.match(heur_result["features"], topk=3)
        except Exception as e:
            logger.error(f"Legacy attribution analysis failed: {e}")
            topk_matches = []
        
        # 4. ROBUSTNESS TESTING - Test against various manipulations
        # Generate heatmap
        heat = None
        if torch_pred.get("available"):
            heat = torch_detector.gradcam(face_im, class_idx=None)
        
        if heat is None:
            heat = _create_fallback_heatmap(face_im)
        
        # Robustness testing
        def detector_func(img):
            """Wrapper function for robustness testing"""
            try:
                heur_result_temp = heur_detector.analyze(img)
                return heur_result_temp["score"]
            except:
                return 0.5
        
        robustness_result = robustness_tester.comprehensive_robustness_test(
            face_im, detector_func, heur_result["score"]
        )
        
        # 5. SELF-LEARNING - Evaluate for learning candidacy
        learning_evaluation = None
        if enable_learning:
            learning_evaluation = self_learning_system.evaluate_learning_candidate(
                data, flattened_features, {"confidence": heur_result["score"], "robustness": robustness_result}
            )
        
        # Save visualizations
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        uid = uuid.uuid4().hex[:12]
        
        base_rgb = np.array(face_im)
        
        # Handle different heatmap dimensions
        if heat.ndim == 2:
            overlay = _overlay_heatmap(base_rgb, heat)
        else:
            heat_2d = heat.squeeze() if heat.ndim > 2 else heat
            overlay = _overlay_heatmap(base_rgb, heat_2d)
        
        # Save files
        heat_path = os.path.join(OUTPUT_DIR, f"heat_{uid}.png")
        overlay_path = os.path.join(OUTPUT_DIR, f"overlay_{uid}.png")
        
        try:
            Image.fromarray(overlay).save(overlay_path)
            
            # Save heatmap as grayscale
            heat_norm = (heat - np.min(heat)) / (np.max(heat) - np.min(heat) + 1e-8)
            heat_uint8 = (255 * heat_norm).astype("uint8")
            if heat_uint8.ndim > 2:
                heat_uint8 = heat_uint8.squeeze()
            Image.fromarray(heat_uint8, mode='L').save(heat_path)
            
        except Exception as e:
            logger.error(f"Failed to save visualization files: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate visualizations")
        
        # Extract metadata
        try:
            exif = ingest_result.get('exif_data', {})
            freq_meta = {
                "fft_high_ratio": heur_result["features"]["fft_high_ratio"],
                "lap_var": heur_result["features"]["lap_var"],
                "jpeg_score": heur_result["features"]["jpeg_score"]
            }
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            exif = {}
            freq_meta = {}
        
        # Build comprehensive response
        # Build response
        result = {
            "id": uid,
            "received_filename": safe_filename,
            "face": {
                "found": face_found,
                "confidence": face_conf,
                "used_region": [face_im.width, face_im.height]
            },
            "quality": {
                "ingest_assessment": quality_assessment,
                "training_candidate": ingest_result.get('training_candidate', {}),
                "comprehensive_features_count": len(flattened_features),
                "feature_categories": list(comprehensive_features.keys()),
                # Legacy compatibility
                "accepted_for_learning": accept,
                "flags": qflags.__dict__
            },
            "scores": {
                "heuristic_deepfake_score": heur_result["score"],
                "deep_model_score": deep_score
            },
            # Enhanced features
            "features": heur_result["features"],
            "comprehensive_features": comprehensive_features,
            "flattened_features": flattened_features,
            # Enhanced attribution
            "attribution": attribution_result,
            "attribution_topk": [
                {"family": name, "similarity": float(sim)} 
                for name, sim in topk_matches
            ],
            # Robustness analysis
            "robustness": robustness_result,
            # Self-learning
            "learning_evaluation": learning_evaluation,
            "legacy_attribution_topk": [
                {"family": name, "similarity": float(sim)} 
                for name, sim in topk_matches
            ],
            "exif": exif,
            "frequency_meta": freq_meta,
            "files": {
                "heatmap_url": f"/files/{os.path.basename(heat_path)}",
                "overlay_url": f"/files/{os.path.basename(overlay_path)}"
            },
            "processing_metadata": {
                "file_hash": ingest_result.get('file_hash'),
                "original_size": ingest_result.get('original_size'),
                "exif_stripped": ingest_result.get('exif_stripped', False),
                "processing_timestamp": ingest_result.get('processing_timestamp')
            },
            "notes": [
                "Deep model unavailable" if not torch_pred.get("available") else "Deep model used",
                "Forensic fallback heatmap" if not torch_pred.get("available") else f"Grad-CAM layer={target_layer}",
                f"Comprehensive analysis with {len(flattened_features)} features",
                f"Attribution: {attribution_result.get('ensemble', {}).get('predicted_family', 'unknown')}",
                f"Robustness score: {robustness_result.get('overall_robustness', 0):.3f}",
                f"Learning candidate: {'Yes' if learning_evaluation and learning_evaluation.get('selected') else 'No'}"
            ]
        }
        
        # 6. REPORT GENERATION - Generate forensic report if requested
        if generate_report:
            try:
                report_files = report_generator.generate_comprehensive_report(
                    result, 
                    original_image_path=None,
                    case_info={"filename": safe_filename, "analysis_id": uid}
                )
                result["forensic_reports"] = report_files
            except Exception as e:
                logger.error(f"Report generation failed: {e}")
                result["forensic_reports"] = {"error": str(e)}
        
        log_analysis_result(logger, result)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, "image analysis")
        raise HTTPException(status_code=500, detail="Internal analysis error")

@router.post("/analyze/consent")
async def process_consent(
    candidate_id: str,
    user_id: str,
    consent_given: bool,
    human_label: Optional[str] = None
):
    """Process user consent for self-learning"""
    try:
        from src.core.self_learning_system import ConsentRequest
        
        consent_request = ConsentRequest(
            candidate_id=candidate_id,
            user_id=user_id,
            consent_given=consent_given,
            human_label=human_label,
            timestamp=datetime.now().isoformat()
        )
        
        result = self_learning_system.process_consent_request(consent_request)
        return result
        
    except Exception as e:
        log_error(logger, e, "consent processing")
        raise HTTPException(status_code=500, detail="Consent processing failed")

@router.get("/analyze/learning/stats")
async def get_learning_stats():
    """Get self-learning system statistics"""
    try:
        stats = self_learning_system.get_comprehensive_statistics()
        return stats
    except Exception as e:
        log_error(logger, e, "learning stats retrieval")
        raise HTTPException(status_code=500, detail="Failed to retrieve learning statistics")

@router.post("/analyze/learning/cleanup")
async def cleanup_learning_data():
    """Clean up expired learning candidates"""
    try:
        result = self_learning_system.cleanup_expired_candidates()
        return result
    except Exception as e:
        log_error(logger, e, "learning data cleanup")
        raise HTTPException(status_code=500, detail="Learning data cleanup failed")

@router.post("/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Analyze multiple images in batch.
    Limited to 5 images per request to prevent resource exhaustion.
    """
    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 images per batch request"
        )
    
    logger.info(f"Batch analysis request: {len(files)} files")
    
    try:
        # Load all images first
        images = []
        for file in files:
            data = await file.read()
            validate_file_upload(sanitize_filename(file.filename or ""), len(data))
            im = load_image_from_bytes(data)
            validate_image_dimensions(im)
            images.append(ensure_min_size(im, 256))
        
        # Process batch
        from src.api.batch_processor import batch_processor
        results = await batch_processor.process_batch(images)
        
        return {
            "batch_id": uuid.uuid4().hex[:12],
            "total_images": len(files),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, "batch analysis")
        raise HTTPException(status_code=500, detail="Batch analysis failed")

from datetime import datetime