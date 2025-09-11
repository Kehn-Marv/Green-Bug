import os
import uuid
import asyncio
import json
from typing import Optional, List, Any, Callable, Dict
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import cv2

from src.config import OUTPUT_DIR, FINGERPRINTS_PATH
from src.utils.img import load_image_from_bytes, ensure_min_size
from src.utils.validation import validate_file_upload, validate_image_dimensions, sanitize_filename
from src.utils.logging import setup_logger, log_analysis_request, log_analysis_result, log_error
from src.ingest.filtering import accept_image
from src.trace.attribution import AttributionIndex
from src.models.deep_model_detector import DeepModelDetector
from src.models.face_detector import face_detector
from src.api.rate_limiter import rate_limiter
from src.api.job_store import get_store
from src.core.ingest_filter import IngestFilter
from src.core.forensic_feature_extractor import ForensicFeatureExtractor
from src.core.attribution_engine import AttributionEngine
from src.core.robustness_tester import RobustnessTester
from src.core.self_learning_system import SelfLearningSystem
from src.core.report_generator import ForensicReportGenerator

logger = setup_logger(__name__)
router = APIRouter()

# Simple in-memory job store for background analysis jobs. Keys are job ids.
# Structure: { job_id: {status: 'queued'|'running'|'completed'|'failed', result: dict|None, error: str|None, created_at: str} }
_store = None

def get_job_store():
    global _store
    if _store is not None:
        return _store
    return get_store()

def set_job_store(store):
    """Set the module-level job store (useful for tests)."""
    global _store
    _store = store


async def _run_background_job(job_id: str, data: bytes, filename: str, target_layer: Optional[str], strip_exif: bool, enable_learning: bool, generate_report: bool):
    store = get_job_store()
    await store.publish_event(job_id, {'status': 'running', 'progress': 1, 'message': 'Started'})
    try:
        # Stage: ingest
        await store.publish_event(job_id, {'status': 'running', 'progress': 10, 'message': 'Ingesting and validating'})

        # Extract features
        await store.publish_event(job_id, {'status': 'running', 'progress': 30, 'message': 'Extracting forensic features'})
        result = await process_image_data(
            data,
            filename,
            target_layer,
            strip_exif,
            enable_learning,
            generate_report,
            progress_callback=lambda p, m: asyncio.create_task(store.publish_event(job_id, {'status': 'running', 'progress': p, 'message': m}))
        )

        # After features
        await store.publish_event(job_id, {'status': 'running', 'progress': 70, 'message': 'Running CNN/grad-cam and robustness tests'})

        # Finalizing
        await store.publish_event(job_id, {'status': 'running', 'progress': 95, 'message': 'Finalizing and saving outputs'})
        await store.set_result(job_id, result)

    except Exception as e:
        await store.set_failed(job_id, str(e))


def _sanitize_for_json(value):
    """Recursively convert numpy types and other non-serializable values to
    native Python types so FastAPI's jsonable_encoder can handle them.
    """
    # Import locally to avoid hard dependency at module import time
    import numpy as _np

    # Numpy scalar types
    if isinstance(value, (_np.generic,)):
        try:
            return value.item()
        except Exception:
            return value.tolist() if hasattr(value, 'tolist') else value

    # Numpy arrays -> lists
    if isinstance(value, _np.ndarray):
        try:
            return value.tolist()
        except Exception:
            # Fall back to converting elements
            return [_sanitize_for_json(v) for v in value]

    # dict -> sanitize contents
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}

    # list/tuple/set -> sanitize elements
    if isinstance(value, (list, tuple, set)):
        return [ _sanitize_for_json(v) for v in value ]

    # Objects with __dict__ -> dict
    if hasattr(value, '__dict__'):
        return _sanitize_for_json(vars(value))

    # Fallback: primitive types (int, float, bool, str, None) or leave as-is
    return value


async def maybe_call(cb: Optional[Callable[[int, str], Any]], pct: int, message: str):
    """Call progress callback which may be sync or async."""
    if cb is None:
        return
    try:
        res = cb(pct, message)
        if asyncio.iscoroutine(res):
            await res
    except Exception:
        # Swallow callback exceptions - progress should never crash processing
        return

# Lazy initializers for heavy components to avoid import-time initialization
_ingest_filter = None
_feature_extractor = None
_attribution_engine = None
_robustness_tester = None
_self_learning_system = None
_report_generator = None

def get_ingest_filter() -> IngestFilter:
    global _ingest_filter
    if _ingest_filter is None:
        _ingest_filter = IngestFilter()
    return _ingest_filter

def get_feature_extractor() -> ForensicFeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = ForensicFeatureExtractor()
    return _feature_extractor

def get_attribution_engine() -> AttributionEngine:
    global _attribution_engine
    if _attribution_engine is None:
        _attribution_engine = AttributionEngine(FINGERPRINTS_PATH, FINGERPRINTS_PATH.replace('.json', '_embeddings.pkl'))
    return _attribution_engine

def get_robustness_tester() -> RobustnessTester:
    global _robustness_tester
    if _robustness_tester is None:
        _robustness_tester = RobustnessTester()
    return _robustness_tester

def get_self_learning_system() -> SelfLearningSystem:
    global _self_learning_system
    if _self_learning_system is None:
        _self_learning_system = SelfLearningSystem(FINGERPRINTS_PATH.replace('.json', '_learning.json'))
    return _self_learning_system

def get_report_generator() -> ForensicReportGenerator:
    global _report_generator
    if _report_generator is None:
        _report_generator = ForensicReportGenerator(OUTPUT_DIR)
    return _report_generator

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

        # Run processing synchronously (existing behavior)
        sanitized_result = await process_image_data(data, safe_filename, target_layer, strip_exif, enable_learning, generate_report)
        log_analysis_result(logger, sanitized_result)
        return sanitized_result

    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, "image analysis")
        raise HTTPException(status_code=500, detail="Internal analysis error")


@router.post('/analyze/submit')
async def submit_analysis(
    file: UploadFile = File(...),
    target_layer: Optional[str] = "layer4.1.conv2",
    strip_exif: bool = True,
    enable_learning: bool = True,
    generate_report: bool = False,
    _: bool = Depends(check_rate_limit)
):
    """Submit an analysis job to be processed in background; returns a job_id to poll."""
    data = await file.read()
    safe_filename = sanitize_filename(file.filename or "unknown.jpg")
    job_id = uuid.uuid4().hex[:12]
    store = get_job_store()
    await store.create_job(job_id, {
        'result': None,
        'error': None,
        'progress': 0,
        'created_at': datetime.now().isoformat()
    })

    # Rough estimate for processing time: base 30s + 0.001 * size_bytes
    estimated_seconds = 30 + int(len(data) * 0.001)

    # Kick off background task
    asyncio.create_task(_run_background_job(job_id, data, safe_filename, target_layer, strip_exif, enable_learning, generate_report))

    return {'job_id': job_id, 'estimated_seconds': estimated_seconds}


@router.get('/analyze/events/{job_id}')
async def analyze_events(request: Request, job_id: str):
    """Server-Sent Events endpoint streaming progress updates for a job_id using the job store."""
    job = await get_job_store().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='job_id not found')

    async def event_generator():
        async for ev in _store.subscribe(job_id):
            if await request.is_disconnected():
                break
            try:
                yield f"data: {json.dumps(ev)}\n\n"
            except Exception:
                # yield minimal event
                yield f"data: {json.dumps({'status': ev.get('status', 'running')})}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@router.get('/analyze/result/{job_id}')
async def get_analysis_result(job_id: str):
    """Get status or result for a submitted job."""
    job = await get_job_store().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job ID not found')

    return {
        'job_id': job_id,
        'status': job.get('status'),
        'result': job.get('result'),
        'error': job.get('error'),
        'progress': job.get('progress')
    }


async def process_image_data(data: bytes, safe_filename: str, target_layer: Optional[str], strip_exif: bool, enable_learning: bool, generate_report: bool, progress_callback: Optional[Callable[[int, str], Any]] = None) -> Dict[str, Any]:
    """Extracted core processing so it can be used synchronously or in background jobs."""
    # 1. INGEST & FILTER - Process upload with quality assessment
    await maybe_call(progress_callback, 5, 'Starting ingest and validation')
    ingest_result: Dict[str, Any] = get_ingest_filter().process_upload(data, safe_filename, strip_exif=strip_exif)
    processed_image = ingest_result['processed_image']
    quality_assessment = ingest_result['quality_assessment']

    validate_image_dimensions(processed_image)
    processed_image = ensure_min_size(processed_image, 256)

    # Face detection with enhanced processing
    face_im, face_found, face_conf = face_detector.detect_largest_face(processed_image)
    await maybe_call(progress_callback, 20, 'Face detection completed')

    # Quality assessment
    accept, qflags = accept_image(face_found, face_conf, face_im.width, face_im.height)

    # 2. FORENSIC FEATURE EXTRACTION - Comprehensive feature extraction
    await maybe_call(progress_callback, 30, 'Extracting forensic features')
    feature_extraction_result = get_feature_extractor().extract_comprehensive_features(face_im)
    comprehensive_features = feature_extraction_result['structured_features']
    flattened_features = feature_extraction_result['flattened_features']

    # Use the deep-learning detector as the primary deepfake model
    await maybe_call(progress_callback, 60, 'Running deep model prediction')
    deep_detector = DeepModelDetector()
    try:
        deep_pred_any = deep_detector.predict(face_im)
    except Exception:
        deep_pred_any = {"available": False}

    # Ensure we have a dict for static analysis
    deep_pred: Dict[str, Any] = deep_pred_any if isinstance(deep_pred_any, dict) else {"available": False}
    # Ensure deep_pred is a mapping to satisfy static checks
    if not isinstance(deep_pred, dict):
        deep_pred = {"available": False}

    # Map deep model output to compatibility fields
    heur_result: Dict[str, Any] = {"score": 0.5, "features": {}}
    deep_score: Optional[float] = None
    torch_pred: Dict[str, Any] = {"available": False}

    if deep_pred.get('available'):
        torch_pred = deep_pred
        probs = deep_pred.get('probs')
        # HuggingFace pipeline returns list of {'label','score'}; local torch returns list of floats
        if isinstance(probs, list) and len(probs) > 0:
            first = probs[0]
            if isinstance(first, dict) and 'score' in first:
                deep_score = float(first['score'])
            else:
                try:
                    # numeric probs
                    deep_score = float(probs[0])
                except Exception:
                    deep_score = None

        heur_result = {"score": deep_score if deep_score is not None else 0.5, "features": {"model_probs": probs}}

    # 3. ATTRIBUTION ENGINE - Ensemble attribution analysis
    raw_attribution = get_attribution_engine().analyze_attribution(flattened_features)
    attribution_result: Dict[str, Any]
    if isinstance(raw_attribution, dict):
        attribution_result = raw_attribution
    else:
        attribution_result = {}

    # Legacy attribution for compatibility
    try:
        # Legacy attribution: previously used feature-based matches. Attempt to match using model_probs if present.
        attribution_idx = AttributionIndex(FINGERPRINTS_PATH)
        raw_feats = heur_result.get("features") or {}
        # Build a typed dict of numeric features for attribution matching
        match_input: Dict[str, float] = {}
        if isinstance(raw_feats, dict):
            for k, v in raw_feats.items():
                try:
                    match_input[k] = float(v)
                except Exception:
                    # skip non-numeric feature entries
                    continue

        topk_matches = attribution_idx.match(match_input, topk=3)
    except Exception as e:
        logger.error(f"Legacy attribution analysis failed: {e}")
        topk_matches = []

    # 4. ROBUSTNESS TESTING - Test against various manipulations
    # Generate heatmap (Grad-CAM) if available from torch-based detector
    heat = None
    try:
        if isinstance(deep_detector, DeepModelDetector) and hasattr(deep_detector, 'gradcam'):
            heat = deep_detector.gradcam(face_im)
    except Exception:
        heat = None

    if heat is None:
        heat = _create_fallback_heatmap(face_im)

    # Robustness testing
    await maybe_call(progress_callback, 75, 'Starting robustness tests')
    def detector_func(img):
        """Wrapper function for robustness testing using deep model"""
        try:
            p = deep_detector.predict(img)
            if p.get('available'):
                probs = p.get('probs')
                if isinstance(probs, list) and len(probs) > 0:
                    first = probs[0]
                    if isinstance(first, dict) and 'score' in first:
                        return float(first['score'])
                    else:
                        try:
                            return float(probs[0])
                        except Exception:
                            return 0.5
            return 0.5
        except Exception:
            return 0.5

    # Ensure we pass numeric float for baseline score
    baseline_score = float(heur_result.get("score", 0.5) or 0.5)
    robustness_result_raw = get_robustness_tester().comprehensive_robustness_test(
        face_im, detector_func, baseline_score
    )
    # Normalize robustness_result to a dict for downstream code
    robustness_result: Dict[str, Any] = robustness_result_raw if isinstance(robustness_result_raw, dict) else {}
    await maybe_call(progress_callback, 85, 'Robustness tests complete')

    # 5. SELF-LEARNING - Evaluate for learning candidacy
    learning_evaluation: Optional[Dict[str, Any]] = None
    if enable_learning:
        raw_learning = get_self_learning_system().evaluate_learning_candidate(
            data, flattened_features, {"confidence": heur_result["score"], "robustness": robustness_result}
        )
        learning_evaluation = raw_learning if isinstance(raw_learning, dict) else None

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
        raise

    await maybe_call(progress_callback, 92, 'Saved visualization files')

    # Extract metadata
    try:
        exif = ingest_result.get('exif_data', {})
    # frequency metadata previously extracted from feature-based analysis; preserve if available
        freq_meta = {
            "fft_high_ratio": heur_result.get("features", {}).get("fft_high_ratio"),
            "lap_var": heur_result.get("features", {}).get("lap_var"),
            "jpeg_score": heur_result.get("features", {}).get("jpeg_score")
        }
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        exif = {}
        freq_meta = {}

    # Build comprehensive response
    # Ensure deep_model_score is always populated for a canonical field
    canonical_deep_score: float = float(deep_score if deep_score is not None else heur_result.get("score", 0.5) or 0.5)

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
            # canonical field only
            "deep_model_score": canonical_deep_score
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
            report_files = get_report_generator().generate_comprehensive_report(
                result,
                original_image_path=safe_filename,
                case_info={"filename": safe_filename, "analysis_id": uid}
            )
            result["forensic_reports"] = report_files
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            result["forensic_reports"] = {"error": str(e)}

    await maybe_call(progress_callback, 98, 'Report generation complete')

    # Ensure response is JSON-serializable (convert numpy types, objects, etc.)
    sanitized_result = _sanitize_for_json(result)
    return sanitized_result

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

        result = get_self_learning_system().process_consent_request(consent_request)
        return result

    except Exception as e:
        log_error(logger, e, "consent processing")
        raise HTTPException(status_code=500, detail="Consent processing failed")

@router.get("/analyze/learning/stats")
async def get_learning_stats():
    """Get self-learning system statistics"""
    try:
        stats = get_self_learning_system().get_comprehensive_statistics()
        return stats
    except Exception as e:
        log_error(logger, e, "learning stats retrieval")
        raise HTTPException(status_code=500, detail="Failed to retrieve learning statistics")

@router.post("/analyze/learning/cleanup")
async def cleanup_learning_data():
    """Clean up expired learning candidates"""
    try:
        result = get_self_learning_system().cleanup_expired_candidates()
        return result
    except Exception as e:
        log_error(logger, e, "learning data cleanup")
        raise HTTPException(status_code=500, detail="Learning data cleanup failed")


# Setter helpers for tests to inject lightweight test doubles (dependency injection)
def set_ingest_filter(obj: IngestFilter):
    global _ingest_filter
    _ingest_filter = obj

def set_feature_extractor(obj: ForensicFeatureExtractor):
    global _feature_extractor
    _feature_extractor = obj

def set_attribution_engine(obj: AttributionEngine):
    global _attribution_engine
    _attribution_engine = obj

def set_robustness_tester(obj):
    global _robustness_tester

def set_self_learning_system(obj: SelfLearningSystem):
    global _self_learning_system
    _self_learning_system = obj

def set_report_generator(obj: ForensicReportGenerator):
    global _report_generator
    _report_generator = obj

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