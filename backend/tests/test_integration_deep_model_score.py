import asyncio
from PIL import Image
from io import BytesIO

from src.api.routes.analyze import process_image_data, set_ingest_filter, set_feature_extractor
from src.models.deep_model_detector import DeepModelDetector

# Create a stub DeepModelDetector for tests
class StubDeepModelDetector(DeepModelDetector):
    def __init__(self):
        # do not call super loader
        self._available = True

    def predict(self, im: Image.Image):
        return {"available": True, "probs": [{"label": "FAKE", "score": 0.87}]}

# Minimal ingest filter stub that returns processed_image and metadata
class StubIngestFilter:
    def process_upload(self, data: bytes, filename: str, strip_exif: bool = True):
        im = Image.open(BytesIO(data)).convert('RGB')
        return {
            'processed_image': im,
            'quality_assessment': {'score': 1.0},
            'training_candidate': {},
            'exif_data': {},
            'file_hash': 'stubhash',
            'original_size': len(data),
            'exif_stripped': strip_exif,
            'processing_timestamp': 'now'
        }

# Use a lightweight feature extractor stub
class StubFeatureExtractor:
    def extract_comprehensive_features(self, im: Image.Image):
        return {'structured_features': {}, 'flattened_features': []}


def _make_test_image_bytes():
    im = Image.new('RGB', (256, 256), color='blue')
    buf = BytesIO()
    im.save(buf, format='JPEG')
    return buf.getvalue()

def test_deep_model_score_present():
    # Inject stubs
    set_ingest_filter(StubIngestFilter())
    set_feature_extractor(StubFeatureExtractor())

    # Monkeypatch the DeepModelDetector used by analyze module to our stub
    import src.api.routes.analyze as analyze_module
    analyze_module.DeepModelDetector = StubDeepModelDetector

    data = _make_test_image_bytes()

    # Run the async function synchronously
    res = asyncio.run(process_image_data(data, 'test.jpg', target_layer='layer4.1.conv2', strip_exif=True, enable_learning=False, generate_report=False))

    assert isinstance(res, dict)
    assert 'scores' in res
    assert 'deep_model_score' in res['scores']
    # deep_model_score should be numeric (float or int)
    assert isinstance(res['scores']['deep_model_score'], (int, float))
