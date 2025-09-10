import time
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Import the real analyze router and setter helpers
from src.api.routes import analyze as analyze_module
from src.api.job_store import InMemoryJobStore


def make_mock_ingest():
    class MockIngest:
        def process_upload(self, data, filename, strip_exif=True):
            from PIL import Image
            im = Image.new('RGB', (256, 256), color=(128, 128, 128))
            return {
                'processed_image': im,
                'quality_assessment': {'score': 0.9},
                'original_size': len(data),
                'processing_timestamp': 'now',
                'file_hash': 'deadbeef'
            }
    return MockIngest()


def make_mock_feature_extractor():
    class MockFE:
        def extract_comprehensive_features(self, im):
            return {'structured_features': {}, 'flattened_features': [0.1, 0.2]}
    return MockFE()


def make_mock_attribution():
    class MockAttr:
        def analyze_attribution(self, features):
            return {'ensemble': {'predicted_family': 'none'}}
    return MockAttr()


def make_mock_robustness():
    class MockR:
        def comprehensive_robustness_test(self, im, detector_func, score):
            return {'overall_robustness': 0.8}
    return MockR()


def make_mock_self_learning():
    class MockSL:
        def evaluate_learning_candidate(self, data, features, meta):
            return {'selected': False}

        def process_consent_request(self, req):
            return {'ok': True}

        def get_comprehensive_statistics(self):
            return {'candidates': 0}

        def cleanup_expired_candidates(self):
            return {'cleaned': 0}
    return MockSL()


def make_mock_report_gen():
    class MockRG:
        def generate_comprehensive_report(self, result, original_image_path=None, case_info=None):
            return {'report': 'ok'}
    return MockRG()


def test_submit_and_poll_real_analyze_router():
    # Prepare app and mount analyze router
    app = FastAPI()

    # Inject an in-memory store so the router uses local queues instead of Redis
    analyze_module.set_job_store(InMemoryJobStore())

    app.include_router(analyze_module.router)

    # Inject test doubles to avoid heavy initialization
    analyze_module.set_ingest_filter(make_mock_ingest())
    analyze_module.set_feature_extractor(make_mock_feature_extractor())
    analyze_module.set_attribution_engine(make_mock_attribution())
    analyze_module.set_robustness_tester(make_mock_robustness())
    analyze_module.set_self_learning_system(make_mock_self_learning())
    analyze_module.set_report_generator(make_mock_report_gen())

    client = TestClient(app)

    # Create a fake file payload
    data = b"\xFF\xD8\xFF" + b"0" * 1024
    files = {'file': ('test.jpg', data, 'image/jpeg')}

    resp = client.post('/analyze/submit', files=files)
    assert resp.status_code == 200
    j = resp.json()
    job_id = j['job_id']

    # Poll result endpoint until completion
    got = None
    timeout = time.time() + 8.0
    while time.time() < timeout:
        r2 = client.get(f'/analyze/result/{job_id}')
        assert r2.status_code == 200
        j2 = r2.json()
        if j2.get('status') == 'completed' or j2.get('result') is not None:
            got = j2
            break
        time.sleep(0.05)

    assert got is not None, 'Job did not complete in time'
