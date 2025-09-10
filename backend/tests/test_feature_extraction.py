import warnings
import numpy as np
from PIL import Image
from src.core.forensic_feature_extractor import ForensicFeatureExtractor


def image_from_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(arr))


def check_no_nans_in_features(features: dict):
    def recurse(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                recurse(v)
        elif isinstance(obj, (int, float, np.floating, np.integer)):
            val = float(obj)
            assert not np.isnan(val), f"Found NaN value: {val}"
            assert not np.isinf(val), f"Found Inf value: {val}"

    recurse(features)


def test_feature_extraction_on_random_image():
    extractor = ForensicFeatureExtractor()

    # Create a random image (safe range 0-255)
    arr = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    img = image_from_array(arr)

    # Capture warnings and ensure no RuntimeWarnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = extractor.extract_comprehensive_features(img)

        # Ensure no RuntimeWarning about invalid value encountered
        for warn in w:
            assert not issubclass(warn.category, RuntimeWarning), f"Found runtime warning: {warn.message}"

    assert 'flattened_features' in result
    check_no_nans_in_features(result.get('flattened_features', {}))


def test_feature_extraction_on_constant_image():
    extractor = ForensicFeatureExtractor()

    # Create a constant image which can trigger zero-variance edge cases
    arr = np.ones((256, 256, 3), dtype=np.uint8) * 128
    img = image_from_array(arr)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = extractor.extract_comprehensive_features(img)
        for warn in w:
            assert not issubclass(warn.category, RuntimeWarning), f"Found runtime warning: {warn.message}"

    assert 'flattened_features' in result
    check_no_nans_in_features(result.get('flattened_features', {}))
