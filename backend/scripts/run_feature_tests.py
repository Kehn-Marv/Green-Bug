import warnings
import traceback
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
            if np.isnan(val) or np.isinf(val):
                raise AssertionError(f"Found invalid numeric value: {val}")

    recurse(features)


def run_test_random_image():
    print('Running test_feature_extraction_on_random_image')
    extractor = ForensicFeatureExtractor()
    arr = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    img = image_from_array(arr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = extractor.extract_comprehensive_features(img)
        for warn in w:
            if issubclass(warn.category, RuntimeWarning):
                raise AssertionError(f'RuntimeWarning emitted: {warn.message}')
    if 'flattened_features' not in result:
        raise AssertionError('flattened_features missing in result')
    check_no_nans_in_features(result.get('flattened_features', {}))
    print('PASS')


def run_test_constant_image():
    print('Running test_feature_extraction_on_constant_image')
    extractor = ForensicFeatureExtractor()
    arr = np.ones((256, 256, 3), dtype=np.uint8) * 128
    img = image_from_array(arr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = extractor.extract_comprehensive_features(img)
        for warn in w:
            if issubclass(warn.category, RuntimeWarning):
                raise AssertionError(f'RuntimeWarning emitted: {warn.message}')
    if 'flattened_features' not in result:
        raise AssertionError('flattened_features missing in result')
    check_no_nans_in_features(result.get('flattened_features', {}))
    print('PASS')


if __name__ == '__main__':
    try:
        run_test_random_image()
    except Exception:
        print('FAILED: random image test')
        traceback.print_exc()

    try:
        run_test_constant_image()
    except Exception:
        print('FAILED: constant image test')
        traceback.print_exc()
