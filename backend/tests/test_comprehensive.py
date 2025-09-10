import tempfile
import os
from PIL import Image
from io import BytesIO

from src.models.detector import LegacyDetector
from src.trace.attribution import AttributionIndex
from src.utils.validation import sanitize_filename
from src.utils.img import load_image_from_bytes
from src.models.face_detector import FaceDetectorSingleton


def test_legacy_detector_basics():
    img = Image.new("RGB", (128, 128), color="white")
    det = LegacyDetector()
    feats = det.features(img)
    assert isinstance(feats, dict)
    score = det.score(feats)
    assert 0.0 <= score <= 1.0


def test_attribution_index_tmpfile():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
        tf.write(b"{}")
        tf.flush()
    idx = AttributionIndex(tf.name)
    assert isinstance(idx.all_families(), list)
    os.unlink(tf.name)


def test_validation_sanitize():
    s = sanitize_filename("../../../etc/passwd")
    assert ".." not in s


def test_image_load_bytes():
    img = Image.new("RGB", (50, 50), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    loaded = load_image_from_bytes(buf.getvalue())
    assert loaded.size == (50, 50)


def test_face_detector_singleton():
    d1 = FaceDetectorSingleton()
    d2 = FaceDetectorSingleton()
    assert d1 is d2
