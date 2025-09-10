import os

# Core paths
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
FINGERPRINTS_PATH = os.getenv("FINGERPRINTS_PATH", "data/fingerprints.json")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/detector.pt")

# Model configuration: provider can be 'huggingface' or 'local_torch'
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "huggingface")
# For huggingface use a model id like 'microsoft/deepfake-detection' or any image-classification model
MODEL_NAME = os.getenv("MODEL_NAME", "Hemg/Deepfake-Detection")

# API Configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))
MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", "224"))

# Detection thresholds
FACE_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.90"))
QUALITY_MIN_SIDE = int(os.getenv("QUALITY_MIN_SIDE", "224"))


# (Legacy) Feature-based scoring weights removed in favor of deep model configuration

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def validate_config():
    """Validate configuration on startup"""
    errors = []

    if MAX_FILE_SIZE_MB <= 0:
        errors.append("MAX_FILE_SIZE_MB must be positive")

    if MAX_IMAGE_DIMENSION <= MIN_IMAGE_DIMENSION:
        errors.append("MAX_IMAGE_DIMENSION must be greater than MIN_IMAGE_DIMENSION")

    if not (0.0 <= FACE_CONFIDENCE_THRESHOLD <= 1.0):
        errors.append("FACE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")

    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    return True
