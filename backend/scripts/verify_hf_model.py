"""Small smoke script to verify Hugging Face model download and run one inference.

Usage:
    python verify_hf_model.py

Reads MODEL_NAME from environment (falls back to 'microsoft/deepfake-detection').
"""
import os
import sys
from PIL import Image

MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deepfake-detection")


def main():
    try:
        print(f"Attempting to load Hugging Face image-classification pipeline for model: {MODEL_NAME}")
        # import lazily to keep this optional
        from transformers import pipeline

        pipe = pipeline("image-classification", model=MODEL_NAME)
        print("Model pipeline created. Running a test inference on a generated image...")

        # Create a tiny test image (RGB white 224x224)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        results = pipe(img)

        print("Inference result:")
        print(results)
        print("Smoke test succeeded. The model was downloaded and a single inference ran OK.")

    except Exception as e:
        print("Smoke test failed:", str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
