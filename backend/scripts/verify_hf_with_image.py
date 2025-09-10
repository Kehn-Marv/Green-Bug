"""Smoke script: download a sample image and run HF image-classification pipeline.

Usage:
  Set MODEL_NAME env var (defaults to Hemg/Deepfake-Detection when unset)
  python verify_hf_with_image.py
"""
import os
import sys
import requests
from PIL import Image
from io import BytesIO

MODEL_NAME = os.environ.get("MODEL_NAME", "Hemg/Deepfake-Detection")
SAMPLE_URL = os.environ.get("SAMPLE_IMAGE_URL", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/640px-PNG_transparency_demonstration_1.png")


def main():
    try:
        print(f"Loading Hugging Face image-classification pipeline for: {MODEL_NAME}")
        from transformers import pipeline

        pipe = pipeline("image-classification", model=MODEL_NAME)
        print("Pipeline ready. Downloading sample image...")

        r = requests.get(SAMPLE_URL, timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize((224, 224))

        print("Running inference on sample image...")
        res = pipe(img)
        print("Inference result:")
        print(res)
        print("Done.")

    except Exception as e:
        print("Error during smoke test:", e)
        sys.exit(2)


if __name__ == '__main__':
    main()
