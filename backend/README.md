# Installation for Hugging Face model support

This project uses an optional Hugging Face pipeline for deepfake detection. The HF dependency is optional but recommended for improved accuracy.

Install backend dependencies (recommended inside a virtualenv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt
```

If you only want the HF extras (when not using the full requirements file):

```powershell
python -m pip install transformers[torch]
```

Recommended model

- Start with a vetted image-classification model. Example placeholder:
  - `microsoft/deepfake-detection` (replace with a specific, evaluated model you trust)

Usage notes

- Set environment variables in production to select model provider and model name (example in `.env`):
  - `MODEL_PROVIDER=huggingface`
  - `MODEL_NAME=microsoft/deepfake-detection`

- If you use a local PyTorch checkpoint instead, leave `MODEL_PROVIDER=local` and set `WEIGHTS_PATH` to the checkpoint file.

Grad-CAM

- Grad-CAM visualizations are supported for local PyTorch models. HF pipelines may not expose intermediate activations required for Grad-CAM.

Troubleshooting

- If you see import errors for `transformers`, ensure you've installed the package into the active Python environment and restarted the server process.

Migration note (API consumers)
-----------------------------

We replaced the old feature/feature-based score field with a canonical deep-learning score. API responses now include:

- `deep_model_score` (float): the canonical probability from the deep-learning detector (preferred).
- `legacy_deepfake_score` (float, optional): kept for backward compatibility where older clients still expect the previous field.

Please migrate clients to use `deep_model_score`. The `legacy_deepfake_score` field will be removed in a future major release; migrate as soon as practical.

Recommended Hugging Face model
-----------------------------

We recommend using a vetted image classification model specialized for deepfake detection. The README uses a placeholder example model id below — replace it with a model you have evaluated and trust for your use case.

- Example (placeholder): `microsoft/deepfake-detection`
 - Example (pinned default): `Hemg/Deepfake-Detection`

To pin the model used by the smoke script and by default, set the environment variable in your environment or `.env` file:

``powershell
$env:MODEL_PROVIDER = 'huggingface'
$env:MODEL_NAME = 'Hemg/Deepfake-Detection'  # pinned default model id used in this repo
``

Smoke verification script
-------------------------

We include a small verification script at `backend/scripts/verify_hf_model.py` that will attempt to download the pinned Hugging Face model and run a single inference on a generated test image.

Run it inside the backend virtualenv:

```powershell
.\.venv\Scripts\Activate.ps1; Set-Location -Path .\backend; python .\scripts\verify_hf_model.py
```

The script reads `MODEL_NAME` from the environment and falls back to the recommended model id above.
