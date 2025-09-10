# Migration notes — Remorph API

This repository moved from feature/feature-based deepfake indicators to a canonical deep-learning model score.

What changed

- Canonical field: `deep_model_score`
  - This is the preferred, authoritative numeric score produced by the deployed deep-learning detector.
- Backwards compatibility: `legacy_deepfake_score`
  - This field has been removed from API responses in this branch. Clients should use `deep_model_score` exclusively.
  - If you depend on `legacy_deepfake_score`, update your clients to read `deep_model_score` and open an issue if you need an extended migration window.

Action for API consumers

- Migration target: read `deep_model_score` only.

- Recommended approach (pseudocode):

  if response.contains('deep_model_score'):
    use response['deep_model_score']
  else:
    use response.get('legacy_deepfake_score')

Verification and testing

- A smoke script is provided at `backend/scripts/verify_hf_model.py` that downloads the configured Hugging Face model and runs a single inference to verify availability and runtime compatibility.
- To run the smoke script locally (using the repository virtual environment):

```powershell
# activate your project's venv first (example):
& .\.venv\Scripts\Activate.ps1
# set MODEL_NAME to the pinned model id or a public HF model id
$env:MODEL_NAME = 'microsoft/deepfake-detection'
python ./backend/scripts/verify_hf_model.py
```

Notes and timeline

- `legacy_deepfake_score` is retained for compatibility and will be removed in a future major release; migrate clients to `deep_model_score` as soon as feasible.
- If you need model explainability (heatmaps / Grad-CAM), check whether your deployment uses the local Torch path or the Hugging Face pipeline; Grad-CAM support is provided for local Torch models only.

Questions

If you want me to:

- Pin a specific vetted Hugging Face model id instead of the placeholder above, say which constraints (max model size, license) you prefer, and I will update the docs and test script.

- Remove all legacy wording from non-code docs and comments, and prepare a deprecation timeline, I can create a `DEPRECATION.md` describing the schedule.

Recommended model

- Recommended (pinned) Hugging Face model id: `Hemg/Deepfake-Detection`
  - Rationale: this identifier is the project's chosen default for smoke tests and deployments in the current branch. It provides a good balance of accuracy and runtimes in typical forensic scenarios. If you prefer a different model (for size, license, or latency constraints), tell me and I'll pin that instead and update the smoke script.

How to change the pinned model

Set `MODEL_NAME` in your environment or adjust `backend/src/config.py` to choose a different Hugging Face model id. The smoke script will attempt to download whatever id you provide.
