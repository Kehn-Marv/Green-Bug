Cleanup proposal and notes

Safe-to-delete (automatable):
- All `__pycache__` directories and `*.pyc`/`*.pyo` files anywhere.
- `.pytest_cache/` at project root and `/backend`.
- `remorph_backend.egg-info/` at `/backend` and `/backend/src` (build metadata).
- `outputs/` at project root and `/backend/outputs/` (generated data).
- `frontend/node_modules/` (if present) — large and can be reinstalled via `npm install`.
- `.venv/` (local virtual environment) — recreate with `python -m venv .venv`/install requirements.

Requires review (do not delete automatically):
- `backend/data/fingerprints.json` and `data/fingerprints.json` — referenced in `backend/src/config.py`; keep.
- `backend/weights/` — machine learning weights; keep unless you have copies elsewhere.
- `frontend/dist` or built assets — if present, ensure they aren't needed before deleting.
- Any `scripts/` files — some are used for running tests or maintenance; inspect before removing.

How to proceed safely:
1. Review this file and the proposed safe-to-delete list.
2. Run `scripts/cleanup_unused.ps1` manually (it only removes the clearly safe items).
3. Commit `.gitignore` changes to ignore these artifacts going forward.

If you want, I can:
- Run the cleanup now (with your confirmation).
- Add/modify `.gitignore` to cover common artifacts.
- Remove any additional files you identify after review.
