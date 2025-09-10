# DEPRECATION NOTICE — legacy_deepfake_score

Summary

- Field: `legacy_deepfake_score`
- Current status: REMOVED in this branch; clients must use `deep_model_score`.

Migration guidance

- All clients must update to read `deep_model_score` and ignore `legacy_deepfake_score`.

Compatibility

- This change is breaking for clients still expecting `legacy_deepfake_score`. If you manage API clients that depend on the removed field, open an issue or a PR to request a longer migration window or to report migration blockers.

Timeline and policy

- Deprecation window: at least 3 months from the date the deprecation note is published in the main repository.
- Removal: planned for the following major release (v2.0). Exact dates will be posted in repo release notes and in `DEPRECATION.md`.

Contact and support

If you manage API clients that depend on `legacy_deepfake_score`, open an issue or a PR to request a longer migration window or to report migration blockers.
