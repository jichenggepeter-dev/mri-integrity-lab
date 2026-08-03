# Project Guidance

## Product intent

Build a research interface for medical-imaging AI researchers and data-quality reviewers. Tumor
classification is the first displayed result, but it must always be interpreted together with the
image-integrity result.

## Architecture rules

- Use Python, PyTorch, Streamlit, and uv.
- Keep data preparation, manipulation, models, training, evaluation, explainability, and UI logic
  in separate modules under `src/mri_integrity_lab/`.
- Generate synthetic manipulations only after the original-image split is fixed.
- Never describe an integrity score as proof of fraud or a tumor score as clinical diagnosis.
- Keep raw images out of Git.

## Agent operating rules

- Keep changes inside this repository.
- Add focused tests for deterministic data logic and model contracts.
- Do not change measured metrics by hand.
- Do not add large dependencies without a concrete need.
- Verify critical Streamlit states in a browser before completion.

## Current phase

- Goal: Create the tested data and model foundation.
- In scope: repository setup, preprocessing, manipulation, model contracts, CLI, and tests.
- Out of scope: report prose and clinical claims.
- Allowed edit paths: all paths in this repository.
- Required verification: `uv run pytest` and `uv run ruff check .`.

