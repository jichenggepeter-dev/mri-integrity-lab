# Task plan

## Goal

Deliver a reproducible CPU-friendly PyTorch project that classifies brain images, screens synthetic
image manipulation, visualizes model attention, and exposes the results in Streamlit.

## In scope

- Deduplicated 70/15/15 original-image split.
- Synthetic copy-move, occlusion, and intensity/noise manipulations after splitting.
- Baseline tumor CNN, improved tumor CNN, and multi-task CNN.
- Tumor, integrity, and robustness metrics.
- Grad-CAM and a research-oriented Streamlit interface.
- uv setup, tests, documentation, saved artifacts, and presentation-ready figures.

## Out of scope

- DICOM ingestion, clinical deployment, pixel-accurate tamper segmentation, patient use, and claims
  of diagnostic or forensic validity.

## Key decisions

- Input: 128 x 128 grayscale image.
- Split: 70/15/15 by deduplicated original image before any manipulation.
- Primary UI result: tumor classification, visibly bound to the integrity status.
- Training device: CPU by default, optional MPS when explicitly requested.
- Package management: uv with a committed lockfile.

## Verification

- Unit tests for preprocessing, deterministic manipulation, model outputs, and reliability status.
- Full data audit with zero cross-split duplicate hashes.
- Saved metrics generated from held-out test data only.
- Streamlit smoke test and desktop/mobile browser screenshots.

