# MRI Integrity Lab

MRI Integrity Lab is a CPU-friendly PyTorch research prototype for two related tasks:

1. Classify a supplied brain image as `tumor` or `normal`.
2. Screen the image for patterns resembling synthetic manipulation.

The intended users are medical-imaging AI researchers and data-quality reviewers. The software is
not a medical device and must not be used for diagnosis, treatment, or allegations of fraud.

## Research question

How do synthetic image manipulations affect CNN-based brain-tumor predictions, and can multi-task
learning improve both tumor classification and image-integrity screening?

## Planned experiments

- Leakage-aware baseline CNN on deduplicated images.
- Improved single-task CNN for tumor classification.
- Multi-task CNN with a shared encoder and separate tumor and integrity heads.
- Deterministic copy-move, local occlusion, and intensity/noise manipulations.
- Robustness analysis based on probability shift and prediction-flip rate.
- Grad-CAM views for both model heads.

## Local setup

```bash
uv sync
uv run pytest
uv run mri-integrity --help
uv run streamlit run app.py
```

See [data/README.md](data/README.md) for dataset setup. Reproducible commands and measured results
will be added as each experiment is completed.

## Academic use

This repository extends an earlier course baseline with new data splitting, synthetic manipulation,
multi-task learning, robustness evaluation, explainability, and a web interface. Coding-agent usage
is disclosed in [AI_USAGE.md](AI_USAGE.md). The course report must be written independently by the
student.

