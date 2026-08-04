# MRI Integrity Lab

MRI Integrity Lab is a CPU-friendly PyTorch research prototype for medical-imaging researchers and
data-quality reviewers. It classifies a brain image as `tumor` or `normal`, screens it for patterns
resembling synthetic manipulation, and shows task-specific Grad-CAM views in Streamlit.

The software is not a medical device and must not be used for diagnosis, treatment, or allegations
of image fraud.

## Research question

How do controlled image manipulations affect CNN-based brain-tumor predictions, and can a compact
PyTorch model flag those manipulations while preserving useful tumor classification?

## Results

All values below use the held-out test split after standardized-pixel deduplication.

| Experiment | Tumor accuracy | Tumor ROC-AUC | Integrity accuracy | Integrity ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Baseline CNN | 69.32% | 0.781 | - | - |
| Improved CNN | 94.36% | 0.988 | - | - |
| Final multi-task model | 94.36% | 0.988 | 78.28% | 0.870 |

Across 603 clean/manipulated test pairs, synthetic manipulation changed the binary tumor prediction
in 4.48% of cases. Occlusion produced the largest mean probability shift and flip rate. The final
integrity threshold was selected on validation data with Youden's J statistic, never on test labels.

## Method

- Audit and decode all 4,600 source files; explicitly exclude one non-brain image.
- Hash standardized pixels and remove 585 duplicate copies before splitting.
- Create a stratified 70/15/15 split with 2,809 train, 602 validation, and 603 test originals.
- Generate deterministic copy-move, occlusion, and intensity/noise variants after splitting.
- Compare a 6K-parameter baseline with a 154K-parameter semantic CNN.
- Initialize the final tumor branch from the improved model and freeze it.
- Train a local residual branch on paired clean/manipulated images for integrity screening.
- Evaluate tumor probability shift and prediction flips for every test pair.

The archived failed integrity experiments under `artifacts/v1_global_pool/` and
`artifacts/v2_residual_unpaired/` document why local features alone were insufficient: paired
training was the decisive change.

## Setup

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
```

Download the [Brain Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)
separately. The images are not redistributed in this repository. See
[`data/README.md`](data/README.md) for the expected layout.

## Reproduce

Set `DATASET` to the directory containing the two class folders:

```bash
export DATASET="/path/to/Brain Tumor Data Set"

uv run mri-integrity audit \
  --source "$DATASET" \
  --output-dir data/processed \
  --exclusions data/exclusions.csv

uv run mri-integrity train \
  --model baseline --data-root "$DATASET" --epochs 10

uv run mri-integrity train \
  --model improved --data-root "$DATASET" --epochs 15

uv run mri-integrity train \
  --model multitask --data-root "$DATASET" --epochs 10 --batch-size 64 \
  --learning-rate 0.001 --integrity-loss-weight 1.0 \
  --initial-checkpoint artifacts/improved.pt --freeze-tumor-backbone

uv run mri-integrity calibrate --data-root "$DATASET"
uv run mri-integrity report --data-root "$DATASET"
```

The complete CPU run took about 25 minutes on the development machine. Saved checkpoints, histories,
predictions, metrics, and presentation-ready figures are included for inspection.

## Web interface

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`, upload a JPG or PNG brain image, and inspect the tumor result together
with the integrity status. The interface keeps the tumor result provisional when integrity risk is
uncertain or high.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/mri_integrity_lab/` | Data audit, tampering, models, training, evaluation, inference |
| `app.py` | Streamlit research interface |
| `artifacts/` | Checkpoints, metrics, histories, held-out predictions |
| `reports/figures/` | Reproducible experiment figures |
| `reports/ui/` | Verified desktop and mobile interface screenshots |
| `reports/evidence_pack.md` | Factual notes for the student's independent report writing |
| `presentation/MRI_Integrity_Lab_Jicheng_Ge.pptx` | Ten-slide presentation deck |
| `tests/` | Unit and Streamlit smoke tests |

## Limitations

- Labels are binary image-level classes, not diagnoses or tumor subtypes.
- The source does not include patient identifiers, so patient-group splitting is impossible.
- Integrity evaluation covers three synthetic transformations, not real acquisition failures or
  malicious edits.
- A high integrity score indicates similarity to training manipulations; it does not prove fraud.
- No external or prospective clinical validation has been performed.

## Academic use

Student: Jicheng Ge (`jg5013`). Coding-agent usage is disclosed in [`AI_USAGE.md`](AI_USAGE.md).
The required three-to-four-page report must be written independently by the student.
