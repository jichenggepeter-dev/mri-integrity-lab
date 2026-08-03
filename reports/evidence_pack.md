# Evidence Pack (factual notes only)

This file contains measured facts and source pointers. It is not a report draft. The student must
write the submitted 3-4 page report independently.

## Identity

- Student: Jicheng Ge
- UNI: jg5013

## Data audit

- Public source: Preet Viradiya, Brain Tumor Dataset on Kaggle.
- Discovered/readable files: 4600
- Explicitly excluded non-brain image: 1
- Duplicate copies removed: 585
- Deduplicated analytical images: 4014
- Split counts: {'train': 2809, 'test': 603, 'validation': 602}

## Held-out test facts

- Baseline tumor accuracy: 0.6932
- Improved tumor accuracy: 0.9436
- Final clean-image tumor accuracy: 0.9436
- Final tumor ROC-AUC: 0.9880
- Integrity accuracy: 0.7828
- Integrity ROC-AUC: 0.8696
- Paired diagnosis flip rate: 0.0448
- Mean absolute tumor-probability shift: 0.0471

## Interpretation constraints

- The labels are image-level binary classes, not clinical diagnoses or tumor subtypes.
- Integrity labels are synthetic and cover copy-move, local occlusion, and intensity/noise only.
- The source contains no patient identifiers, so deduplication is image-hash based rather than
  patient-group based.
- The model is not externally validated and is not a medical device or forensic detector.
- Do not claim that a high integrity score proves malicious alteration.

## Source links

- Dataset: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
- Dataset DOI: https://doi.org/10.3740/KAGGLE/DSV/1183165
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/
