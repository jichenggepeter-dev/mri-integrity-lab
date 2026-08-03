# Dataset setup

The project uses the binary Brain Tumor Dataset published by Preet Viradiya on Kaggle:

- Dataset page: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
- Dataset DOI: https://doi.org/10.3740/KAGGLE/DSV/1183165
- Expected images: 4,600 total
- Expected classes: 2,513 `Brain Tumor` and 2,087 `Healthy`

The repository does not redistribute the images. Place the downloaded class folders under:

```text
data/raw/Brain Tumor Data Set/Brain Tumor/
data/raw/Brain Tumor Data Set/Healthy/
```

Alternatively, pass an existing dataset directory to the audit command. The audit pipeline decodes
every image, converts it to the standardized 128 x 128 grayscale representation, groups analytical
duplicates by SHA-256 pixel hash, and writes a leakage-aware manifest under `data/processed/`.

The source dataset contains mixed file formats, dimensions, acquisition views, and incomplete
clinical metadata. Results therefore describe this public benchmark only and do not establish
clinical validity.

`exclusions.csv` records source-quality exclusions separately from model labels. The initial audit
confirmed that `Brain Tumor/Cancer (2040).jpg` is an abdominal cross-sectional image rather than a
brain image. Keeping the exclusion explicit makes the decision reviewable and reproducible.
