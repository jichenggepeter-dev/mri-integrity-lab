from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from mri_integrity_lab.explainability import overlay_heatmap
from mri_integrity_lab.inference import (
    ReliabilityState,
    load_model_bundle,
    run_inference,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "multitask.pt"


@st.cache_resource
def get_model():
    return load_model_bundle(CHECKPOINT_PATH)


def reliability_label(state: ReliabilityState) -> tuple[str, str]:
    if state is ReliabilityState.RESEARCH_READY:
        return "Research review ready", "status-ready"
    if state is ReliabilityState.NEEDS_REVIEW:
        return "Needs image-quality review", "status-review"
    return "Integrity result uncertain", "status-uncertain"


st.set_page_config(
    page_title="MRI Integrity Lab",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172126;
        --muted: #5b686e;
        --line: #d8e0e3;
        --teal: #006d67;
        --amber: #a15c00;
        --red: #a33b32;
    }
    .block-container {max-width: 1180px; padding-top: 1.4rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: 0; color: var(--ink);}
    h1 {font-size: 2rem; margin-bottom: 0;}
    [data-testid="stMetric"] {border-top: 2px solid var(--line); padding-top: 0.8rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    .workspace-note {color: var(--muted); margin: 0.15rem 0 1.2rem;}
    .result-status {padding: 0.7rem 0.8rem; border-left: 4px solid; font-weight: 650;}
    .status-ready {background: #eaf5f2; border-color: var(--teal); color: #064f4a;}
    .status-uncertain {background: #fff4de; border-color: var(--amber); color: #6f4000;}
    .status-review {background: #faeae8; border-color: var(--red); color: #762820;}
    .research-limit {border-top: 1px solid var(--line); margin-top: 1.5rem; padding-top: 0.8rem;
                     color: var(--muted); font-size: 0.88rem;}
    [data-testid="stFileUploader"] section {border-radius: 4px; border-color: var(--line);}
    [data-testid="stImage"] {width: 100%; max-width: 420px;}
    [data-testid="stImage"] img {width: 100%; height: auto;}
    button {border-radius: 4px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MRI Integrity Lab")
st.markdown(
    '<p class="workspace-note">Brain-image classification with synthetic integrity screening</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model")
    st.caption("Semantic CNN + local forensic residual branch")
    st.divider()
    st.subheader("Input")
    st.caption("JPG or PNG image, standardized to 128 x 128 grayscale")
    st.divider()
    st.subheader("Decision rule")
    st.caption("Tumor probability is interpreted together with image-integrity risk.")

if not CHECKPOINT_PATH.is_file():
    st.error("The trained multitask checkpoint is not available yet.")
    st.code(
        "uv run mri-integrity train --model multitask --data-root /path/to/dataset",
        language="bash",
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Brain image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("Upload one brain image to run the research model.")
    st.stop()

try:
    uploaded_image = Image.open(uploaded_file)
    uploaded_image.load()
except (OSError, UnidentifiedImageError):
    st.error("The uploaded file could not be decoded as an image.")
    st.stop()

with st.spinner("Running both model heads and attention maps..."):
    result = run_inference(uploaded_image, get_model())

image_column, result_column = st.columns([1.05, 1.25], gap="large")
with image_column:
    st.subheader("Input image")
    st.image(uploaded_image, width="stretch")

with result_column:
    st.subheader("Tumor classification")
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Predicted class", result.predicted_class)
    metric_right.metric("Tumor probability", f"{result.tumor_probability:.1%}")
    st.progress(result.tumor_probability, text="Tumor probability")

    st.subheader("Image integrity")
    status_text, status_class = reliability_label(result.reliability)
    st.markdown(
        f'<div class="result-status {status_class}">{status_text}</div>',
        unsafe_allow_html=True,
    )
    st.metric("Synthetic manipulation risk", f"{result.integrity_probability:.1%}")
    if result.reliability is not ReliabilityState.RESEARCH_READY:
        st.warning("Treat the tumor classification as provisional until the image is reviewed.")

st.divider()
st.subheader("Model attention")
tumor_tab, integrity_tab = st.tabs(["Tumor head", "Integrity head"])
with tumor_tab:
    st.image(
        overlay_heatmap(result.standardized_image, result.tumor_heatmap),
        caption="Grad-CAM for the tumor class",
        width="stretch",
    )
with integrity_tab:
    st.image(
        overlay_heatmap(result.standardized_image, result.integrity_heatmap),
        caption="Grad-CAM for synthetic manipulation risk",
        width="stretch",
    )

st.markdown(
    """
    <p class="research-limit">
    Research prototype only. The model was trained on a public two-dimensional benchmark and
    synthetic manipulations. It is not validated for diagnosis, treatment, or forensic conclusions.
    </p>
    """,
    unsafe_allow_html=True,
)
