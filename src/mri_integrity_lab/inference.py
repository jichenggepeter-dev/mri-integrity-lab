from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .explainability import GradCAM
from .models import MultiTaskCNN, MultiTaskOutput, build_model
from .preprocessing import standardize_image


class ReliabilityState(StrEnum):
    RESEARCH_READY = "research_ready"
    UNCERTAIN = "uncertain"
    NEEDS_REVIEW = "needs_review"


def reliability_state(
    integrity_probability: float,
    *,
    lower_threshold: float = 0.35,
    upper_threshold: float = 0.65,
) -> ReliabilityState:
    if not 0.0 <= integrity_probability <= 1.0:
        raise ValueError("integrity_probability must be between 0 and 1.")
    if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
        raise ValueError("Reliability thresholds must be ordered within [0, 1].")
    if integrity_probability <= lower_threshold:
        return ReliabilityState.RESEARCH_READY
    if integrity_probability >= upper_threshold:
        return ReliabilityState.NEEDS_REVIEW
    return ReliabilityState.UNCERTAIN


@dataclass(frozen=True)
class ModelBundle:
    model: MultiTaskCNN
    image_size: int
    mean: float
    std: float
    device: torch.device


@dataclass(frozen=True)
class InferenceResult:
    tumor_probability: float
    integrity_probability: float
    predicted_class: str
    reliability: ReliabilityState
    standardized_image: np.ndarray
    tumor_heatmap: np.ndarray
    integrity_heatmap: np.ndarray


def load_model_bundle(checkpoint_path: Path, device_name: str = "cpu") -> ModelBundle:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "multitask":
        raise ValueError("The Streamlit application requires a multitask checkpoint.")
    model = build_model("multitask", dropout=checkpoint["train_config"]["dropout"])
    if not isinstance(model, MultiTaskCNN):
        raise TypeError("Expected MultiTaskCNN from the model factory.")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    normalization = checkpoint["normalization"]
    return ModelBundle(
        model=model,
        image_size=int(checkpoint["image_size"]),
        mean=float(normalization["mean"]),
        std=float(normalization["std"]),
        device=device,
    )


def prepare_input(image: Image.Image, bundle: ModelBundle) -> tuple[np.ndarray, torch.Tensor]:
    standardized = standardize_image(image, image_size=bundle.image_size)
    pixels = np.asarray(standardized, dtype=np.float32) / 255.0
    normalized = (pixels - bundle.mean) / max(bundle.std, 1e-6)
    tensor = torch.from_numpy(normalized.copy()).unsqueeze(0).unsqueeze(0).to(bundle.device)
    return pixels, tensor


def run_inference(image: Image.Image, bundle: ModelBundle) -> InferenceResult:
    standardized, tensor = prepare_input(image, bundle)
    with torch.no_grad():
        output: MultiTaskOutput = bundle.model(tensor)
        tumor_probability = float(torch.softmax(output.tumor_logits, 1)[0, 1].cpu())
        integrity_probability = float(torch.softmax(output.integrity_logits, 1)[0, 1].cpu())

    tumor_gradcam = GradCAM(bundle.model, bundle.model.gradcam_layer)
    integrity_gradcam = GradCAM(bundle.model, bundle.model.integrity_gradcam_layer)
    try:
        tumor_heatmap = tumor_gradcam.generate(tensor, task="tumor", target_class=1)
        integrity_heatmap = integrity_gradcam.generate(
            tensor, task="integrity", target_class=1
        )
    finally:
        tumor_gradcam.close()
        integrity_gradcam.close()

    return InferenceResult(
        tumor_probability=tumor_probability,
        integrity_probability=integrity_probability,
        predicted_class="Tumor" if tumor_probability >= 0.5 else "Normal",
        reliability=reliability_state(integrity_probability),
        standardized_image=standardized,
        tumor_heatmap=tumor_heatmap,
        integrity_heatmap=integrity_heatmap,
    )
