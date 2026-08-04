from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from mri_integrity_lab.explainability import GradCAM
from mri_integrity_lab.inference import ModelBundle, run_inference
from mri_integrity_lab.models import MultiTaskCNN


class ReloadedOutput(NamedTuple):
    tumor_logits: torch.Tensor
    integrity_logits: torch.Tensor


class ReloadedMultiTaskCNN(MultiTaskCNN):
    def forward(self, images: torch.Tensor) -> ReloadedOutput:
        output = super().forward(images)
        return ReloadedOutput(output.tumor_logits, output.integrity_logits)


def test_gradcam_accepts_multitask_output_after_module_reload() -> None:
    model = ReloadedMultiTaskCNN().eval()
    gradcam = GradCAM(model, model.gradcam_layer)
    try:
        heatmap = gradcam.generate(torch.randn(1, 1, 32, 32), task="tumor")
    finally:
        gradcam.close()

    assert heatmap.shape == (32, 32)


def test_inference_can_run_twice_without_leaking_hooks() -> None:
    model = MultiTaskCNN().eval()
    bundle = ModelBundle(
        model=model,
        image_size=32,
        mean=0.5,
        std=0.25,
        device=torch.device("cpu"),
        integrity_lower_threshold=0.35,
        integrity_upper_threshold=0.65,
    )
    image = Image.fromarray(np.full((32, 32), 127, dtype=np.uint8), mode="L")

    first = run_inference(image, bundle)
    second = run_inference(image, bundle)

    assert first.predicted_class == second.predicted_class
    assert not model.gradcam_layer._forward_hooks
    assert not model.integrity_gradcam_layer._forward_hooks
