from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib import colormaps
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._capture_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output

    def _capture_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        image: torch.Tensor,
        *,
        task: str = "tumor",
        target_class: int = 1,
    ) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        output = self.model(image)
        # Attribute checks survive Streamlit hot reloads, where NamedTuple class identity changes.
        if hasattr(output, "tumor_logits") and hasattr(output, "integrity_logits"):
            logits = output.tumor_logits if task == "tumor" else output.integrity_logits
        else:
            logits = output
        logits[:, target_class].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        return heatmap.detach().cpu().numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    grayscale = np.asarray(image, dtype=np.float32)
    if grayscale.max() > 1.0:
        grayscale /= 255.0
    base = np.repeat(grayscale[..., None], 3, axis=2)
    color = colormaps["magma"](np.clip(heatmap, 0, 1))[..., :3]
    return np.clip((1.0 - alpha) * base + alpha * color, 0.0, 1.0)
