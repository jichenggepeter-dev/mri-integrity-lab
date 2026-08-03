import torch

from mri_integrity_lab.models import BaselineCNN, ImprovedCNN, MultiTaskCNN


def test_single_task_models_return_two_class_logits() -> None:
    batch = torch.randn(3, 1, 128, 128)

    assert BaselineCNN()(batch).shape == (3, 2)
    assert ImprovedCNN()(batch).shape == (3, 2)


def test_multi_task_model_returns_separate_two_class_heads() -> None:
    batch = torch.randn(3, 1, 128, 128)
    model = MultiTaskCNN()
    output = model(batch)

    assert output.tumor_logits.shape == (3, 2)
    assert output.integrity_logits.shape == (3, 2)
    assert model.gradcam_layer is not model.integrity_gradcam_layer
