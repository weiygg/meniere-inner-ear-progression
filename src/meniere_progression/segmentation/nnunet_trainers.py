from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from monai.losses import SoftclDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn


class BoundedBiasFieldTransform(ImageOnlyTransform):
    """Smooth multiplicative 3-D bias field with a frozen small amplitude."""

    def __init__(self, coefficient_range: tuple[float, float] = (-0.15, 0.15)) -> None:
        super().__init__()
        self.coefficient_range = coefficient_range

    def get_parameters(self, **data_dict: Any) -> dict[str, torch.Tensor]:
        image = data_dict["image"]
        low, high = self.coefficient_range
        coefficients = torch.empty(9, device=image.device, dtype=image.dtype).uniform_(low, high)
        return {"coefficients": coefficients}

    def _apply_to_image(self, img: torch.Tensor, **params: Any) -> torch.Tensor:
        if img.ndim != 4:
            raise ValueError(f"Bias field expects (C, X, Y, Z), got {tuple(img.shape)}")
        axes = [torch.linspace(-1, 1, size, device=img.device, dtype=img.dtype) for size in img.shape[1:]]
        x, y, z = torch.meshgrid(*axes, indexing="ij")
        c = params["coefficients"]
        log_field = (
            c[0] * x
            + c[1] * y
            + c[2] * z
            + c[3] * x * y
            + c[4] * x * z
            + c[5] * y * z
            + c[6] * x.square()
            + c[7] * y.square()
            + c[8] * z.square()
        )
        field = torch.exp(log_field)
        field = field / field.mean().clamp_min(torch.finfo(field.dtype).eps)
        return img * field.unsqueeze(0)


class nnUNetTrainerProtocolV2M2(nnUNetTrainer):
    """M2: nnU-Net defaults plus one prespecified bounded bias field."""

    @staticmethod
    def get_training_transforms(*args: Any, **kwargs: Any):
        composed = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        composed.transforms.insert(
            2,
            RandomTransform(
                BoundedBiasFieldTransform(coefficient_range=(-0.15, 0.15)),
                apply_probability=0.20,
            ),
        )
        return composed


class DiceCEPlusSoftClDice(nn.Module):
    def __init__(self, base_loss: nn.Module, weight: float = 0.1, iterations: int = 3) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weight = weight
        self.cldice = SoftclDiceLoss(iter_=iterations, smooth=1.0)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        label = target[:, 0].long()
        one_hot = F.one_hot(label, num_classes=net_output.shape[1]).movedim(-1, 1).to(net_output.dtype)
        probabilities = torch.softmax(net_output, dim=1)
        return self.base_loss(net_output, target) + self.weight * self.cldice(one_hot, probabilities)


class nnUNetTrainerProtocolV2M3(nnUNetTrainerProtocolV2M2):
    """M3: M2 augmentation plus frozen Dice/CE + 0.1 soft-clDice."""

    def _build_loss(self):
        loss = super()._build_loss()
        if isinstance(loss, DeepSupervisionWrapper):
            return DeepSupervisionWrapper(
                DiceCEPlusSoftClDice(loss.loss, weight=0.1, iterations=3),
                loss.weight_factors,
            )
        return DiceCEPlusSoftClDice(loss, weight=0.1, iterations=3)
