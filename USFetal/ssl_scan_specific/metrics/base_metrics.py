import torch
from monai.metrics import SSIMMetric
from torchmetrics.image import PeakSignalNoiseRatio
import monai.metrics as Metrics
import torch.nn.functional as F


class SSIMMetricWrapper:
    def __init__(self, spatial_dims=3):
        self.metric = SSIMMetric(spatial_dims=spatial_dims)

    def __call__(self, preds, targets):
        return self.metric(preds, targets).mean().item()


class PSNRMetricWrapper:
    def __init__(self, data_range=1.0, device="cpu"):
        self.metric = PeakSignalNoiseRatio(data_range=data_range).to(device)

    def __call__(self, preds, targets):
        return self.metric(preds, targets).mean().item()


class DiceCoefficient:
    def __init__(self, smooth=1e-5):
        self.smooth = smooth

    def __call__(self, preds, targets):
        preds = preds.float()
        targets = targets.float()
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.item()


class DiceMetric:
    def __init__(self, include_background=True, reduction="mean", get_not_nans=False):
        self.metric = Metrics.DiceMetric(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    def __call__(self, preds, targets):
        dice_scores = self.metric(y_pred=preds, y=targets)
        return dice_scores.mean().item()


class IoUMetric:
    def __init__(self, include_background=False, ignore_empty=True):
        self.include_background = include_background
        self.ignore_empty = ignore_empty

    def __call__(self, preds, targets):
        iou = Metrics.compute_iou(
            y_pred=preds,
            y=targets,
            include_background=self.include_background,
            ignore_empty=self.ignore_empty,
        )
        return iou.mean().item()

