import torch


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute the Dice coefficient between predictions and targets.

        Dice = 2 * |P ∩ T| / (|P| + |T|)

    Args:
        preds     : Predicted probabilities or logits, shape (B, 1, H, W) or (B, H, W).
        targets   : Ground-truth binary masks,         shape (B, 1, H, W) or (B, H, W).
        threshold : Threshold to binarise `preds`.  Default: 0.5.
        smooth    : Small constant to prevent division by zero.

    Returns:
        Scalar float – mean Dice over the batch.
    """
    # Threshold predictions to binary
    preds_bin = (preds > threshold).float()

    # Flatten spatial + channel dims; keep batch dim
    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets.float().view(targets.size(0), -1)

    # Element-wise intersection
    intersection = (preds_flat * targets_flat).sum(dim=1)

    # Dice per sample, then average over the batch
    dice = (2.0 * intersection + smooth) / (
        preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth
    )
    return dice.mean().item()


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute the Intersection-over-Union (Jaccard Index) between predictions
    and targets.

        IoU = |P ∩ T| / |P ∪ T|

    Args:
        preds     : Predicted probabilities or logits, shape (B, 1, H, W) or (B, H, W).
        targets   : Ground-truth binary masks,         shape (B, 1, H, W) or (B, H, W).
        threshold : Threshold to binarise `preds`.  Default: 0.5.
        smooth    : Small constant to prevent division by zero.

    Returns:
        Scalar float – mean IoU over the batch.
    """
    # Threshold predictions to binary
    preds_bin = (preds > threshold).float()

    # Flatten
    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets.float().view(targets.size(0), -1)

    # Intersection and Union
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
