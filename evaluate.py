import os
import random
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import load_coco_splits
from model import UNet
from metrics import dice_coefficient, iou_score

# ──────────────────────────────────────────────────────────────────────────────
# Config  (must match train.py)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = "."
IMG_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0
MODEL_SAVE = "unet_keyboard.pth"
VIS_SAVE = "predictions_sample.png"
NUM_VIS = 6  # number of samples to visualise

THRESHOLD = 0.5  # binarisation threshold for predictions

# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # ── Device ───────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    print(f"\n{'='*55}")
    print(f" Keyboard Segmentation — U-Net Evaluation")
    print(f"{'='*55}")
    print(f" Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_SAVE):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_SAVE}'.\n"
            f"Please run `python train.py` first."
        )

    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    model.eval()
    print(f"\n Model loaded from : {MODEL_SAVE}")

    # ── Dataset split (same seed as training) ────────────────────────────────
    print(" Loading dataset …")
    _, _, test_ds = load_coco_splits(DATASET_ROOT, img_size=IMG_SIZE)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type != "cpu"),
    )
    print(f" Test set size : {len(test_ds)} samples")

    # ── Inference & metric accumulation ──────────────────────────────────────
    print("\n Running inference on test set …")
    total_dice = 0.0
    total_iou = 0.0
    n_samples = 0

    # Collect samples for visualisation
    vis_images, vis_masks, vis_preds = [], [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)  # (B, 3, H, W)
            masks = masks.to(device)  # (B, 1, H, W)

            preds = model(images)  # (B, 1, H, W) in [0, 1]

            # Accumulate metrics (weighted by batch size)
            b = images.size(0)
            total_dice += dice_coefficient(preds, masks, threshold=THRESHOLD) * b
            total_iou += iou_score(preds, masks, threshold=THRESHOLD) * b
            n_samples += b

            # Store tensors for visualisation (CPU)
            vis_images.append(images.cpu())
            vis_masks.append(masks.cpu())
            vis_preds.append(preds.cpu())

    avg_dice = total_dice / n_samples
    avg_iou = total_iou / n_samples

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f" EVALUATION RESULTS  (test set, n={n_samples})")
    print(f"{'='*55}")
    print(f"  Average Dice Coefficient : {avg_dice:.4f}")
    print(f"  Average IoU (Jaccard)    : {avg_iou:.4f}")
    print(f"{'─'*55}")

    # Interpretation
    print("\n PERFORMANCE INTERPRETATION:")
    if avg_dice > 0.85:
        print(f"  ✓ Dice = {avg_dice:.4f} > 0.85 → Strong segmentation performance.")
    elif avg_dice > 0.70:
        print(f"  ~ Dice = {avg_dice:.4f} → Moderate performance; room to improve.")
    else:
        print(f"  ✗ Dice = {avg_dice:.4f} < 0.70 → Weak performance.")
        print(f"    Consider more epochs, data augmentation, or a deeper model.")

    if avg_iou > 0.75:
        print(f"  ✓ IoU  = {avg_iou:.4f} → Good object overlap.")
    else:
        print(f"  ~ IoU  = {avg_iou:.4f} → Acceptable but improveable.")

    print()

    # ── Visualisation ─────────────────────────────────────────────────────────
    print(f" Generating visualisations ({NUM_VIS} samples) …")

    # Concatenate all batches and sample randomly
    all_images = torch.cat(vis_images, dim=0)  # (N, 3, H, W)
    all_masks = torch.cat(vis_masks, dim=0)  # (N, 1, H, W)
    all_preds = torch.cat(vis_preds, dim=0)  # (N, 1, H, W)

    n_vis = min(NUM_VIS, len(all_images))
    idxs = random.sample(range(len(all_images)), n_vis)

    fig, axes = plt.subplots(n_vis, 3, figsize=(10, n_vis * 3.2))
    if n_vis == 1:
        axes = [axes]  # ensure 2-D indexing for single row

    col_titles = ["Original Image", "Ground-Truth Mask", "Predicted Mask"]

    for row, idx in enumerate(idxs):
        # Original image: (3, H, W) float [0,1] → HWC for imshow
        img = all_images[idx].permute(1, 2, 0).numpy()

        # Ground-truth mask: (1, H, W) → (H, W)
        gt = all_masks[idx].squeeze(0).numpy()

        # Predicted mask: threshold at 0.5 → binary (H, W)
        pred = (all_preds[idx].squeeze(0).numpy() > THRESHOLD).astype(np.float32)

        for col, (data, cmap) in enumerate(
            zip([img, gt, pred], [None, "gray", "gray"])
        ):
            ax = axes[row][col]
            ax.imshow(data, cmap=cmap)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold")

    plt.suptitle(
        f"U-Net Predictions  |  Dice: {avg_dice:.4f}  |  IoU: {avg_iou:.4f}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(VIS_SAVE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Visualisation saved to : {VIS_SAVE}\n")


if __name__ == "__main__":
    main()
