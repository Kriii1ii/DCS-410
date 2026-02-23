"""
train.py
--------
Training script for Binary U-Net on the Keyboard segmentation dataset.

Usage:
    python train.py

What it does:
  1. Loads the dataset and splits it into train / validation / test sets
  2. Builds the U-Net model and moves it to GPU/MPS (if available)
  3. Trains for NUM_EPOCHS using BCELoss and Adam optimiser
  4. Validates after every epoch (no gradient updates)
  5. Prints per-epoch train/val losses
  6. Saves the best model weights to `unet_keyboard.pth`
  7. Plots training & validation loss curves to `loss_curve.png`
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt

from dataset import load_coco_splits
from model import UNet

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = "."  # dataset root (contains the train/ folder)
IMG_SIZE = 256  # image & mask target size (square)
BATCH_SIZE = 8  # mini-batch size
NUM_EPOCHS = 20  # total training epochs
LEARNING_RATE = 1e-3  # Adam learning rate
NUM_WORKERS = 0  # DataLoader workers (0 = main process for macOS safety)
MODEL_SAVE = "unet_keyboard.pth"
PLOT_SAVE = "loss_curve.png"

# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # ── Device selection ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # pin_memory only works on CUDA, not MPS or CPU
    use_pin_memory = device.type == "cuda"

    print(f"\n{'='*55}", flush=True)
    print(f" Keyboard Segmentation — U-Net Training", flush=True)
    print(f"{'='*55}", flush=True)
    print(f" Device     : {device}", flush=True)
    print(f" Batch size : {BATCH_SIZE}", flush=True)
    print(f" Epochs     : {NUM_EPOCHS}", flush=True)
    print(f" LR         : {LEARNING_RATE}", flush=True)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …", flush=True)
    train_ds, val_ds, test_ds = load_coco_splits(
        dataset_root=DATASET_ROOT,
        img_size=IMG_SIZE,
    )

    # Sanity-check: verify masks are non-trivial
    sample_img, sample_mask = train_ds[0]
    fg_frac = sample_mask.mean().item()
    print(f"  Sample mask foreground fraction : {fg_frac:.3f}", flush=True)
    if fg_frac == 0.0:
        print(
            "  ⚠  Warning: sample mask is all background — check annotations!",
            flush=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    print("\n[2/4] Building model …", flush=True)
    model = UNet().to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}", flush=True)
    print(f"  Loss       : BCELoss", flush=True)
    print(f"  Optimiser  : Adam  (lr={LEARNING_RATE})", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[3/4] Training …", flush=True)
    print(f"{'─'*55}", flush=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Train phase ───────────────────────────────────────────────────
        model.train()
        running_train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)  # (B, 3, H, W)
            masks = masks.to(device)  # (B, 1, H, W)

            optimiser.zero_grad()  # clear previous gradients
            preds = model(images)  # forward pass → (B, 1, H, W) in [0,1]
            loss = criterion(preds, masks)
            loss.backward()  # backpropagate
            optimiser.step()  # update weights

            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_ds)

        # ── Validation phase ─────────────────────────────────────────────
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():  # disable gradient computation
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)
                loss = criterion(preds, masks)
                running_val_loss += loss.item() * images.size(0)

        val_loss = running_val_loss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ── Save best model ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            saved_marker = " ✓ saved"
        else:
            saved_marker = ""

        print(
            f"  Epoch [{epoch:>2}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}{saved_marker}",
            flush=True,
        )

    print(f"{'─'*55}", flush=True)
    print(f"  Best validation loss : {best_val_loss:.4f}", flush=True)
    print(f"  Model saved to       : {MODEL_SAVE}", flush=True)

    # ── Plot loss curves ───────────────────────────────────────────────────
    print("\n[4/4] Plotting loss curves …", flush=True)
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs_range, train_losses, label="Training Loss", color="#4C72B0", linewidth=2
    )
    plt.plot(
        epochs_range,
        val_losses,
        label="Validation Loss",
        color="#DD8452",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("BCE Loss", fontsize=13)
    plt.title("U-Net Training & Validation Loss — Keyboard Segmentation", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE, dpi=150)
    plt.close()
    print(f"  Loss curve saved to : {PLOT_SAVE}", flush=True)

    # ── Performance summary ────────────────────────────────────────────────
    final_train = train_losses[-1]
    final_val = val_losses[-1]

    print(f"\n{'='*55}")
    print(f" PERFORMANCE SUMMARY")
    print(f"{'='*55}")
    print(f"  Final Training Loss   : {final_train:.4f}")
    print(f"  Final Validation Loss : {final_val:.4f}")
    print(f"  Best Validation Loss  : {best_val_loss:.4f}")

    # Overfitting / underfitting heuristics
    gap = final_val - final_train
    if gap > 0.05:
        print(f"\n  ⚠  Validation loss > Training loss by {gap:.4f}")
        print(f"     → Possible OVERFITTING detected.")
        print(
            f"     → Consider adding dropout, more augmentation, or reducing model capacity."
        )
    elif final_train > 0.3:
        print(f"\n  ⚠  Training loss is still relatively high ({final_train:.4f}).")
        print(f"     → Possible UNDERFITTING. Try more epochs or a higher LR.")
    else:
        print(f"\n  ✓  Loss gap is small — model generalises well.")

    print(f"\n  Run  python evaluate.py  to get Dice & IoU on the test set.\n")


if __name__ == "__main__":
    main()
