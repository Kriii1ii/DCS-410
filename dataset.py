import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class KeyboardSegDataset(Dataset):
    """
    Binary segmentation dataset for keyboard images.

    Args:
        image_paths (list[str]): Absolute paths to image files.
        coco_data  (dict):       Parsed COCO JSON dict (the full annotation dict).
        img_size   (int):        Target spatial resolution (square). Default: 256.
        augment    (bool):       If True, apply simple horizontal-flip augmentation.
    """

    def __init__(self, image_paths, coco_data, img_size=256, augment=False):
        self.image_paths = image_paths
        self.img_size = img_size
        self.augment = augment

        # ── Build path→id and id→annotations index from COCO JSON ──────────
        self.id_to_anns = {}  # image_id  → list of annotation dicts
        for ann in coco_data.get("annotations", []):
            iid = ann["image_id"]
            self.id_to_anns.setdefault(iid, []).append(ann)

        self.fname_to_info = {}  # file_name → image info dict
        for img_info in coco_data.get("images", []):
            self.fname_to_info[img_info["file_name"]] = img_info

        # Pre-resolve image info for every path we own
        self.entries = []
        for p in image_paths:
            fname = os.path.basename(p)
            info = self.fname_to_info.get(fname)
            if info is not None:
                self.entries.append((p, info))
            # If no annotation entry exists, skip the file silently

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, img_info = self.entries[idx]

        # ── Load image ────────────────────────────────────────────────────
        image = cv2.imread(img_path)
        if image is None:
            # Fall back to a blank image if the file is unreadable
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h = img_info.get("height", image.shape[0])
        orig_w = img_info.get("width", image.shape[1])

        # ── Build binary mask from COCO annotations ──────────────────────
        # Strategy:
        #   • If the annotation has polygon segmentation → use polygon fill
        #   • Otherwise → use the bounding-box (x, y, w, h) as a rectangular mask
        #   This handles Roboflow COCO exports where segmentation = []
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        annotations = self.id_to_anns.get(img_info["id"], [])

        for ann in annotations:
            seg = ann.get("segmentation", [])

            if isinstance(seg, list) and len(seg) > 0:
                # Polygon format: list of [x1,y1,x2,y2,…] flattened lists
                for polygon in seg:
                    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], color=1)

            elif isinstance(seg, dict):
                # RLE format – skip (uncommon in Roboflow exports)
                pass

            else:
                # ── Fallback: use bounding box to create a rectangular mask ──
                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    x, y, bw, bh = bbox
                    x, y = int(float(x)), int(float(y))
                    bw, bh = int(float(bw)), int(float(bh))
                    # Clamp to image boundaries
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(orig_w, x + bw)
                    y2 = min(orig_h, y + bh)
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = 1

        # ── Resize to (img_size × img_size) ─────────────────────────────
        image = cv2.resize(
            image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )

        # ── Optional augmentation ─────────────────────────────────────────
        if self.augment and np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # ── Convert to tensors ────────────────────────────────────────────
        # Image: HWC uint8  → CHW float32 in [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Mask: HW uint8 binary → (1,H,W) float32 in {0.0, 1.0}
        mask_binary = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return image_tensor, mask_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load and split dataset
# ──────────────────────────────────────────────────────────────────────────────


def load_coco_splits(
    dataset_root: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    img_size: int = 256,
    seed: int = 42,
):
    """
    Loads the Roboflow COCO dataset from `dataset_root/train/` and
    automatically splits it into train / val / test subsets.

    Returns:
        train_ds, val_ds, test_ds – KeyboardSegDataset instances
    """
    split_dir = os.path.join(dataset_root, "train")
    ann_file = os.path.join(split_dir, "_annotations.coco.json")

    if not os.path.exists(ann_file):
        raise FileNotFoundError(
            f"Annotation file not found: {ann_file}\n"
            f"Make sure the dataset is in COCO format and placed at: {dataset_root}"
        )

    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Build list of image paths that actually exist on disk
    all_paths = []
    for img_info in coco_data["images"]:
        p = os.path.join(split_dir, img_info["file_name"])
        if os.path.exists(p):
            all_paths.append(p)

    # Shuffle deterministically
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_paths)).tolist()

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_paths = [all_paths[i] for i in indices[:n_train]]
    val_paths = [all_paths[i] for i in indices[n_train : n_train + n_val]]
    test_paths = [all_paths[i] for i in indices[n_train + n_val :]]

    print(f"Dataset split (seed={seed}):")
    print(f"  Total images  : {n}")
    print(f"  Train         : {len(train_paths)}")
    print(f"  Validation    : {len(val_paths)}")
    print(f"  Test          : {len(test_paths)}")

    train_ds = KeyboardSegDataset(
        train_paths, coco_data, img_size=img_size, augment=True
    )
    val_ds = KeyboardSegDataset(val_paths, coco_data, img_size=img_size, augment=False)
    test_ds = KeyboardSegDataset(
        test_paths, coco_data, img_size=img_size, augment=False
    )

    return train_ds, val_ds, test_ds
