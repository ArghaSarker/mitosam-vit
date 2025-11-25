# datagen.py
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from sklearn.model_selection import train_test_split

from utils import (
    save_figure_png,
    make_patch_dataset,
    build_augmenter,
    augment_data,
    percentile_normalize_batch,
)

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_TIF = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/raw/Dataset/training.tif"
MASK_TIF  = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/raw/Dataset/training_groundtruth.tif"

OUT_DIR = Path("/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256
TRAIN_OVERLAP = 64
VAL_OVERLAP = 32
MIN_COVERAGE = 0.009

N_DEBUG = None
SHOW_SANITY_PLOTS = True

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
large_images = tifffile.imread(TRAIN_TIF)
large_masks  = tifffile.imread(MASK_TIF)

if N_DEBUG:
    large_images = large_images[:N_DEBUG]
    large_masks  = large_masks[:N_DEBUG]

assert large_images.shape == large_masks.shape, "Mismatch in image and mask stacks"

# -------------------------------------------------------------------------
# Train/val split
# -------------------------------------------------------------------------
ids = list(range(len(large_images)))
train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=SEED, shuffle=True)
train_images = large_images[train_ids]
train_masks  = large_masks[train_ids]
val_images   = large_images[val_ids]
val_masks    = large_masks[val_ids]

# -------------------------------------------------------------------------
# Extract patches
# -------------------------------------------------------------------------
train_filtered_imgs, train_filtered_msks = make_patch_dataset(
    train_images, train_masks,
    patch_size=PATCH_SIZE, overlap=TRAIN_OVERLAP, min_coverage=MIN_COVERAGE
)

val_filtered_imgs, val_filtered_msks = make_patch_dataset(
    val_images, val_masks,
    patch_size=PATCH_SIZE, overlap=VAL_OVERLAP, min_coverage=MIN_COVERAGE
)

# -------------------------------------------------------------------------
# Visualize random samples
# -------------------------------------------------------------------------
if SHOW_SANITY_PLOTS:
    n_cols = min(3, max(len(train_filtered_imgs), len(val_filtered_imgs)))
    train_idxs = random.sample(range(len(train_filtered_imgs)), n_cols) if train_filtered_imgs else []
    val_idxs = random.sample(range(len(val_filtered_imgs)), n_cols) if val_filtered_imgs else []

    fig, axes = plt.subplots(4, n_cols, figsize=(2 * n_cols, 8))
    for i in range(n_cols):
        t_idx = train_idxs[i] if i < len(train_idxs) else None
        v_idx = val_idxs[i] if i < len(val_idxs) else None

        for r, (src, title) in enumerate([
            (train_filtered_imgs, "Train Image"),
            (train_filtered_msks, "Train Mask"),
            (val_filtered_imgs, "Val Image"),
            (val_filtered_msks, "Val Mask"),
        ]):
            if (idx := (t_idx if r < 2 else v_idx)) is not None:
                axes[r, i].imshow(src[idx], cmap="gray")
                axes[r, i].set_title(title)
            axes[r, i].axis("off")

    plt.tight_layout()
    save_figure_png(fig, REPORT_DIR / "sanity_patch_grid.png")
    plt.close(fig)

# -------------------------------------------------------------------------
# Build augmenter
# -------------------------------------------------------------------------
geometric_tf, non_geometric_tf = build_augmenter(PATCH_SIZE)

# -------------------------------------------------------------------------
# Augmentation visualization
# -------------------------------------------------------------------------
if SHOW_SANITY_PLOTS and train_filtered_imgs:
    parent_idx = 0
    aug_pairs = augment_data(
        train_filtered_imgs[parent_idx],
        train_filtered_msks[parent_idx],
        geometric_tf,
        non_geometric_tf,
        n_aug=3
    )

    fig, axes = plt.subplots(2, len(aug_pairs), figsize=(4 * len(aug_pairs), 6))
    for i, (img, msk) in enumerate(aug_pairs):
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"Aug Image {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(msk, cmap="gray")
        axes[1, i].set_title(f"Aug Mask {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    save_figure_png(fig, REPORT_DIR / f"augmentation_sanity_parent_{parent_idx}.png")
    plt.close(fig)

# -------------------------------------------------------------------------
# Apply augmentation
# -------------------------------------------------------------------------
train_aug_imgs, train_aug_msks = [], []
for img, msk in zip(train_filtered_imgs, train_filtered_msks):
    pairs = augment_data(img, msk, geometric_tf, non_geometric_tf, n_aug=3)
    for i, (aug_img, aug_msk) in enumerate(pairs):
        train_aug_imgs.append(aug_img)
        train_aug_msks.append(aug_msk)

# -------------------------------------------------------------------------
# Normalize
# -------------------------------------------------------------------------
train_imgs_norm = percentile_normalize_batch(train_aug_imgs, pmin=1, pmax=99)
val_imgs_norm   = percentile_normalize_batch(val_filtered_imgs, pmin=1, pmax=99)

train_imgs_np = np.asarray(train_imgs_norm, dtype=np.float32)
val_imgs_np   = np.asarray(val_imgs_norm, dtype=np.float32)
train_msks_np = np.asarray(train_aug_msks, dtype=np.uint8)
val_msks_np   = np.asarray(val_filtered_msks, dtype=np.uint8)

# -------------------------------------------------------------------------
# Save processed data
# -------------------------------------------------------------------------
np.savez_compressed(OUT_DIR / "train_data_processed.npz", images=train_imgs_np, masks=train_msks_np)
np.savez_compressed(OUT_DIR / "val_data_processed.npz", images=val_imgs_np, masks=val_msks_np)

print(f"✅ Saved training set: {train_imgs_np.shape} | {train_msks_np.shape}")
print(f"✅ Saved validation set: {val_imgs_np.shape} | {val_msks_np.shape}")