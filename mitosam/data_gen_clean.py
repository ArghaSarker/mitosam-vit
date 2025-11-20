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

OUT_DIR = Path("/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/processed/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256
TRAIN_OVERLAP = 64
VAL_OVERLAP = 32
MIN_COVERAGE = 0.009

N_DEBUG = 40  # kept as you requested
SHOW_SANITY_PLOTS = True


def main():
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    large_images = tifffile.imread(TRAIN_TIF)
    large_masks  = tifffile.imread(MASK_TIF)

    # Debug slice (kept)
    large_images = large_images[:N_DEBUG]
    large_masks  = large_masks[:N_DEBUG]

    print(
        "Loaded:",
        large_images.shape, large_images.dtype,
        "min/max:", large_images.min(), large_images.max()
    )
    assert large_images.shape == large_masks.shape, "Image/mask stack mismatch!"

    # -------------------------------------------------------------------------
    # Train/val split at image level
    # -------------------------------------------------------------------------
    image_ids = list(range(len(large_images)))
    train_ids, val_ids = train_test_split(
        image_ids, test_size=0.2, random_state=SEED, shuffle=True
    )

    train_images = large_images[train_ids]
    train_masks  = large_masks[train_ids]
    val_images   = large_images[val_ids]
    val_masks    = large_masks[val_ids]

    print(f"Train images: {train_images.shape}  Val images: {val_images.shape}")

    # -------------------------------------------------------------------------
    # Patch extraction + filtering
    # -------------------------------------------------------------------------
    train_filtered_imgs, train_filtered_msks = make_patch_dataset(
        train_images, train_masks,
        patch_size=PATCH_SIZE, overlap=TRAIN_OVERLAP,
        min_coverage=MIN_COVERAGE
    )

    val_filtered_imgs, val_filtered_msks = make_patch_dataset(
        val_images, val_masks,
        patch_size=PATCH_SIZE, overlap=VAL_OVERLAP,
        min_coverage=MIN_COVERAGE
    )

    # Optional sanity plot: random train/val patches
    if SHOW_SANITY_PLOTS and (len(train_filtered_imgs) > 0 or len(val_filtered_imgs) > 0):
        n_show = 3
        n_cols = max(1, min(n_show, max(len(train_filtered_imgs), len(val_filtered_imgs))))

        def safe_sample(n, k):
            if n == 0:
                return [None] * k
            if n >= k:
                return random.sample(range(n), k)
            return [random.choice(range(n)) for _ in range(k)]

        train_idxs = safe_sample(len(train_filtered_imgs), n_cols)
        val_idxs   = safe_sample(len(val_filtered_imgs), n_cols)

        fig, axes = plt.subplots(4, n_cols, figsize=(2 * n_cols, 8))
        if n_cols == 1:
            axes = axes.reshape(4, 1)

        for c in range(n_cols):
            t_idx, v_idx = train_idxs[c], val_idxs[c]

            if t_idx is not None:
                axes[0, c].imshow(train_filtered_imgs[t_idx], cmap="gray")
                axes[1, c].imshow(train_filtered_msks[t_idx], cmap="gray")
                axes[0, c].set_title(f"Train img {t_idx}")
                axes[1, c].set_title("Train mask")
            else:
                axes[0, c].axis("off")
                axes[1, c].axis("off")

            if v_idx is not None:
                axes[2, c].imshow(val_filtered_imgs[v_idx], cmap="gray")
                axes[3, c].imshow(val_filtered_msks[v_idx], cmap="gray")
                axes[2, c].set_title(f"Val img {v_idx}")
                axes[3, c].set_title("Val mask")
            else:
                axes[2, c].axis("off")
                axes[3, c].axis("off")

            for r in range(4):
                axes[r, c].axis("off")

        fig.suptitle("Sample Train/Val Patches")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # Build augmenter
    # -------------------------------------------------------------------------
    geometric_tf, non_geometric_tf = build_augmenter(PATCH_SIZE)

    # -------------------------------------------------------------------------
    # Augmentation sanity check: show one original + its augmentations
    # -------------------------------------------------------------------------
    if SHOW_SANITY_PLOTS and len(train_filtered_imgs) > 0:
        parent_idx = 0  # change to random.randint(...) if you want random parent
        parent_img = train_filtered_imgs[parent_idx]
        parent_msk = train_filtered_msks[parent_idx]

        sanity_pairs = augment_data(
            parent_img, parent_msk,
            geometric_tf, non_geometric_tf,
            n_aug=3
        )

        n_cols = len(sanity_pairs)
        fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        for i, (img, msk) in enumerate(sanity_pairs):
            title_prefix = "Original" if i == 0 else f"Aug {i}"

            axes[0, i].imshow(img, cmap="gray")
            axes[0, i].set_title(f"{title_prefix} Image")
            axes[0, i].axis("off")

            axes[1, i].imshow(msk, cmap="gray")
            axes[1, i].set_title(f"{title_prefix} Mask")
            axes[1, i].axis("off")

        fig.suptitle(f"Augmentation sanity check (parent patch #{parent_idx})")
        plt.tight_layout()
        save_figure_png(fig, f"augmentation_sanity_parent_{parent_idx}.png")
        plt.show()

    # -------------------------------------------------------------------------
    # Augment training set (train only)
    # -------------------------------------------------------------------------
    train_augmented_imgs, train_augmented_msks = [], []
    for img, msk in zip(train_filtered_imgs, train_filtered_msks):
        pairs = augment_data(img, msk, geometric_tf, non_geometric_tf, n_aug=3)
        for a_img, a_msk in pairs:
            train_augmented_imgs.append(a_img)
            train_augmented_msks.append(a_msk)

    print(f"Original train patches: {len(train_filtered_imgs)}")
    print(f"Augmented train samples: {len(train_augmented_imgs)}")

    # -------------------------------------------------------------------------
    # Percentile normalization (images only)
    # -------------------------------------------------------------------------
    train_data_norm = percentile_normalize_batch(train_augmented_imgs, pmin=1, pmax=99)
    val_data_norm   = percentile_normalize_batch(val_filtered_imgs, pmin=1, pmax=99)

    # Masks unchanged
    train_masks_final = train_augmented_msks
    val_masks_final   = val_filtered_msks

    print("Normalization done.")
    print(
        f"Train norm dtype: {train_data_norm[0].dtype}, "
        f"range: {train_data_norm[0].min():.3f}-{train_data_norm[0].max():.3f}"
    )
    print(
        f"Val   norm dtype: {val_data_norm[0].dtype}, "
        f"range: {val_data_norm[0].min():.3f}-{val_data_norm[0].max():.3f}"
    )

    # -------------------------------------------------------------------------
    # Save processed data
    # -------------------------------------------------------------------------
    train_images_to_save = np.asarray(train_data_norm, dtype=np.float32)
    train_masks_to_save  = np.asarray(train_masks_final)

    val_images_to_save = np.asarray(val_data_norm, dtype=np.float32)
    val_masks_to_save  = np.asarray(val_masks_final)

    train_out = OUT_DIR / "train_data_processed.npz"
    val_out   = OUT_DIR / "val_data_processed.npz"

    np.savez_compressed(train_out, images=train_images_to_save, masks=train_masks_to_save)
    np.savez_compressed(val_out, images=val_images_to_save, masks=val_masks_to_save)

    print(f"Saved train: {train_out} -> images {train_images_to_save.shape}, masks {train_masks_to_save.shape}")
    print(f"Saved val:   {val_out} -> images {val_images_to_save.shape}, masks {val_masks_to_save.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
