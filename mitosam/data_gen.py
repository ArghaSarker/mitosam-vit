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

REPORT_DIR = Path("/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/reports/processed_data")
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



print(f'train_images shape: {train_images.shape} , min: {train_images.min()} , max: {train_images.max()} , dtype: {train_images.dtype}')
print(f'train_masks shape: {train_masks.shape} , min: {train_masks.min()} , max: {train_masks.max()} , dtype: {train_masks.dtype}')
print(f'val_images shape: {val_images.shape} , min: {val_images.min()} , max: {val_images.max()} , dtype: {val_images.dtype}')
print(f'val_masks shape: {val_masks.shape} , min: {val_masks.min()} , max: {val_masks.max()} , dtype: {val_masks.dtype}')





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


print(f'train_filtered_imgs shape: {len(train_filtered_imgs)} dtype: {train_filtered_imgs[0].dtype} , min: {train_filtered_imgs[0].min()} , max: {train_filtered_imgs[0].max()}')
print(f'train_filtered_msks shape: {len(train_filtered_msks)} dtype: {train_filtered_msks[0].dtype} , min: {train_filtered_msks[0].min()} , max: {train_filtered_msks[0].max()}')
print(f'val_filtered_imgs shape: {len(val_filtered_imgs)} dtype: {val_filtered_imgs[0].dtype} , min: {val_filtered_imgs[0].min()} , max: {val_filtered_imgs[0].max()}')
print(f'val_filtered_msks shape: {len(val_filtered_msks)} dtype: {val_filtered_msks[0].dtype} , min: {val_filtered_msks[0].min()} , max: {val_filtered_msks[0].max()}')

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



print(f'befoore normalization train_aug_imgs shape: {len(train_aug_imgs)} dtype: {train_aug_imgs[0].dtype} , min: {train_aug_imgs[0].min()} , max: {train_aug_imgs[0].max()}'  )
print(f'befoore normalization train_aug_msks shape: {len(train_aug_msks)} dtype: {train_aug_msks[0].dtype} , min: {train_aug_msks[0].min()} , max: {train_aug_msks[0].max()}'  )


print(f'befoore normalization val_filtered_imgs shape: {len(val_filtered_imgs)} dtype: {val_filtered_imgs[0].dtype} , min: {val_filtered_imgs[0].min()} , max: {val_filtered_imgs[0].max()}'  )
print(f'befoore normalization val_filtered_msks shape: {len(val_filtered_msks)} dtype: {val_filtered_msks[0].dtype} , min: {val_filtered_msks[0].min()} , max: {val_filtered_msks[0].max()}'  )


# SAM is tarined on images that is RGB and unit8. so values are in between 0 and 255. the mask must be 0 and 1 for Dice Focal Loss calculation. 


# should not normalize iamge and mask
# train_imgs_norm = percentile_normalize_batch(train_aug_imgs, pmin=1, pmax=99)
# val_imgs_norm   = percentile_normalize_batch(val_filtered_imgs, pmin=1, pmax=99)


train_imgs_norm = train_aug_imgs
val_imgs_norm   = val_filtered_imgs

# lets make the mask 0 and 1 as type unit8
train_aug_msks = [(msk > 0).astype(np.uint8) for msk in train_aug_msks]
val_filtered_msks = [(msk > 0).astype(np.uint8) for msk in val_filtered_msks]



# print(f'after normalization train_imgs_norm shape: {len(train_imgs_norm)} dtype: {train_imgs_norm[0].dtype} , min: {train_imgs_norm[0].min()} , max: {train_imgs_norm[0].max()}'  )
print(f'after normalization train_aug_msks shape: {len(train_aug_msks)} dtype: {train_aug_msks[0].dtype} , min: {train_aug_msks[0].min()} , max: {train_aug_msks[0].max()}'  )

# print(f'after normalization val_imgs_norm shape: {len(val_imgs_norm)} dtype: {val_imgs_norm[0].dtype} , min: {val_imgs_norm[0].min()} , max: {val_imgs_norm[0].max()}'  )
print(f'after normalization val_filtered_msks shape: {len(val_filtered_msks)} dtype: {val_filtered_msks[0].dtype} , min: {val_filtered_msks[0].min()} , max: {val_filtered_msks[0].max()}'  )



# -------------------------------------------------------------------------
train_imgs_np = np.asarray(train_imgs_norm, dtype=np.uint8)
val_imgs_np   = np.asarray(val_imgs_norm, dtype=np.uint8)
train_msks_np = np.asarray(train_aug_msks, dtype=np.uint8)
val_msks_np   = np.asarray(val_filtered_msks, dtype=np.uint8)


print('print dataset summary of values and adt types')  
print(f'train_imgs_np shape: {train_imgs_np.shape} dtype: {train_imgs_np.dtype} , min: {train_imgs_np.min()} , max: {train_imgs_np.max()}'  )
print(f'val_imgs_np shape: {val_imgs_np.shape} dtype: {val_imgs_np.dtype} , min: {val_imgs_np.min()} , max: {val_imgs_np.max()}'  )
print(f'train_msks_np shape: {train_msks_np.shape} dtype: {train_msks_np.dtype} , min: {train_msks_np.min()} , max: {train_msks_np.max()}'  )
print(f'val_msks_np shape: {val_msks_np.shape} dtype: {val_msks_np.dtype} , min:    {val_msks_np.min()} , max: {val_msks_np.max()}'  )




# -------------------------------------------------------------------------
# Save processed data
# -------------------------------------------------------------------------
np.savez_compressed(OUT_DIR / "train_data_processed.npz", images=train_imgs_np, masks=train_msks_np)
np.savez_compressed(OUT_DIR / "val_data_processed.npz", images=val_imgs_np, masks=val_msks_np)

print(f"✅ Saved training set: {train_imgs_np.shape} | {train_msks_np.shape}")
print(f"✅ Saved validation set: {val_imgs_np.shape} | {val_msks_np.shape}")


## outputs ## 

# print dataset summary of values and adt types
# train_imgs_np shape: (608, 256, 256) dtype: uint8 , min: 0 , max: 255
# val_imgs_np shape: (33, 256, 256) dtype: uint8 , min: 48 , max: 222
# train_msks_np shape: (608, 256, 256) dtype: uint8 , min: 0 , max: 1
# val_msks_np shape: (33, 256, 256) dtype: uint8 , min:    0 , max: 1
# ✅ Saved training set: (608, 256, 256) | (608, 256, 256)
# ✅ Saved validation set: (33, 256, 256) | (33, 256, 256)