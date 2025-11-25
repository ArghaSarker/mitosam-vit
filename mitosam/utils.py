
import os
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2
import albumentations as A


def save_figure_png(fig, name, out_dir="report/figures", overwrite=False, dpi=300, bbox_inches="tight"):
    """
    Save a matplotlib Figure as a PNG, auto-creating the directory.
    If overwrite=False and file exists, appends _{i} to filename.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = str(name)
    base, ext = os.path.splitext(name)
    if ext.lower() != ".png":
        name = base + ".png"

    out_path = out_dir / name

    if out_path.exists() and not overwrite:
        i = 1
        while True:
            candidate = out_dir / f"{base}_{i}.png"
            if not candidate.exists():
                out_path = candidate
                break
            i += 1

    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure to: {out_path}")
    return str(out_path)


def _get_starts(L, patch_size, step):
    """
    Valid patch start indices along a dimension, no padding.
    Ensures last patch touches border.
    """
    if L <= patch_size:
        return [0]
    starts = list(range(0, L - patch_size + 1, step))
    last_start = L - patch_size
    if last_start not in starts:
        starts.append(last_start)
    return sorted(starts)


def extract_overlapping_patches(image, mask, patch_size=256, overlap=32):
    """
    Extract overlapping 2D patches without padding.
    Works for image shape (H,W) or (H,W,C).
    """
    step = patch_size - overlap
    H, W = image.shape[:2]

    y_starts = _get_starts(H, patch_size, step)
    x_starts = _get_starts(W, patch_size, step)

    img_list, mask_list = [], []
    for y in y_starts:
        for x in x_starts:
            img_patch = image[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            img_list.append(img_patch)
            mask_list.append(mask_patch)

    return img_list, mask_list


def filter_patches(image_patches, mask_patches, min_coverage=0.005):
    """
    Keep patches where mask coverage >= min_coverage.
    """
    kept_imgs, kept_msks = [], []
    for img, msk in zip(image_patches, mask_patches):
        ratio = np.count_nonzero(msk) / msk.size
        if ratio >= min_coverage:
            kept_imgs.append(img)
            kept_msks.append(msk)
    return kept_imgs, kept_msks


def make_patch_dataset(images, masks, patch_size, overlap, min_coverage):
    """
    Patchify a stack of images/masks, then filter by min_coverage.
    Returns filtered (imgs, masks) as lists.
    """
    img_patches, mask_patches = [], []
    for img, msk in zip(images, masks):
        ip, mp = extract_overlapping_patches(
            img, msk, patch_size=patch_size, overlap=overlap
        )
        img_patches.extend(ip)
        mask_patches.extend(mp)

    print(f"Total patches before filtering: {len(img_patches)}")
    img_f, msk_f = filter_patches(img_patches, mask_patches, min_coverage=min_coverage)
    print(f"After filtering: {len(img_f)} kept")
    return img_f, msk_f


def build_augmenter(patch_size=256):
    """
    Albumentations v1.x compatible augmenter.
    Geometric transforms apply to both image & mask.
    Non-geometric apply to image only.
    """
    geometric = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.025,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_LINEAR,   # image interpolation
        ),
        A.ElasticTransform(
            alpha=1,
            sigma=20,
            p=0.5,
            interpolation=cv2.INTER_LINEAR,   # image interpolation
        ),
        # A.RandomCrop(height=patch_size, width=patch_size, p=0.5),
    ])

    non_geometric = A.Compose([
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # odd limits only in v1.x
        A.RandomBrightnessContrast(p=0.2),
    ])

    return geometric, non_geometric


def augment_data(image, mask, geometric_tf, non_geometric_tf, n_aug=3):
    """
    Returns [(orig_img, orig_mask), (aug1_img, aug1_mask), ...].
    """
    out = [(image, mask)]
    for _ in range(n_aug):
        geo = geometric_tf(image=image, mask=mask)
        geo_img, geo_msk = geo["image"], geo["mask"]
        nong_img = non_geometric_tf(image=geo_img)["image"]
        out.append((nong_img, geo_msk))
    return out


def percentile_normalize_batch(images, pmin=1.0, pmax=99.0, out_range=(0.0, 1.0), eps=1e-8):
    """
    Per-image (per-patch) percentile normalization.
    images: list/array of (H,W) or (H,W,C).
    """
    out_imgs = []
    out_min, out_max = out_range

    for img in images:
        x = np.asarray(img, np.float32)

        lo = np.percentile(x, pmin)
        hi = np.percentile(x, pmax)

        if hi - lo < eps:
            lo, hi = float(x.min()), float(x.max())

        denom = max(hi - lo, eps)
        x = np.clip(x, lo, hi)
        x = (x - lo) / denom
        x = x * (out_max - out_min) + out_min

        out_imgs.append(x.astype(np.float32))

    return out_imgs





import numpy as np
import matplotlib.pyplot as plt

def plot_random_image_mask_pairs(
    images,
    masks,
    num_samples: int = 5,
    seed: int | None = None,
    cmap: str = "gray",
    figsize: tuple[int, int] = (10, 4)
):
    """
    Randomly plots image–mask pairs from a dataset.

    Args:
        images: Array-like of shape (N, H, W) or (N, H, W, C)
            Dataset of images.
        masks: Array-like of shape (N, H, W)
            Dataset of corresponding masks.
        num_samples: Number of random pairs to plot (default: 5).
        seed: Optional random seed for reproducibility.
        cmap: Colormap for imshow (default: 'gray').
        figsize: Size of the entire figure.

    Returns:
        fig, axes: Matplotlib Figure and Axes objects.
    """
    images = np.asarray(images)
    masks = np.asarray(masks)

    assert len(images) == len(masks), (
        f"images and masks must have the same length, "
        f"got {len(images)} and {len(masks)}"
    )

    n = len(images)
    if n == 0:
        raise ValueError("Empty dataset: 'images' has length 0.")

    # Ensure we don't request more samples than available
    num_samples = min(num_samples, n)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_samples, replace=False)

    fig, axes = plt.subplots(2, num_samples, figsize=figsize)

    if num_samples == 1:
        # When num_samples=1, axes are not a 2D array, so normalize
        axes = np.array([[axes[0]], [axes[1]]])

    for col, idx in enumerate(indices):
        img = images[idx]
        msk = masks[idx]

        # First row: images
        ax_img = axes[0, col]
        ax_img.imshow(img, cmap=cmap)
        ax_img.set_title(f"Image {idx}")
        ax_img.axis("off")

        # Second row: masks
        ax_msk = axes[1, col]
        ax_msk.imshow(msk, cmap=cmap)
        ax_msk.set_title(f"Mask {idx}")
        ax_msk.axis("off")

    plt.tight_layout()
    return fig, axes

