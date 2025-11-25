import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import Any, Mapping, Optional, Tuple


def visualize_sam_sample(
    dataset_or_sample: Mapping[str, Any] | Any,
    idx: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 4),
    print_info: bool = True,
):
    """
    Visualize one processed sample from SAMDataset (or a single sample dict).

    Panels:
      1) SAM input image (resized, normalized by SamProcessor)
      2) Ground-truth mask (resized to SAM image size)
      3) Overlay of mask + SAM bounding box

    Titles include:
      - Image size (H, W) at SAM resolution
      - Bbox coordinates (x_min, y_min, x_max, y_max)
      - Original & reshaped sizes if available in the sample

    Args:
        dataset_or_sample:
            - A dataset-like object (supports __getitem__), OR
            - A single sample dict with keys:
                'pixel_values', 'ground_truth_mask', 'input_boxes'
              (and optionally 'original_sizes', 'reshaped_input_sizes')
        idx:
            - If not None, we use dataset_or_sample[idx] as the sample.
            - If None, dataset_or_sample is assumed to be the sample itself.
        figsize:
            - Matplotlib figure size.
        print_info:
            - Whether to print shapes and bbox info in the console.

    Returns:
        fig, axes: Matplotlib Figure and Axes array (1x3).
    """
    # ---------- 1. Get sample dict ----------
    if idx is not None:
        sample = dataset_or_sample[idx]
    else:
        sample = dataset_or_sample

    pixel_values = sample["pixel_values"]
    mask = sample["ground_truth_mask"]
    bbox = sample["input_boxes"]

    # Optional extra info from SamProcessor (if present)
    orig_size = sample.get("original_sizes", None)
    reshaped_size = sample.get("reshaped_input_sizes", None)

    # ---------- 2. Convert pixel_values -> numpy RGB in [0,1] ----------
    if isinstance(pixel_values, torch.Tensor):
        img = pixel_values.detach().cpu()
        # (C,H,W) -> (H,W,C) if needed
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = img.numpy()
    else:
        img = np.array(pixel_values)

    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    # Ensure 3 channels for overlay
    if img.ndim == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_rgb = np.repeat(img, 3, axis=-1)
    else:
        img_rgb = img

    H_sam, W_sam = img_rgb.shape[:2]  # SAM-resolution size

    # ---------- 3. Convert mask -> 2D uint8, resize to (H_sam, W_sam) ----------
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.array(mask)

    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)

    mask_np = mask_np.astype(np.uint8)

    if mask_np.shape != (H_sam, W_sam):
        mask_resized = np.array(
            Image.fromarray(mask_np).resize((W_sam, H_sam), Image.NEAREST)
        )
    else:
        mask_resized = mask_np

    # ---------- 4. Convert bbox -> numpy (4,) ----------
    if isinstance(bbox, torch.Tensor):
        bbox_np = bbox.detach().cpu().numpy()
    else:
        bbox_np = np.array(bbox)

    # Handle shapes like (1,4) or (1,1,4)
    bbox_np = np.array(bbox_np).reshape(-1, 4)[0]
    x_min, y_min, x_max, y_max = bbox_np

    # ---------- 5. Build overlay (mask in red) ----------
    overlay = img_rgb.copy()
    overlay[mask_resized > 0] = [1.0, 0.0, 0.0]

    # ---------- 6. Plot ----------
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Build optional text for original / reshaped sizes
    size_info = ""
    if orig_size is not None:
        # orig_size is typically a tensor [H_orig, W_orig]
        if isinstance(orig_size, torch.Tensor):
            orig = orig_size.detach().cpu().numpy().tolist()
        else:
            orig = np.array(orig_size).tolist()
        size_info += f"orig={orig}"
    if reshaped_size is not None:
        if size_info:
            size_info += ", "
        if isinstance(reshaped_size, torch.Tensor):
            resh = reshaped_size.detach().cpu().numpy().tolist()
        else:
            resh = np.array(reshaped_size).tolist()
        size_info += f"reshaped={resh}"

    # Panel 1: SAM input
    title_img = f"SAM Input Image (H={H_sam}, W={W_sam})"
    if size_info:
        title_img += f"\n{size_info}"
    axes[0].imshow(img_rgb)
    axes[0].set_title(title_img, fontsize=9)
    axes[0].axis("off")

    # Panel 2: resized mask
    title_mask = f"Mask (resized to H={H_sam}, W={W_sam})"
    axes[1].imshow(mask_resized, cmap="gray")
    axes[1].set_title(title_mask, fontsize=9)
    axes[1].axis("off")

    # Panel 3: overlay + bbox
    title_overlay = (
        f"Overlay + Box\n"
        f"bbox=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]"
    )
    axes[2].imshow(overlay)
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none",
    )
    axes[2].add_patch(rect)
    axes[2].set_title(title_overlay, fontsize=9)
    axes[2].axis("off")

    plt.tight_layout()

    # ---------- 7. Optional console logging ----------
    if print_info:
        idx_str = f"{idx}" if idx is not None else "<direct sample>"
        print(f"Sample #{idx_str}")
        print(f"  SAM image size:   (H={H_sam}, W={W_sam})")
        if orig_size is not None:
            print(f"  original_sizes:   {sample['original_sizes']}")
        if reshaped_size is not None:
            print(f"  reshaped_sizes:   {sample['reshaped_input_sizes']}")
        print(f"  pixel_values shape: {sample['pixel_values'].shape}")
        print(f"  input_boxes (first): {bbox_np}")
        print(f"  mask (resized) shape: {mask_resized.shape}, "
              f"unique values: {np.unique(mask_resized)}")

    return fig, axes
