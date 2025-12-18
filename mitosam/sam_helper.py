import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Any, Tuple


def _add_grid_with_ticks(
    ax: plt.Axes,
    height: int,
    width: int,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Add a coarse grid and integer ticks to an image axis."""
    step_x = max(width // 10, 1)
    step_y = max(height // 10, 1)

    ax.set_xticks(np.arange(0, width, step_x))
    ax.set_yticks(np.arange(0, height, step_y))
    ax.tick_params(axis="both", labelsize=6, pad=1)

    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")
        label.set_horizontalalignment("right")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=6)

    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def visualize_sam_sample(sam_dataset: Any, idx: int) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize a SAMDataset sample: image, mask, and processed SAM input."""
    entry = sam_dataset.samples[idx]
    image_pil = entry["image"]
    mask_np = np.array(entry["mask"])
    bbox = entry["bbox"]  # [x_min, y_min, x_max, y_max]

    processor = sam_dataset.processor

    inputs = processor(
        image_pil,
        input_boxes=[[bbox]],
        return_tensors="pt",
    )

    image_np = np.array(image_pil)

    print("ORIGINAL IMAGE:")
    print("  shape:", image_np.shape)
    print("  dtype:", image_np.dtype)
    print("  min/max:", image_np.min(), image_np.max())

    print("\nORIGINAL MASK:")
    print("  shape:", mask_np.shape)
    print("  dtype:", mask_np.dtype)
    print("  min/max:", mask_np.min(), mask_np.max())

    print("\nBBOX (original coords):", bbox)

    print("\n=== Processor outputs ===")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v_min = v.min().item() if v.numel() else "n/a"
            v_max = v.max().item() if v.numel() else "n/a"
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, min={v_min}, max={v_max}")
        else:
            print(f"{k}: type={type(v)} -> {v}")

    bbox_proc = inputs["input_boxes"][0, 0].detach().cpu().numpy()
    print("\nBBOX (processor/model coords):", bbox_proc)

    h0, w0 = image_np.shape[:2]
    H_in, W_in = inputs["reshaped_input_sizes"][0].tolist()
    scale_x = W_in / w0
    scale_y = H_in / h0
    bbox_scaled_manual = np.array(
        [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y,
        ]
    )
    print("BBOX manually scaled:", bbox_scaled_manual)

    pv_tensor = inputs["pixel_values"]
    pv_min = pv_tensor.min().item()
    pv_max = pv_tensor.max().item()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Original image + bbox
    ax0 = axes[0]
    ax0.imshow(image_np, cmap="gray")
    x_min, y_min, x_max, y_max = bbox
    rect0 = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
    ax0.add_patch(rect0)
    ax0.set_title(
        "Original image + bbox\n"
        f"shape={image_np.shape}, dtype={image_np.dtype}\n"
        f"min={image_np.min():.3g}, max={image_np.max():.3g}",
        fontsize=7,
    )
    h, w = image_np.shape[:2]
    _add_grid_with_ticks(ax0, h, w, xlabel="x", ylabel="y")

    # Mask + bbox
    ax1 = axes[1]
    ax1.imshow(mask_np, cmap="gray")
    rect1 = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
    ax1.add_patch(rect1)
    ax1.set_title(
        "Mask + bbox\n"
        f"bbox={bbox}\n"
        f"shape={mask_np.shape}, dtype={mask_np.dtype}, "
        f"min={mask_np.min():.3g}, max={mask_np.max():.3g}",
        fontsize=7,
    )
    mh, mw = mask_np.shape[:2]
    _add_grid_with_ticks(ax1, mh, mw, xlabel="x", ylabel="y")

    # SAM input (pixel_values) + processed bbox
    ax2 = axes[2]
    pv = pv_tensor[0].detach().cpu()         # (3, H, W)
    pv_np = pv.permute(1, 2, 0).numpy()      # (H, W, 3)
    pv_np_vis = (pv_np - pv_np.min()) / (pv_np.max() - pv_np.min() + 1e-8)

    ax2.imshow(pv_np_vis)
    x_min_p, y_min_p, x_max_p, y_max_p = bbox_proc
    rect2 = plt.Rectangle(
        (x_min_p, y_min_p),
        x_max_p - x_min_p,
        y_max_p - y_min_p,
        edgecolor="lime",
        facecolor="none",
        linewidth=2,
    )
    ax2.add_patch(rect2)
    ax2.set_title(
        "Model input + processed bbox\n"
        f"input size=({H_in}, {W_in}), bbox={np.round(bbox_proc, 1)}\n"
        f"pixel_values min={pv_min:.3g}, max={pv_max:.3g}",
        fontsize=7,
    )
    _add_grid_with_ticks(ax2, H_in, W_in, xlabel="x", ylabel="y")

    plt.tight_layout()
    return fig, axes


def compute_batch_dice_iou(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Compute mean Dice and IoU over a batch of predicted masks."""
    if pred_logits.ndim == 5:
        pred_logits = pred_logits.squeeze(2)  # (B, 1, H, W)
    elif pred_logits.ndim == 3:
        pred_logits = pred_logits.unsqueeze(1)  # (B, 1, H, W)

    probs = torch.sigmoid(pred_logits)

    if gt_masks.ndim == 3:
        gt_masks = gt_masks.unsqueeze(1)
    gt_masks = (gt_masks > 0.5).float()

    preds = (probs > threshold).float()

    dims = (1, 2, 3)
    intersection = (preds * gt_masks).sum(dim=dims)
    pred_area = preds.sum(dim=dims)
    gt_area = gt_masks.sum(dim=dims)
    union = pred_area + gt_area - intersection

    dice = (2 * intersection + eps) / (pred_area + gt_area + eps)
    iou = (intersection + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()
