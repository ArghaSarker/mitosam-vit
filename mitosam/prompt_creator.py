import numpy as np
from scipy import ndimage


def get_bounding_boxes(mask, perturb=True, max_perturb=20):
    """
    Compute bounding boxes for each connected object in a binary mask.

    Args:
        mask (ndarray): Binary mask (H, W), with nonzero pixels for objects.
        perturb (bool): If True, adds small random perturbation to each bbox.
        max_perturb (int): Maximum random expansion (in pixels).

    Returns:
        boxes (list of [x_min, y_min, x_max, y_max])
    """
    boxes = []
    labeled_mask, num_features = ndimage.label(mask > 0)

    H, W = mask.shape
    for i in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled_mask == i)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        if perturb:
            x_min = max(0, x_min - np.random.randint(0, max_perturb))
            x_max = min(W, x_max + np.random.randint(0, max_perturb))
            y_min = max(0, y_min - np.random.randint(0, max_perturb))
            y_max = min(H, y_max + np.random.randint(0, max_perturb))

        boxes.append([x_min, y_min, x_max, y_max])

    return boxes


def get_union_bounding_box(mask, perturb: bool = True, max_perturb: int = 20):
    """
    Compute a single bounding box that covers all foreground in the mask.

    Args:
        mask (ndarray): Binary mask (H, W).
        perturb (bool): If True, adds small random perturbation.
        max_perturb (int): Maximum random expansion (pixels).

    Returns:
        list[int] | None: [x_min, y_min, x_max, y_max], or None if mask is empty.
    """
    mask_array = np.array(mask)
    y_indices, x_indices = np.where(mask_array > 0)

    # If there is no foreground, return None
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    H, W = mask_array.shape
    if perturb:
        x_min = max(0, x_min - np.random.randint(0, max_perturb))
        x_max = min(W, x_max + np.random.randint(0, max_perturb))
        y_min = max(0, y_min - np.random.randint(0, max_perturb))
        y_max = min(H, y_max + np.random.randint(0, max_perturb))

    return [x_min, y_min, x_max, y_max]





def build_sam_prompt_dataset(
    images,
    masks,
    prompt_augment: bool = True,
    include_union_box: bool = True,
    perturb: bool = True,
    max_perturb: int = 20,
    skip_empty_masks: bool = True,
):
    """
    Build a dataset of SAM prompts (bounding boxes) from images and masks.

    - If prompt_augment is True:
        For each (image, mask), compute per-object boxes
        and return multiple samples:
            (image, mask, bbox=boxA),
            (image, mask, bbox=boxB), ...
        Optionally also add a union box sample.

    - If prompt_augment is False:
        For each (image, mask), compute ONE union bounding box
        that covers all foreground:
            (image, mask, bbox=union_box)

    Args:
        images: Iterable of images (np.ndarray or PIL.Image).
        masks: Iterable of masks (same length as images).
        prompt_augment: If True, expand per-object; if False, one union box.
        include_union_box: Only used when prompt_augment=True.
        perturb: Whether to perturb bounding boxes.
        max_perturb: Max random expansion in pixels.
        skip_empty_masks: If True, skip samples with no foreground.

    Returns:
        list of dict:
            [{"image": img, "mask": mask, "bbox": [x_min, y_min, x_max, y_max]}, ...]
    """
    if len(images) != len(masks):
        raise ValueError(
            f"images and masks must have same length, "
            f"got {len(images)} and {len(masks)}"
        )

    dataset = []

    for img, mask in zip(images, masks):
        mask_array = np.array(mask)

        if prompt_augment:
            # Many boxes per mask (one per connected component)
            boxes = get_bounding_boxes(
                mask_array,
                perturb=perturb,
                max_perturb=max_perturb,
            )

            if len(boxes) == 0:
                if skip_empty_masks:
                    continue
                else:
                    # Keep but with no bbox (you can decide how to handle later)
                    dataset.append({"image": img, "mask": mask, "bbox": None})
                    continue

            # One sample per box
            for box in boxes:
                dataset.append({
                    "image": img,
                    "mask": mask,
                    "bbox": box,
                })

            # Optional extra union box sample
            if include_union_box and len(boxes) > 1:
                all_xmin = min(b[0] for b in boxes)
                all_ymin = min(b[1] for b in boxes)
                all_xmax = max(b[2] for b in boxes)
                all_ymax = max(b[3] for b in boxes)
                union_box = [all_xmin, all_ymin, all_xmax, all_ymax]

                dataset.append({
                    "image": img,
                    "mask": mask,
                    "bbox": union_box,
                })

        else:
            # Exactly ONE box per image: union of all foreground
            union_box = get_union_bounding_box(
                mask_array,
                perturb=perturb,
                max_perturb=max_perturb,
            )

            if union_box is None:
                if skip_empty_masks:
                    continue
                else:
                    dataset.append({"image": img, "mask": mask, "bbox": None})
                    continue

            dataset.append({
                "image": img,
                "mask": mask,
                "bbox": union_box,
            })

    print(
        f"Built SAM prompt dataset: {len(images)} images → "
        f"{len(dataset)} samples (prompt_augment={prompt_augment})"
    )
    return dataset



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_prompted_dataset(
    expanded_data,
    n_cols: int = 5,
    cell_size: float = 2.2,
    start_idx: int = 0,
    random_pick: bool = False,
    seed: int | None = None,
):
    """
    Visualize samples from an expanded dataset.

    Top row:  images
    Bottom row: masks with bounding boxes

    Args:
        expanded_data: list of dicts, each like
            {"image": img, "mask": mask, "bbox": [x_min, y_min, x_max, y_max]}
            or
            {"image": img, "mask": mask, "bboxes": [[...], [...], ...]}
        n_cols: number of samples to show in one row.
        cell_size: base size of each cell in inches.
        start_idx: starting index (used when random_pick=False).
        random_pick: if True, choose random samples instead of sequential.
        seed: random seed for reproducibility when random_pick=True.

    Returns:
        fig, axes: Matplotlib Figure and Axes array.
    """
    total = len(expanded_data)
    if total == 0:
        raise ValueError("expanded_data is empty, nothing to visualize.")

    n_cols = min(n_cols, total)

    # --- choose indices ---
    if random_pick:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=n_cols, replace=False)
    else:
        indices = [(start_idx + i) % total for i in range(n_cols)]

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(cell_size * n_cols, cell_size * 2)
    )

    # Handle the case n_cols == 1 (matplotlib returns 1D axes)
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, idx in enumerate(indices):
        entry = expanded_data[idx]
        img = np.array(entry["image"])
        mask = np.array(entry["mask"])

        # Support both "bbox" and "bboxes" formats
        if "bbox" in entry and entry["bbox"] is not None:
            bboxes = [entry["bbox"]]
        elif "bboxes" in entry and entry["bboxes"] is not None:
            bboxes = entry["bboxes"]
        else:
            bboxes = []

        # --- Top row: image ---
        ax_img = axes[0, col]
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"Image #{idx}", fontsize=8)
        ax_img.axis("off")

        # --- Bottom row: mask + bbox(es) ---
        ax_mask = axes[1, col]
        ax_mask.imshow(mask, cmap="gray")

        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
            )
            ax_mask.add_patch(rect)

        if len(bboxes) == 1:
            title = f"bbox={bboxes[0]}"
        elif len(bboxes) > 1:
            title = f"{len(bboxes)} boxes"
        else:
            title = "no bbox"

        ax_mask.set_title(title, fontsize=7)
        ax_mask.axis("off")

    fig.suptitle("top : image, Bottom: mask + prompt", fontsize=10)
    plt.tight_layout(pad=1.0)
    return fig, axes
