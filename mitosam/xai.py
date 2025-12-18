from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from metrics import dice_score


def _prepare_inputs(
    processor,
    image: Image.Image,
    prompt_type: str,
    prompt_data,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    if prompt_type not in {"boxes", "points"}:
        raise ValueError(
            f"Unsupported prompt_type '{prompt_type}'. Expected 'boxes' or 'points'."
        )

    if prompt_type == "boxes":
        if prompt_data is None or len(prompt_data) != 4:
            raise ValueError(
                "For boxes prompt, prompt_data must be a list [x_min, y_min, x_max, y_max]"
            )
        processed = processor(
            image,
            input_boxes=[[prompt_data]],
            return_tensors="pt",
        )
    else:
        try:
            points, labels = prompt_data
        except Exception as e:
            raise ValueError(
                "For points prompt, prompt_data must be a tuple (points, labels)"
            ) from e

        processed = processor(
            image,
            input_points=[np.asarray(points, dtype=float).tolist()],
            input_labels=[np.asarray(labels, dtype=int).tolist()],
            return_tensors="pt",
        )

    pixel_values = processed["pixel_values"].to(device)
    extra_args = {}
    for key in [
        "input_boxes",
        "input_points",
        "input_labels",
        "original_sizes",
        "reshaped_input_sizes",
    ]:
        if key in processed:
            extra_args[key] = processed[key].to(device)

    return pixel_values, extra_args


def _pred_masks_to_logits(pred_masks: torch.Tensor) -> torch.Tensor:
    if pred_masks.ndim == 5:
        return pred_masks.squeeze(1).squeeze(1)  # (B, H, W)
    if pred_masks.ndim == 4:
        return pred_masks.squeeze(1)  # (B, H, W)
    if pred_masks.ndim == 3:
        return pred_masks  # (B, H, W)
    raise ValueError(f"Unexpected pred_masks shape: {tuple(pred_masks.shape)}")


def predict_mask_and_prob(
    model,
    processor,
    image_np: np.ndarray,
    prompt_type: str,
    prompt_data,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if image_np.ndim == 2:
        image_pil = Image.fromarray(image_np)
    elif image_np.ndim == 3 and image_np.shape[2] == 3:
        image_pil = Image.fromarray(image_np)
    else:
        raise ValueError(
            f"Unsupported image shape {image_np.shape}; expected (H,W) or (H,W,3)"
        )

    pixel_values, extra_args = _prepare_inputs(
        processor, image_pil, prompt_type, prompt_data, device
    )

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            multimask_output=False,
            **extra_args,
        )

        logits = _pred_masks_to_logits(outputs.pred_masks)  # (1, Hm, Wm)
        prob = torch.sigmoid(logits)  # (1, Hm, Wm)

    H, W = image_np.shape[:2]
    prob_up = F.interpolate(
        prob.unsqueeze(1),  # (1,1,Hm,Wm)
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    prob_np = prob_up.detach().cpu().numpy().astype(np.float32)
    pred_bin = (prob_np > threshold).astype(np.uint8)
    return pred_bin, prob_np


def integrated_gradients_sam(
    model,
    processor,
    image_np: np.ndarray,
    prompt_type: str,
    prompt_data,
    steps: int = 24,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()

    if image_np.ndim == 2:
        image_pil = Image.fromarray(image_np)
    elif image_np.ndim == 3 and image_np.shape[2] == 3:
        image_pil = Image.fromarray(image_np)
    else:
        raise ValueError(
            f"Unsupported image shape {image_np.shape}; expected (H,W) or (H,W,3)"
        )

    pixel_values, extra_args = _prepare_inputs(
        processor, image_pil, prompt_type, prompt_data, device
    )

    baseline = torch.zeros_like(pixel_values)
    alphas = torch.linspace(0.0, 1.0, steps).to(device)

    integrated_grads = torch.zeros_like(pixel_values, dtype=torch.float32)

    for alpha in alphas:
        scaled_input = baseline + alpha * (pixel_values - baseline)
        scaled_input.requires_grad_(True)

        outputs = model(
            pixel_values=scaled_input,
            multimask_output=False,
            **extra_args,
        )

        logits = _pred_masks_to_logits(outputs.pred_masks)  # (1, Hm, Wm)
        score = torch.sigmoid(logits).sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        grad = scaled_input.grad.detach()
        integrated_grads += grad

    integrated_grads /= float(steps)
    ig = (pixel_values - baseline) * integrated_grads

    ig_map = ig.abs().sum(dim=1).squeeze(0)  # (H', W')
    ig_map = ig_map.detach().cpu().numpy().astype(np.float32)

    H, W = image_np.shape[:2]
    if ig_map.shape != (H, W):
        ig_t = torch.from_numpy(ig_map).unsqueeze(0).unsqueeze(0)
        ig_up = F.interpolate(ig_t, size=(H, W), mode="bilinear", align_corners=False)
        ig_map = ig_up.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    return ig_map


def occlusion_sensitivity_sam(
    model,
    processor,
    image_np: np.ndarray,
    mask_np: np.ndarray,
    prompt_type: str,
    prompt_data,
    patch: int = 48,
    stride: int = 24,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    model.eval()
    H, W = image_np.shape[:2]

    pred_bin, _ = predict_mask_and_prob(
        model,
        processor,
        image_np,
        prompt_type,
        prompt_data,
        device=device,
        threshold=threshold,
    )
    base_dice = dice_score(pred_bin, mask_np)

    heat = np.zeros((H, W), dtype=np.float32)

    if image_np.ndim == 2:
        mean_pixel = float(image_np.mean())
    else:
        mean_pixel = image_np.mean(axis=(0, 1), dtype=float)

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(H, y0 + patch)
            x1 = min(W, x0 + patch)

            occluded = image_np.copy()
            if image_np.ndim == 2:
                occluded[y0:y1, x0:x1] = mean_pixel
            else:
                occluded[y0:y1, x0:x1, :] = mean_pixel

            occl_pred_bin, _ = predict_mask_and_prob(
                model,
                processor,
                occluded,
                prompt_type,
                prompt_data,
                device=device,
                threshold=threshold,
            )
            d = dice_score(occl_pred_bin, mask_np)
            drop = max(0.0, base_dice - d)

            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], drop)

    return heat.astype(np.float32)


def overlay_prompt(
    image_np: np.ndarray,
    prompt_type: str,
    prompt_data,
    positive_color: Tuple[int, int, int] = (0, 255, 0),
    negative_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    if image_np.ndim == 2:
        img_rgb = np.stack([image_np] * 3, axis=-1).astype(np.uint8)
    else:
        img_rgb = image_np.astype(np.uint8).copy()

    H, W = img_rgb.shape[:2]

    if prompt_type == "boxes":
        x0, y0, x1, y1 = map(int, prompt_data)

        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H, y1))

        img_rgb[y0:y0 + 2, x0:x1] = positive_color
        img_rgb[y1 - 2:y1, x0:x1] = positive_color
        img_rgb[y0:y1, x0:x0 + 2] = positive_color
        img_rgb[y0:y1, x1 - 2:x1] = positive_color

    elif prompt_type == "points":
        points, labels = prompt_data
        points = np.asarray(points, dtype=int)
        labels = np.asarray(labels, dtype=int)

        for (x, y), lbl in zip(points, labels):
            col = positive_color if int(lbl) == 1 else negative_color
            yy, xx = int(y), int(x)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    yi = yy + dy
                    xi = xx + dx
                    if 0 <= yi < H and 0 <= xi < W:
                        img_rgb[yi, xi] = col
    else:
        raise ValueError(
            f"Unsupported prompt_type '{prompt_type}'. Expected 'boxes' or 'points'."
        )

    return img_rgb
