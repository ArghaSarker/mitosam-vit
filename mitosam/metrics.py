from __future__ import annotations

from typing import Tuple
import numpy as np


def _prepare_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.squeeze(pred_mask)
    gt = np.squeeze(gt_mask)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred_mask has shape {pred.shape}, gt_mask has shape {gt.shape}"
        )

    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    return pred_bin, gt_bin


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred_bin, gt_bin = _prepare_masks(pred_mask, gt_mask)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    sum_foreground = pred_bin.sum() + gt_bin.sum()
    return float((2.0 * intersection) / (sum_foreground + eps))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred_bin, gt_bin = _prepare_masks(pred_mask, gt_mask)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(intersection / (union + eps))


def precision_recall_accuracy(
    pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8
) -> Tuple[float, float, float]:
    pred_bin, gt_bin = _prepare_masks(pred_mask, gt_mask)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return float(precision), float(recall), float(accuracy)
