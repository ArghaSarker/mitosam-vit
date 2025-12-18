import sys
import os
import random
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from PIL import Image
from tqdm import tqdm

from paths import PROJECT_ROOT
from prompt_creator import build_sam_prompt_dataset
from metrics import dice_score, iou_score, precision_recall_accuracy
from xai import (
    predict_mask_and_prob,
    occlusion_sensitivity_sam,
    integrated_gradients_sam,
    overlay_prompt,
)

from utils import normalize_heatmap, make_tp_fp_fn_overlay

# -------------------------------------------------------------------------
# 0. Configuration
# -------------------------------------------------------------------------

SEED = 42
BASE_MODEL_NAME = "facebook/sam-vit-base"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

VAL_OUTPUT_FILENAME = PROCESSED_DIR / "val_data_processed.npz"

EXPERIMENT_NAME = "BASE_SAM_BBOX"
EXPERIMENT_REPORT_DIR = REPORTS_DIR / EXPERIMENT_NAME
VISUALIZATION_DIR = EXPERIMENT_REPORT_DIR / "inference_visualisations"

BATCH_SIZE = 8
PREDICTION_THRESHOLD = 0.7

METRICS_CSV_PATH = EXPERIMENT_REPORT_DIR / "inference_metrics_BASE_SAM_BBOX.csv"

VISUALIZE_IDX = 13

OCCLUSION_PATCH = 16
OCCLUSION_STRIDE = 8
IG_STEPS = 64


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return img[..., 0]


set_seed()

# -------------------------------------------------------------------------
# 1. Environment / versions
# -------------------------------------------------------------------------

print("\n=== Environment ===")
print("Python:", sys.version.splitlines()[0])
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("=== End environment ===\n")

# -------------------------------------------------------------------------
# 2. Paths
# -------------------------------------------------------------------------

EXPERIMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("VAL_OUTPUT_FILENAME:", VAL_OUTPUT_FILENAME)
print("EXPERIMENT_REPORT_DIR:", EXPERIMENT_REPORT_DIR)

# -------------------------------------------------------------------------
# 3. Load processed validation data
# -------------------------------------------------------------------------

with np.load(VAL_OUTPUT_FILENAME) as data:
    val_img = data["images"]
    val_mask = data["masks"]

print(f"\nLoaded validation data from {VAL_OUTPUT_FILENAME.name}")
print(f"  val_img:  {val_img.shape}, dtype={val_img.dtype}")
print(f"  val_mask: {val_mask.shape}, dtype={val_mask.dtype}")

# -------------------------------------------------------------------------
# 4. Build prompts (ONE union bbox per image)
# -------------------------------------------------------------------------

val_dataset_dict = {
    "image": [Image.fromarray(img) for img in val_img],
    "label": [Image.fromarray(mask) for mask in val_mask],
}

expanded_val = build_sam_prompt_dataset(
    images=val_dataset_dict["image"],
    masks=val_dataset_dict["label"],
    prompt_augment=False,      # union bbox
    include_union_box=True,
    perturb=False,
    max_perturb=0,
)

# -------------------------------------------------------------------------
# 5. Dataset + DataLoader
# -------------------------------------------------------------------------

class SAMDataset(Dataset):
    def __init__(self, expanded_data, processor):
        self.samples = expanded_data
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        image = entry["image"]
        mask = np.array(entry["mask"])
        box = entry["bbox"]
        if box is None:
            raise ValueError(f"Sample {idx} has bbox=None (unexpected with skip_empty_masks=True)")

        enc = self.processor(image, input_boxes=[[box]], return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["ground_truth_mask"] = mask
        enc["sample_idx"] = torch.tensor(idx, dtype=torch.long)
        return enc


processor = SamProcessor.from_pretrained(BASE_MODEL_NAME)
val_dataset = SAMDataset(expanded_data=expanded_val, processor=processor)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# -------------------------------------------------------------------------
# 6. Model (BASE SAM ONLY)
# -------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

model = SamModel.from_pretrained(BASE_MODEL_NAME)
model.to(device)
model.eval()

print("Running BASE SAM (no LoRA, no checkpoint).")

# -------------------------------------------------------------------------
# 7. Inference on full validation set
# -------------------------------------------------------------------------

rows = []

for batch in tqdm(val_dataloader, desc="Inference [Val]"):
    pixel_values = batch["pixel_values"].to(device)
    input_boxes = batch["input_boxes"].to(device)
    ground_truth_masks = batch["ground_truth_mask"].float().to(device)
    sample_idx = batch["sample_idx"].cpu().numpy().tolist()

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False,
        )

    pred_logits = outputs.pred_masks
    if pred_logits.ndim == 5:
        pred_logits = pred_logits.squeeze(1).squeeze(1)
    else:
        pred_logits = pred_logits.squeeze(1)

    if ground_truth_masks.shape[-2:] != pred_logits.shape[-2:]:
        ground_truth_masks = torch.nn.functional.interpolate(
            ground_truth_masks.unsqueeze(1),
            size=pred_logits.shape[-2:],
            mode="nearest",
        ).squeeze(1)

    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > PREDICTION_THRESHOLD).to(torch.uint8)

    pred_np = pred_bin.cpu().numpy()
    gt_np = (ground_truth_masks > 0).to(torch.uint8).cpu().numpy()

    for j in range(pred_np.shape[0]):
        d = dice_score(pred_np[j], gt_np[j])
        i = iou_score(pred_np[j], gt_np[j])
        p, r, a = precision_recall_accuracy(pred_np[j], gt_np[j])
        rows.append((sample_idx[j], d, i, p, r, a))

import csv
with open(METRICS_CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "dice", "iou", "precision", "recall", "accuracy"])
    w.writerows(rows)

print("\nSaved metrics to:", METRICS_CSV_PATH)

# -------------------------------------------------------------------------
# 8. Visualisation (optional)
# -------------------------------------------------------------------------

if VISUALIZE_IDX is not None:
    set_seed(SEED)

    if VISUALIZE_IDX < 0 or VISUALIZE_IDX >= len(expanded_val):
        raise ValueError(f"VISUALIZE_IDX {VISUALIZE_IDX} out of range 0..{len(expanded_val)-1}")

    sample = expanded_val[VISUALIZE_IDX]
    img_np = np.array(sample["image"])
    gt_np = (np.array(sample["mask"]) > 0).astype(np.uint8)
    box = sample["bbox"]

    pred_bin, prob_map = predict_mask_and_prob(
        model,
        processor,
        img_np,
        "boxes",
        box,
        device=device,
        threshold=PREDICTION_THRESHOLD,
    )

    pred_bin = np.squeeze(pred_bin).astype(np.uint8)
    prob_map = np.squeeze(prob_map).astype(np.float32)

    dice_v = dice_score(pred_bin, gt_np)
    iou_v = iou_score(pred_bin, gt_np)

    occ_map = occlusion_sensitivity_sam(
        model,
        processor,
        img_np,
        gt_np,
        "boxes",
        box,
        patch=OCCLUSION_PATCH,
        stride=OCCLUSION_STRIDE,
        device=device,
    )

    sal_map = integrated_gradients_sam(
        model,
        processor,
        img_np,
        "boxes",
        box,
        steps=IG_STEPS,
        device=device,
    )

    prompt_overlay = overlay_prompt(img_np, "boxes", box)
    base_gray = to_gray(img_np)
    overlay_rgb = make_tp_fp_fn_overlay(base_gray, gt_np, pred_bin, alpha=0.75)

    occ_norm = normalize_heatmap(occ_map, p_low=2, p_high=98)
    sal_norm = normalize_heatmap(sal_map, p_low=2, p_high=99)

    fig, axes = plt.subplots(1, 8, figsize=(34, 4))

    axes[0].imshow(prompt_overlay)
    axes[0].set_title("Input image + bbox prompt", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(gt_np, cmap="gray")
    axes[1].set_title("Ground truth mask", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(pred_bin, cmap="gray")
    axes[2].set_title("Predicted mask", fontsize=14)
    axes[2].axis("off")

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("GT vs Pred (TP=green, FN=red, FP=blue)", fontsize=14)
    axes[3].axis("off")

    im_prob = axes[4].imshow(prob_map, vmin=0.0, vmax=1.0)
    axes[4].set_title("Probability map", fontsize=14)
    axes[4].axis("off")
    fig.colorbar(im_prob, ax=axes[4], fraction=0.046, pad=0.04)

    im_occ = axes[5].imshow(occ_norm, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[5].set_title("Occlusion sensitivity", fontsize=14)
    axes[5].axis("off")
    fig.colorbar(im_occ, ax=axes[5], fraction=0.046, pad=0.04)

    im_sal = axes[6].imshow(sal_norm, cmap="plasma", vmin=0.0, vmax=1.0)
    axes[6].set_title("Saliency (Integrated Gradients)", fontsize=14)
    axes[6].axis("off")
    fig.colorbar(im_sal, ax=axes[6], fraction=0.046, pad=0.04)

    txt = f"Dice: {dice_v:.3f}\nIoU:  {iou_v:.3f}\n"
    axes[7].text(0.35, 0.35, txt, fontsize=18)
    axes[7].set_title("Metrics", fontsize=14)
    axes[7].axis("off")

    plt.tight_layout()
    out_path = VISUALIZATION_DIR / f"val_sample_{VISUALIZE_IDX:03d}_BASE_SAM_BBOX.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved visualisation to:", out_path)

print("Inference complete.")
