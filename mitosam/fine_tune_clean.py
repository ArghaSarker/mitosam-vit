import transformers
print(transformers.__version__)

import sys
print(sys.version)

from peft import TaskType
print(TaskType.__members__)

import numpy as np
import matplotlib.pyplot as plt
import tifffile  # currently unused, kept for now
import os
from patchify import patchify  # currently unused, kept for now
import random  # currently unused, kept for now
from scipy import ndimage  # currently unused, kept for now

from pathlib import Path
from paths import PROJECT_ROOT  # <--- project root (MitoSAM-ViT)

from peft import LoraConfig, get_peft_model, TaskType
from utils import plot_random_image_mask_pairs
from prompt_creator import (
    build_sam_prompt_dataset,
    get_bounding_boxes,
    get_union_bounding_box,
    visualize_prompted_dataset,
)

# -------------------------------------------------------------------------
# 0. Paths: data, models, reports
# -------------------------------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

train_output_filename = PROCESSED_DIR / "train_data_processed.npz"
val_output_filename   = PROCESSED_DIR / "val_data_processed.npz"

print("\nPROJECT_ROOT:", PROJECT_ROOT)
print("PROCESSED_DIR:", PROCESSED_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("REPORTS_DIR:", REPORTS_DIR)

# -------------------------------------------------------------------------
# 1. Load processed NPZ data
# -------------------------------------------------------------------------

# Load training data
try:
    with np.load(train_output_filename) as data:
        train_img = data["images"]
        train_mask = data["masks"]
    print(f"\n--- Loaded Training Data from {train_output_filename.name} ---")
    print(f"  'train_img' shape: {train_img.shape}, dtype: {train_img.dtype}")
    print(f"  'train_mask' shape: {train_mask.shape}, dtype: {train_mask.dtype}")
except FileNotFoundError:
    print(f"Error: The file {train_output_filename} was not found. Please ensure it exists.")
    raise
except Exception as e:
    print(f"An error occurred while loading the training NPZ file: {e}")
    raise

# Load validation data
try:
    with np.load(val_output_filename) as data:
        val_img = data["images"]
        val_mask = data["masks"]
    print(f"\n--- Loaded Validation Data from {val_output_filename.name} ---")
    print(f"  'val_img' shape: {val_img.shape}, dtype: {val_img.dtype}")
    print(f"  'val_mask' shape: {val_mask.shape}, dtype: {val_mask.dtype}")
except FileNotFoundError:
    print(f"Error: The file {val_output_filename} was not found. Please ensure it exists.")
    raise
except Exception as e:
    print(f"An error occurred while loading the validation NPZ file: {e}")
    raise

# -------------------------------------------------------------------------
# 2. Debug subset (keep exactly as you had it)
# -------------------------------------------------------------------------

# train_img = train_img[:20]
# train_mask = train_mask[:20]

# val_img = val_img[:5]
# val_mask = val_mask[:5]

print("\nAfter debug slicing:")
print(f"  'train_img' shape: {train_img.shape}, dtype: {train_img.dtype}")
print(f"  'train_mask' shape: {train_mask.shape}, dtype: {train_mask.dtype}")
print(f"  'val_img' shape: {val_img.shape}, dtype: {val_img.dtype}")
print(f"  'val_mask' shape: {val_mask.shape}, dtype: {val_mask.dtype}")

# -------------------------------------------------------------------------
# 3. Quick visualization of raw train/val patches → reports/
# -------------------------------------------------------------------------

fig, axes = plot_random_image_mask_pairs(train_img, train_mask, num_samples=5, seed=42)
fig.savefig(REPORTS_DIR / "sam_raw_train_samples.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, axes = plot_random_image_mask_pairs(val_img, val_mask, num_samples=5, seed=24)
fig.savefig(REPORTS_DIR / "sam_raw_val_samples.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# 4. Wrap into HF-style dicts + Datasets (for info / sanity)
# -------------------------------------------------------------------------

from datasets import Dataset
from PIL import Image

train_dataset_dict = {
    "image": [Image.fromarray(img) for img in train_img],
    "label": [Image.fromarray(mask) for mask in train_mask],
}

val_dataset_dict = {
    "image": [Image.fromarray(img) for img in val_img],
    "label": [Image.fromarray(mask) for mask in val_mask],
}

train_dataset_hf = Dataset.from_dict(train_dataset_dict)
val_dataset_hf   = Dataset.from_dict(val_dataset_dict)

print(f"\ninfo train dataset: {train_dataset_hf}")
print(f"info val dataset: {val_dataset_hf}")

# -------------------------------------------------------------------------
# 5. Build SAM prompts (bounding boxes) using your helper
# -------------------------------------------------------------------------

expanded_train = build_sam_prompt_dataset(
    images=train_dataset_dict["image"],
    masks=train_dataset_dict["label"],
    prompt_augment=False,        # one union bbox per image
    include_union_box=True,      # no effect when prompt_augment=False
    perturb=True,
    max_perturb=20,
)

expanded_val = build_sam_prompt_dataset(
    images=val_dataset_dict["image"],
    masks=val_dataset_dict["label"],
    prompt_augment=False,
    include_union_box=True,
    perturb=True,
    max_perturb=20,
)

print(f"\nExpanded train samples: {len(expanded_train)}")
print(f"Expanded val   samples: {len(expanded_val)}")

# Visualize prompted samples → reports/
fig, axes = visualize_prompted_dataset(expanded_train, n_cols=5, start_idx=0)
fig.savefig(REPORTS_DIR / "sam_prompted_train_samples.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, axes = visualize_prompted_dataset(expanded_val, n_cols=5, start_idx=0)
fig.savefig(REPORTS_DIR / "sam_prompted_val_samples.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# 6. SAMDataset class + DataLoaders
# -------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor

class SAMDataset(Dataset):
    """
    Wraps expanded_data entries: {"image": PIL.Image, "mask": mask, "bbox": [x_min,y_min,x_max,y_max]}
    and uses SamProcessor to create SAM inputs.
    """
    def __init__(self, expanded_data, processor):
        self.samples = expanded_data
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        image = entry["image"]          # PIL.Image
        mask  = np.array(entry["mask"]) # (H, W), e.g., 0/255
        box   = entry["bbox"]           # [x_min, y_min, x_max, y_max]

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # SAM processor: handles resizing + box coordinate transform internally
        enc = self.processor(image, input_boxes=[[box]], return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove batch dim

        # Ground truth mask as float; same resolution as original image
        enc["ground_truth_mask"] = torch.from_numpy(mask).float()  # (H, W)

        return enc

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

train_dataset = SAMDataset(expanded_train, processor)
val_dataset   = SAMDataset(expanded_val,   processor)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,  drop_last=False)
val_dataloader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, drop_last=False)

# Quick sanity print for one sample
enc0 = train_dataset[0]
print("\nSample[0]:")
print("  pixel_values:", tuple(enc0["pixel_values"].shape))            # (3, 1024, 1024) typically
print("  input_boxes:", tuple(enc0["input_boxes"].shape))              # (1, 4)
print("  original_sizes:", enc0["original_sizes"].tolist())            # [256, 256]
print("  reshaped_input_sizes:", enc0["reshaped_input_sizes"].tolist())# [1024, 1024]
print("  gt mask:", tuple(enc0["ground_truth_mask"].shape))            # (256, 256)

# -------------------------------------------------------------------------
# 7. Visual sanity check of SAM preprocessing → reports/
# -------------------------------------------------------------------------

from sam_helper import visualize_sam_sample

fig, axes = visualize_sam_sample(train_dataset, idx=13)
fig.savefig(REPORTS_DIR / "sam_sample_debug.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# 8. Define SAM model + LoRA + loss
# -------------------------------------------------------------------------

from transformers import SamModel
from monai.losses import DiceFocalLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

# Base SAM model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# LoRA config on attention projections
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)

# Segmentation loss 
seg_loss = DiceFocalLoss(
    sigmoid=True,
    lambda_dice=1.0,
    lambda_focal=1.0,
    reduction="mean",
)

# -------------------------------------------------------------------------
# 9. Optimizer, scheduler, early stopping
# -------------------------------------------------------------------------

from tqdm import tqdm
from statistics import mean

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    threshold=0.001,
    threshold_mode="abs",
)

early_stopping_patience = 7
early_stopping_min_delta = 0.001

best_val_loss = float("inf")
epochs_without_improvement = 0

# Save best checkpoint under models/
best_model_path = MODELS_DIR / "sam_lora_best.pth"

# -------------------------------------------------------------------------
# 10. Training + validation loop 
# -------------------------------------------------------------------------

num_epochs = 200
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # ---- TRAIN ----
    model.train()
    epoch_train_losses = []

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes  = batch["input_boxes"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)  # (B, H, W)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False,
        )

        # SAM returns (B, 1, 1, H, W) for one box + one mask
        # squeeze(1) → (B, 1, H, W), matching tutorial style
        predicted_masks = outputs.pred_masks.squeeze(1)   # (B, 1, H, W)

        loss = seg_loss(
            predicted_masks,
            ground_truth_masks.unsqueeze(1),              # (B, 1, H, W)
        )

        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())

    mean_train_loss = mean(epoch_train_losses)
    train_losses.append(mean_train_loss)

    # ---- VALIDATION ----
    model.eval()
    epoch_val_losses = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes  = batch["input_boxes"].to(device)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )

            predicted_masks = outputs.pred_masks.squeeze(1)  # (B, 1, H, W)

            val_loss = seg_loss(
                predicted_masks,
                ground_truth_masks.unsqueeze(1),
            )
            epoch_val_losses.append(val_loss.item())

    mean_val_loss = mean(epoch_val_losses)
    val_losses.append(mean_val_loss)

    # ---- LR scheduler step ----
    scheduler.step(mean_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {mean_train_loss:.4f} | "
        f"Val Loss: {mean_val_loss:.6f} | "
        f"LR: {current_lr:.2e}"
    )

    # ---- Early stopping ----
    if mean_val_loss < best_val_loss - early_stopping_min_delta:
        best_val_loss = mean_val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), best_model_path)
        print(f"  ✅ New best val loss: {best_val_loss:.6f}. Model saved to {best_model_path}.")
    else:
        epochs_without_improvement += 1
        print(f"  No meaningful improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print("⏹ Early stopping triggered.")
            break

# -------------------------------------------------------------------------
# 11. Reload best checkpoint and plot losses → reports/
# -------------------------------------------------------------------------

model.load_state_dict(torch.load(best_model_path, map_location=device))

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (DiceFocal) with LR scheduling & early stopping")
plt.legend()
plt.grid(True)
plt.savefig(REPORTS_DIR / "sam_training_validation_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()
