import sys
import os
import random
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from monai.losses import DiceFocalLoss
from statistics import mean
from PIL import Image
from tqdm import tqdm

from paths import PROJECT_ROOT
from utils import plot_random_image_mask_pairs
from prompt_creator import (
    build_sam_prompt_dataset,
    visualize_prompted_dataset,
)
from sam_LoRA import apply_lora_to_sam, freeze_sam_except_lora
from sam_helper import (
    visualize_sam_sample,
    compute_batch_dice_iou,
)

# -------------------------------------------------------------------------
# 0. Configuration
# -------------------------------------------------------------------------

SEED = 42

BASE_MODEL_NAME = "facebook/sam-vit-base"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_OUTPUT_FILENAME = PROCESSED_DIR / "train_data_processed.npz"
VAL_OUTPUT_FILENAME = PROCESSED_DIR / "val_data_processed.npz"

DEBUG_TRAIN_LIMIT = None
DEBUG_VAL_LIMIT = None

BATCH_SIZE = 4
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

PREDICTION_THRESHOLD = 0.7

EXPERIMENT_NAME = "SAM_ViT_Peft_rank64"

EXPERIMENT_DIR = MODELS_DIR / EXPERIMENT_NAME
EXPERIMENT_REPORT_DIR = REPORTS_DIR / EXPERIMENT_NAME

CONFIG_PATH = EXPERIMENT_DIR / "config.json"
BEST_MODEL_PATH = EXPERIMENT_DIR / "best_model.pth"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# -------------------------------------------------------------------------
# 1. Environment / versions
# -------------------------------------------------------------------------

print("\n=== Environment ===")
print("Python:", sys.version.splitlines()[0])
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("Platform:", sys.platform)
print("CPU (logical cores):", os.cpu_count())

if torch.cuda.is_available():
    print("CUDA available: True")
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = getattr(props, "total_memory", 0) / 1e9
        print(
            f" GPU {i}: name={props.name}, "
            f"compute_capability={props.major}.{props.minor}, "
            f"total_mem={total_mem:.2f} GB"
        )
else:
    print("CUDA available: False")

print("=== End environment ===\n")

# -------------------------------------------------------------------------
# 2. Paths: data, models, reports
# -------------------------------------------------------------------------

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nPROJECT_ROOT:", PROJECT_ROOT)
print("PROCESSED_DIR:", PROCESSED_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("EXPERIMENT_DIR:", EXPERIMENT_DIR)
print("REPORTS_DIR:", REPORTS_DIR)
print("EXPERIMENT_REPORT_DIR:", EXPERIMENT_REPORT_DIR)

# -------------------------------------------------------------------------
# 3. Load processed NPZ data
# -------------------------------------------------------------------------

with np.load(TRAIN_OUTPUT_FILENAME) as data:
    train_img = data["images"]
    train_mask = data["masks"]
print(f"\nLoaded training data from {TRAIN_OUTPUT_FILENAME.name}")
print(f"  train_img:  {train_img.shape}, dtype={train_img.dtype}")
print(f"  train_mask: {train_mask.shape}, dtype={train_mask.dtype}")

with np.load(VAL_OUTPUT_FILENAME) as data:
    val_img = data["images"]
    val_mask = data["masks"]
print(f"\nLoaded validation data from {VAL_OUTPUT_FILENAME.name}")
print(f"  val_img:  {val_img.shape}, dtype={val_img.dtype}")
print(f"  val_mask: {val_mask.shape}, dtype={val_mask.dtype}")

# -------------------------------------------------------------------------
# 4. Debug subset  -> for experimental runs on a small subset
# -------------------------------------------------------------------------
if DEBUG_TRAIN_LIMIT is not None:

    train_img = train_img[:DEBUG_TRAIN_LIMIT]
    train_mask = train_mask[:DEBUG_TRAIN_LIMIT]
    val_img = val_img[:DEBUG_VAL_LIMIT]
    val_mask = val_mask[:DEBUG_VAL_LIMIT]

print(
    f"\nUsing {len(train_img)} training and {len(val_img)} validation samples "
    f"(debug limits: {DEBUG_TRAIN_LIMIT}, {DEBUG_VAL_LIMIT})"
)

# -------------------------------------------------------------------------
# 5. Quick visualization of raw train/val patches → experiment reports/
# -------------------------------------------------------------------------

fig, _ = plot_random_image_mask_pairs(train_img, train_mask, num_samples=5, seed=42)
fig.savefig(
    EXPERIMENT_REPORT_DIR / "sam_raw_train_samples_metricies_added.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

fig, _ = plot_random_image_mask_pairs(val_img, val_mask, num_samples=5, seed=24)
fig.savefig(
    EXPERIMENT_REPORT_DIR / "sam_raw_val_samples_metricies_added.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# -------------------------------------------------------------------------
# 6. Wrap into dicts (PIL) for SAM prompt building
# -------------------------------------------------------------------------

train_dataset_dict = {
    "image": [Image.fromarray(img) for img in train_img],
    "label": [Image.fromarray(mask) for mask in train_mask],
}

val_dataset_dict = {
    "image": [Image.fromarray(img) for img in val_img],
    "label": [Image.fromarray(mask) for mask in val_mask],
}

print(f"\nTrain set size: {len(train_dataset_dict['image'])}")
print(f"Val set size:   {len(val_dataset_dict['image'])}")

# -------------------------------------------------------------------------
# 7. Build SAM prompts
# -------------------------------------------------------------------------

expanded_train = build_sam_prompt_dataset(
    images=train_dataset_dict["image"],
    masks=train_dataset_dict["label"],
    prompt_augment=True,
    include_union_box=True,
    perturb=False,
    max_perturb=20,
)

expanded_val = build_sam_prompt_dataset(
    images=val_dataset_dict["image"],
    masks=val_dataset_dict["label"],
    prompt_augment=False,
    include_union_box=True,
    perturb=False,
    max_perturb=20,
)

fig, _ = visualize_prompted_dataset(expanded_train, n_cols=5, start_idx=0)
fig.savefig(
    EXPERIMENT_REPORT_DIR / "sam_prompted_train_samples_metricies_added.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

fig, _ = visualize_prompted_dataset(expanded_val, n_cols=5, start_idx=0)
fig.savefig(
    EXPERIMENT_REPORT_DIR / "sam_prompted_val_samples_metricies_added.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# -------------------------------------------------------------------------
# 8. SAMDataset class + DataLoaders
# -------------------------------------------------------------------------


class SAMDataset(Dataset):
    """
    Wraps entries: {"image": PIL.Image, "mask": mask, "bbox": [x_min,y_min,x_max,y_max]}
    and uses SamProcessor to create SAM inputs.
    """

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

        enc = self.processor(image, input_boxes=[[box]], return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["ground_truth_mask"] = mask
        return enc


processor = SamProcessor.from_pretrained(BASE_MODEL_NAME)

train_dataset = SAMDataset(expanded_data=expanded_train, processor=processor)
val_dataset = SAMDataset(expanded_data=expanded_val, processor=processor)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)

enc0 = train_dataset[0]
print("\nSample[0]:")
print("  pixel_values:", tuple(enc0["pixel_values"].shape))
print("  input_boxes:", tuple(enc0["input_boxes"].shape))
print("  original_sizes:", enc0["original_sizes"].tolist())
print("  reshaped_input_sizes:", enc0["reshaped_input_sizes"].tolist())
print("  gt mask:", tuple(enc0["ground_truth_mask"].shape))

# -------------------------------------------------------------------------
# 9. Visual sanity check
# -------------------------------------------------------------------------

fig, _ = visualize_sam_sample(train_dataset, idx=13)
fig.savefig(
    EXPERIMENT_REPORT_DIR / "sam_sample_debug_metricies_added.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# -------------------------------------------------------------------------
# 10. Define SAM model + LoRA + parameter stats
# -------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

base_sam = SamModel.from_pretrained(BASE_MODEL_NAME)
apply_lora_to_sam(base_sam, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
freeze_sam_except_lora(base_sam)

model = base_sam


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


print("\nExample trainable parameters:")
count_printed = 0
for name, p in model.named_parameters():
    if p.requires_grad:
        print("  ", name, p.shape)
        count_printed += 1
        if count_printed >= 12:
            break
if count_printed == 0:
    print("  [WARNING] No trainable parameters found!")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

ve_total, ve_train = count_params(model.vision_encoder)
pe_total, pe_train = count_params(model.prompt_encoder)
md_total, md_train = count_params(model.mask_decoder)

print("\n=== Parameter stats ===")
print(f"Total params (whole model):     {total_params:,}")
print(
    f"Trainable params (whole model): {trainable_params:,} "
    f"({100.0 * trainable_params / total_params:.4f}%)\n"
)
print(
    f"vision_encoder : {ve_train:,} / {ve_total:,} trainable "
    f"({100.0 * ve_train / ve_total if ve_total > 0 else 0:.4f}%)"
)
print(
    f"prompt_encoder : {pe_train:,} / {pe_total:,} trainable "
    f"({100.0 * pe_train / pe_total if pe_total > 0 else 0:.4f}%)"
)
print(
    f"mask_decoder   : {md_train:,} / {md_total:,} trainable "
    f"({100.0 * md_train / md_total if md_total > 0 else 0:.4f}%)"
)
print("=======================\n")

model.to(device)

# -------------------------------------------------------------------------
# 11. Loss and metrics
# -------------------------------------------------------------------------

seg_loss = DiceFocalLoss(
    sigmoid=True,
    lambda_dice=0.8,
    lambda_focal=0.2,
    reduction="mean",
)

lora_ve_params = sum(
    p.numel()
    for name, p in model.named_parameters()
    if "vision_encoder" in name and "lora_" in name and p.requires_grad
)
print("LoRA params in vision_encoder:", lora_ve_params)
print("Should match ve_train:", ve_train)

# -------------------------------------------------------------------------
# 12. Optimizer, scheduler, early stopping
# -------------------------------------------------------------------------

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    threshold=0.001,
    threshold_mode="abs",
    min_lr=1e-7,
)

best_val_loss = float("inf")
epochs_without_improvement = 0

print(f"\nBest model will be saved to: {BEST_MODEL_PATH}")

# -------------------------------------------------------------------------
# 13. TRAINING LOOP
# -------------------------------------------------------------------------

train_losses, val_losses = [], []
lr_values = []

train_dice_history, val_dice_history = [], []
train_iou_history, val_iou_history = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_train_losses = []
    epoch_train_dice = []
    epoch_train_iou = []

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False,
        )

        predicted_masks = outputs.pred_masks.squeeze(1)

        if ground_truth_masks.shape[-2:] != predicted_masks.shape[-2:]:
            ground_truth_masks = torch.nn.functional.interpolate(
                ground_truth_masks.unsqueeze(1),
                size=predicted_masks.shape[-2:],
                mode="nearest",
            ).squeeze(1)

        loss = seg_loss(
            predicted_masks,
            ground_truth_masks.unsqueeze(1),
        )

        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())

        with torch.no_grad():
            dice_batch, iou_batch = compute_batch_dice_iou(
                predicted_masks.detach(), ground_truth_masks
            )
        epoch_train_dice.append(dice_batch)
        epoch_train_iou.append(iou_batch)

    mean_train_loss = mean(epoch_train_losses)
    train_losses.append(mean_train_loss)
    mean_train_dice = mean(epoch_train_dice)
    mean_train_iou = mean(epoch_train_iou)
    train_dice_history.append(mean_train_dice)
    train_iou_history.append(mean_train_iou)

    model.eval()
    epoch_val_losses = []
    epoch_val_dice = []
    epoch_val_iou = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )

            predicted_masks = outputs.pred_masks.squeeze(1)

            if ground_truth_masks.shape[-2:] != predicted_masks.shape[-2:]:
                ground_truth_masks = torch.nn.functional.interpolate(
                    ground_truth_masks.unsqueeze(1),
                    size=predicted_masks.shape[-2:],
                    mode="nearest",
                ).squeeze(1)

            val_loss = seg_loss(
                predicted_masks,
                ground_truth_masks.unsqueeze(1),
            )
            epoch_val_losses.append(val_loss.item())

            dice_batch, iou_batch = compute_batch_dice_iou(
                predicted_masks, ground_truth_masks
            )
            epoch_val_dice.append(dice_batch)
            epoch_val_iou.append(iou_batch)

    mean_val_loss = mean(epoch_val_losses)
    val_losses.append(mean_val_loss)
    mean_val_dice = mean(epoch_val_dice)
    mean_val_iou = mean(epoch_val_iou)
    val_dice_history.append(mean_val_dice)
    val_iou_history.append(mean_val_iou)

    scheduler.step(mean_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    lr_values.append(current_lr)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {mean_train_loss:.4f} | "
        f"Val Loss: {mean_val_loss:.6f} | "
        f"Train Dice: {mean_train_dice:.4f} | "
        f"Val Dice: {mean_val_dice:.4f} | "
        f"Train IoU: {mean_train_iou:.4f} | "
        f"Val IoU: {mean_val_iou:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if mean_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
        best_val_loss = mean_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  New best val loss: {best_val_loss:.6f}. Model saved.")
    else:
        epochs_without_improvement += 1
        print(f"  No meaningful improvement for {epochs_without_improvement} epoch(s).")
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

# -------------------------------------------------------------------------
# 14. Reload best checkpoint & save minimal inference config
# -------------------------------------------------------------------------

if BEST_MODEL_PATH.exists():
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print(f"\nLoaded best model from: {BEST_MODEL_PATH}")
else:
    print(
        f"\nWarning: Best model checkpoint not found at {BEST_MODEL_PATH}. "
        "Skipping load."
    )

config = {
    "base_model_name": BASE_MODEL_NAME,
    "checkpoint_path": str(BEST_MODEL_PATH),
    "lora": {
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
    },
    "prediction_threshold": PREDICTION_THRESHOLD,
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"Saved minimal inference config to: {CONFIG_PATH}")

# -------------------------------------------------------------------------
# 15. PLOTS → experiment reports/
# -------------------------------------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (DiceFocal)")
plt.legend()
plt.grid(True)
loss_curve_path = (
    EXPERIMENT_REPORT_DIR / "sam_training_validation_loss_curve_metricies_added.png"
)
plt.savefig(loss_curve_path, dpi=300, bbox_inches="tight")
print(f"Saved loss curve to: {loss_curve_path}")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(lr_values, label="Learning Rate", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule During Training")
plt.grid(True)
plt.legend()
lr_curve_path = EXPERIMENT_REPORT_DIR / "sam_learning_rate_curve_metricies_added.png"
plt.savefig(lr_curve_path, dpi=300, bbox_inches="tight")
print(f"Saved learning rate curve to: {lr_curve_path}")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(train_dice_history, label="Train Dice", marker="o")
plt.plot(val_dice_history, label="Validation Dice", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Training vs Validation Dice")
plt.legend()
plt.grid(True)
dice_curve_path = (
    EXPERIMENT_REPORT_DIR / "sam_training_validation_dice_curve_metricies_added.png"
)
plt.savefig(dice_curve_path, dpi=300, bbox_inches="tight")
print(f"Saved Dice curve to: {dice_curve_path}")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(train_iou_history, label="Train IoU", marker="o")
plt.plot(val_iou_history, label="Validation IoU", marker="s")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Training vs Validation IoU")
plt.legend()
plt.grid(True)
iou_curve_path = (
    EXPERIMENT_REPORT_DIR / "sam_training_validation_iou_curve_metricies_added.png"
)
plt.savefig(iou_curve_path, dpi=300, bbox_inches="tight")
print(f"Saved IoU curve to: {iou_curve_path}")
plt.close()

# -------------------------------------------------------------------------
# 16. INFERENCE (single validation sample, using  config if available)
# -------------------------------------------------------------------------

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    print(f"\nLoaded config from: {CONFIG_PATH}")

    base_model_name = cfg.get("base_model_name", BASE_MODEL_NAME)
    checkpoint_path = Path(cfg.get("checkpoint_path", str(BEST_MODEL_PATH)))

    lora_cfg = cfg.get("lora", {})
    lora_r = lora_cfg.get("r", LORA_R)
    lora_alpha = lora_cfg.get("alpha", LORA_ALPHA)
    lora_dropout = lora_cfg.get("dropout", LORA_DROPOUT)

    prediction_threshold = cfg.get("prediction_threshold", PREDICTION_THRESHOLD)
else:
    print(f"\nConfig not found at {CONFIG_PATH}, using script defaults.")
    base_model_name = BASE_MODEL_NAME
    checkpoint_path = BEST_MODEL_PATH
    lora_r = LORA_R
    lora_alpha = LORA_ALPHA
    lora_dropout = LORA_DROPOUT
    prediction_threshold = PREDICTION_THRESHOLD

if not checkpoint_path.exists():
    print(f"Checkpoint not found at {checkpoint_path}. Skipping inference.")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device for inference:", device)

    base_sam_infer = SamModel.from_pretrained(base_model_name)
    apply_lora_to_sam(
        base_sam_infer,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    freeze_sam_except_lora(base_sam_infer)
    model_infer = base_sam_infer

    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_infer.load_state_dict(state_dict)
    model_infer.to(device)
    model_infer.eval()
    print("Loaded SAM with LoRA + fine-tuned mask_decoder\n")

    idx = random.randint(0, len(expanded_val) - 1)
    print(f"Inference on sample index: {idx}")

    sample_expanded = expanded_val[idx]
    test_image_display = np.array(sample_expanded["image"])
    ground_truth_mask = np.array(sample_expanded["mask"])
    prompt_box = sample_expanded["bbox"]

    sample_proc = val_dataset[idx]

    inputs_to_model = {
        "pixel_values": sample_proc["pixel_values"].unsqueeze(0).to(device),
        "input_boxes": sample_proc["input_boxes"].unsqueeze(0).to(device),
        "original_sizes": sample_proc["original_sizes"].unsqueeze(0).to(device),
        "reshaped_input_sizes": sample_proc["reshaped_input_sizes"]
        .unsqueeze(0)
        .to(device),
    }

    with torch.no_grad():
        outputs = model_infer(**inputs_to_model, multimask_output=False)

    medsam_seg_prob = (
        torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()
    )
    medsam_seg = (medsam_seg_prob > prediction_threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(test_image_display, cmap="gray")
    x_min, y_min, x_max, y_max = prompt_box
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title("Original Image + Prompt")
    axes[0].axis("off")

    axes[1].imshow(ground_truth_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(medsam_seg, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    im = axes[3].imshow(medsam_seg_prob, cmap="viridis")
    axes[3].set_title("Probability Map")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    inference_path = (
        EXPERIMENT_REPORT_DIR / "sam_inference_sample_metricies_added.png"
    )
    plt.savefig(inference_path, dpi=300, bbox_inches="tight")
    print(f"Saved inference figure to: {inference_path}")
    plt.close()
