import transformers
print(transformers.__version__)
import sys
print(sys.version)

from peft import TaskType
print(TaskType.__members__)
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
from peft import LoraConfig, get_peft_model, TaskType
from utils import plot_random_image_mask_pairs
from prompt_creator import build_sam_prompt_dataset, get_bounding_boxes, get_union_bounding_box, visualize_prompted_dataset


import numpy as np
import os

base_dir = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/processed"

# Define paths for both NPZ files
train_output_filename = os.path.join(base_dir, 'train_data_processed.npz')
val_output_filename   = os.path.join(base_dir, 'val_data_processed.npz')

# Load training data
try:
    with np.load(train_output_filename) as data:
        train_img = data['images']
        train_mask = data['masks']
    print(f"--- Loaded Training Data from {os.path.basename(train_output_filename)} ---")
    print(f"  'train_img' shape: {train_img.shape}, dtype: {train_img.dtype}")
    print(f"  'train_mask' shape: {train_mask.shape}, dtype: {train_mask.dtype}")
except FileNotFoundError:
    print(f"Error: The file {train_output_filename} was not found. Please ensure it exists.")
except Exception as e:
    print(f"An error occurred while loading the training NPZ file: {e}")

print("\n") # Add a newline for better readability

# Load validation data
try:
    with np.load(val_output_filename) as data:
        val_img = data['images']
        val_mask = data['masks']
    print(f"--- Loaded Validation Data from {os.path.basename(val_output_filename)} ---")
    print(f"  'val_img' shape: {val_img.shape}, dtype: {val_img.dtype}")
    print(f"  'val_mask' shape: {val_mask.shape}, dtype: {val_mask.dtype}")
except FileNotFoundError:
    print(f"Error: The file {val_output_filename} was not found. Please ensure it exists.")
except Exception as e:
    print(f"An error occurred while loading the validation NPZ file: {e}")










## keep this fromats. its just for debugging the whole workflow. 

train_img = train_img [:20]
train_mask = train_mask[:20]

val_img = val_img [:5]
val_mask = val_mask[:5]

print(f"  'train_img' shape: {train_img.shape}, dtype: {train_img.dtype}")
print(f"  'train_mask' shape: {train_mask.shape}, dtype: {train_mask.dtype}")

print(f"  'val_img' shape: {val_img.shape}, dtype: {val_img.dtype}")
print(f"  'val_mask' shape: {val_mask.shape}, dtype: {val_mask.dtype}")



## visualize the train images and make sure everything is loaded fine before traing. debugging steps 
fig, axes = plot_random_image_mask_pairs(train_img, train_mask, num_samples=5, seed=42)
# optionally save
fig.savefig("random_train_samples.png", dpi=300, bbox_inches="tight")

# -------------------------------------------------------------------------
## visualize the val images and make sure everything is loaded fine before traing. debugging steps
fig, axes = plot_random_image_mask_pairs(val_img, val_mask, num_samples=5, seed=24)
# optionally save
fig.savefig("random_val_samples.png", dpi=300, bbox_inches="tight")


# -------------------------------------------------------------------------
# Lets create the SAM data loaders now
# -------------------------------------------------------------------------'''


from datasets import Dataset
from PIL import Image

# Convert the NumPy arrays to Pillow images and store them in a dictionary
train_dataset_dict = {
    "image": [Image.fromarray(img) for img in train_img],
    "label": [Image.fromarray(mask) for mask in train_mask],
}

# Create the dataset using the datasets.Dataset class
train_dataset = Dataset.from_dict(train_dataset_dict)

# for val images

# Convert the NumPy arrays to Pillow images and store them in a dictionary
val_dataset_dict = {
    "image": [Image.fromarray(img) for img in val_img],
    "label": [Image.fromarray(mask) for mask in val_mask],
}

# Create the dataset using the datasets.Dataset class
val_dataset = Dataset.from_dict(val_dataset_dict)


# print dataset informations
print(f'info train dataset: {train_dataset}')
print(f'info val dataset: {val_dataset}')


## Lets gets the prompts
expanded_train = build_sam_prompt_dataset(
    images=train_dataset_dict["image"],
    masks=train_dataset_dict["label"],
    prompt_augment=False,        # changed from False to True to create one sample per bbox
    include_union_box=True,     # also add union box sample
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

# visualize the traing data with mask and prompt box
fig, axes = visualize_prompted_dataset(expanded_train, n_cols=5, start_idx=0)

# optionally save
fig.savefig("random_prompted_train_samples.png", dpi=300, bbox_inches="tight")

# visualize the val data with mask and prompt box
fig, axes = visualize_prompted_dataset(expanded_val, n_cols=5, start_idx=0)
# optionally save
fig.savefig("random_prompted_val_samples.png", dpi=300, bbox_inches="tight")

## lets start with defining the model

# put these imports in the SAME cell as the class, or above it
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image # Ensure Image is imported

class SAMDataset(Dataset):
    def __init__(self, expanded_data, processor):
        self.samples = expanded_data
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        image = entry["image"]            # PIL.Image
        mask  = np.array(entry["mask"])   # H×W
        box   = entry["bbox"]             # [x_min, y_min, x_max, y_max] in ORIGINAL coords

        # Convert single-channel image to RGB if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        enc = self.processor(image, input_boxes=[[box]], return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        enc["ground_truth_mask"] = torch.from_numpy(mask).float()  # H×W
        return enc

from transformers import SamProcessor
from torch.utils.data import DataLoader

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")



train_dataset = SAMDataset(expanded_train, processor)
val_dataset   = SAMDataset(expanded_val, processor)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
val_dataloader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, drop_last=False)

### print information regarding the dataset: 
enc0 = train_dataset[0]
print("pixel_values:", tuple(enc0["pixel_values"].shape))           # (3, 1024, 1024) typically
print("input_boxes:", tuple(enc0["input_boxes"].shape))             # (1, 4)
print("original_sizes:", enc0["original_sizes"].tolist())           # [H_orig, W_orig] ~ [256, 256]
print("reshaped_input_sizes:", enc0["reshaped_input_sizes"].tolist())  # [H_resized, W_resized] ~ [1024, 1024]
print("gt mask:", tuple(enc0["ground_truth_mask"].shape))           # (H_orig, W_orig) ~ (256, 256)


# lets visualize the image, mask and prompt from iside SAM processor to ake sure, its resizing and mapping properly. 

from sam_helper import visualize_sam_sample# From the dataset + index
fig, axes = visualize_sam_sample(train_dataset, idx=13 )

# --------------------------------------------------
# Define the PEFT and LoRA configs.
#---------------------------------------------------

import torch
from transformers import SamModel
from monai.losses import DiceFocalLoss
from peft import LoraConfig, get_peft_model, TaskType

device = "cuda" if torch.cuda.is_available() else "cpu"

# Base SAM model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# LoRA config applied to attention/projection layers across SAM
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # attention proj layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,  # vision / feature task
)

# IMPORTANT: wrap the WHOLE model, not model.mask_decoder
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()  # sanity check: % of trainable params

model.to(device)

# Segmentation loss
seg_loss = DiceFocalLoss(
    sigmoid=True,
    lambda_dice=1.0,
    lambda_focal=1.0,
    reduction="mean",
)



from tqdm import tqdm
from statistics import mean
import torch
import matplotlib.pyplot as plt

# ==========================
# OPTIMIZER, SCHEDULER, EARLY STOP
# ==========================

# Train only parameters that actually require gradients (LoRA adapters, etc.)
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,   # Higher LR is fine for PEFT/LoRA
)

# Reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,      # LR = LR * 0.5
    patience=3,      # epochs with no improvement before LR drop
    threshold=0.001, # ignore improvements smaller than this
    threshold_mode="abs",
)

early_stopping_patience = 7        # stop if no improvement for 7 epochs
early_stopping_min_delta = 0.001   # need at least this much improvement in val loss

best_val_loss = float("inf")
epochs_without_improvement = 0

best_model_path = (
    "/content/drive/MyDrive/"
    "Electron_Microscope_Practice_Projects/"
    "Mitochondria_segmentation/best_sam_model.pth"
)

# ==========================
# TRAINING LOOP
# ==========================

num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # ---- TRAIN ----
    model.train()
    epoch_train_losses = []

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)  # (B, H, W)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False,
        )

        predicted_masks = outputs.pred_masks.squeeze(1)  # (B, H_pred, W_pred)

        # Resize GT masks to match predicted mask size if needed
        if ground_truth_masks.shape[-2:] != predicted_masks.shape[-2:]:
            ground_truth_masks = torch.nn.functional.interpolate(
                ground_truth_masks.unsqueeze(1),          # (B, 1, H, W)
                size=predicted_masks.shape[-2:],          # (H_pred, W_pred)
                mode="nearest",
            ).squeeze(1)                                  # (B, H_pred, W_pred)

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
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )

            predicted_masks = outputs.pred_masks.squeeze(1)

            # Match spatial dims
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

    mean_val_loss = mean(epoch_val_losses)
    val_losses.append(mean_val_loss)

    # ---- LR SCHEDULER: step on validation loss ----
    scheduler.step(mean_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {mean_train_loss:.4f} | "
        f"Val Loss: {mean_val_loss:.6f} | "
        f"LR: {current_lr:.2e}"
    )

    # ---- EARLY STOPPING ----
    if mean_val_loss < best_val_loss - early_stopping_min_delta:
        best_val_loss = mean_val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), best_model_path)
        print(f"  ✅ New best val loss: {best_val_loss:.6f}. Model saved.")
    else:
        epochs_without_improvement += 1
        print(f"  No meaningful improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print("⏹ Early stopping triggered.")
            break

# Optionally reload best checkpoint
model.load_state_dict(torch.load(best_model_path))

# ==========================
# PLOT LOSSES
# ==========================
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (DiceFocal) with LR scheduling & early stopping")
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()
