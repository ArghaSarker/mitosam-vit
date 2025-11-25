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


