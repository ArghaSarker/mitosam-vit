import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
from csbdeep.data import create_patches
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

import cv2

print('all library loaded successfully')


# -------------------------------------------------------------------------
#              Load the Dataset. (in .tif format)
# -------------------------------------------------------------------------


# #165 large images as tiff image stack
large_images = tifffile.imread("/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/raw/Dataset/training.tif")
large_masks = tifffile.imread("/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/raw/Dataset/training_groundtruth.tif")

# lets take a small sample:
large_images = large_images[:40]
large_masks = large_masks[:40]

print('dtype' , large_images.dtype , 'shape' , large_images.shape, 'min' , large_images.min(), 'max',  large_images.max())




image_ids = list(range(len(large_images)))  # or use file name list
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42, shuffle=True)

print("train images:", len(train_ids), "val images:", len(val_ids))
print('train ids' , train_ids)
print('val ids' , val_ids)

# Extract training and validation images/masks based on IDs
train_images = large_images[train_ids]
train_masks  = large_masks[train_ids]

val_images = large_images[val_ids]
val_masks  = large_masks[val_ids]

# Print summaries
print(f"Train images: {train_images.shape}, dtype={train_images.dtype}")
print(f"Train masks:  {train_masks.shape}, dtype={train_masks.dtype}")
print(f"Val images:   {val_images.shape}, dtype={val_images.dtype}")
print(f"Val masks:    {val_masks.shape}, dtype={val_masks.dtype}")

# Quick sanity check — ensure shapes match
assert train_images.shape == train_masks.shape, "Train image/mask mismatch!"
assert val_images.shape == val_masks.shape, "Val image/mask mismatch!"



# -------------------------------------------------------------------------
# Create Patches from Large Images amd Filter the Pacthes based on Mask ratio
# -------------------------------------------------------------------------
def save_figure_png(fig, name, out_dir="report/figures", overwrite=False, dpi=300, bbox_inches='tight'):
    """
    Save a matplotlib Figure as a PNG.

    Args:
        fig: matplotlib.figure.Figure to save.
        name: filename or basename for the PNG (with or without .png).
        out_dir: directory to save into (default "report/figures").
        overwrite: if True, overwrite existing file; if False, create a unique name.
        dpi: resolution for savefig.
        bbox_inches: bbox_inches passed to savefig.
    Returns:
        out_path: full path to the saved PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(name)
    if ext.lower() != ".png":
        name = base + ".png"

    out_path = os.path.join(out_dir, name)

    if os.path.exists(out_path) and not overwrite:
        i = 1
        while True:
            candidate = os.path.join(out_dir, f"{base}_{i}.png")
            if not os.path.exists(candidate):
                out_path = candidate
                break
            i += 1

    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved augmented patches figure to: {out_path}")
    return out_path


def _get_starts(L, patch_size, step):
    """
    Compute valid starting indices along one dimension (0..L-1)
    so that all patches are fully inside the image, and the last
    patch touches the border. This may create extra overlap at the end.
    """
    # If image is smaller than a patch, just start at 0
    if L <= patch_size:
        return [0]

    # Regular starts with fixed step, staying inside bounds
    starts = list(range(0, L - patch_size + 1, step))

    # Ensure we also have a patch that ends exactly at L
    last_start = L - patch_size
    if last_start not in starts:
        starts.append(last_start)

    return sorted(starts)


def create_patches(image, mask, patch_size=256, overlap=32):
    """
    Divide image & mask into overlapping patches WITHOUT padding.
    Returns:
      image_patches: list of (patch_size, patch_size) arrays
      mask_patches:  list of (patch_size, patch_size) arrays
    """
    step = patch_size - overlap

    # Assume 2D grayscale; if image is 3D (H,W,C), use image.shape[:2]
    H, W = image.shape

    y_starts = _get_starts(H, patch_size, step)
    x_starts = _get_starts(W, patch_size, step)

    img_list, mask_list = [], []

    for y in y_starts:
        for x in x_starts:
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            img_list.append(img_patch)
            mask_list.append(mask_patch)

    return img_list, mask_list


def filter_patches(image_patches, mask_patches, min_coverage=0.005):
    """
    Filters patches based on mask coverage ratio.
    Keeps only patches where the mask has at least 'min_coverage' fraction of foreground pixels.

    Args:
        image_patches (list of np.array): Image patches
        mask_patches (list of np.array): Corresponding mask patches
        min_coverage (float): Minimum fraction of nonzero pixels required to keep patch

    Returns:
        kept_images (list)
        kept_masks (list)
    """
    kept_imgs, kept_msks = [], []
    for img, msk in zip(image_patches, mask_patches):
        ratio = np.count_nonzero(msk) / msk.size
        if ratio >= min_coverage:
            kept_imgs.append(img)
            kept_msks.append(msk)
    return kept_imgs, kept_msks

# -------------------------------------------------------------------------
# Create training dataset
# -------------------------------------------------------------------------
train_img_patches = []
train_mask_patches = []

for img, msk in zip(train_images, train_masks):
    img_patches, mask_patches = create_patches(img, msk, patch_size=256, overlap=64)
    train_img_patches.extend(img_patches)
    train_mask_patches.extend(mask_patches)

print(f"Total patches before filtering: {len(train_img_patches)}")

# ✅ Filter all patches in one go
train_filtered_imgs, train_filtered_msks = filter_patches(
    train_img_patches,
    train_mask_patches,
    min_coverage=0.009
)

print(f"After filtering: {len(train_filtered_imgs)} kept")


# -------------------------------------------------------------------------
# Create validation dataset
# -------------------------------------------------------------------------

print(val_images.shape, val_masks.shape)

# Collect patches from ALL validation images
val_img_patches = []
val_mask_patches = []

for img, msk in zip(val_images, val_masks):
    img_patches, mask_patches = create_patches(
        img,
        msk,
        patch_size=256,
        overlap=32
    )
    val_img_patches.extend(img_patches)
    val_mask_patches.extend(mask_patches)

print(f"Total patches before filtering: {len(val_img_patches)}")

# Filter patches based on mask coverage
val_filtered_imgs, val_filtered_msks = filter_patches(
    val_img_patches,
    val_mask_patches,
    min_coverage=0.009
)

print(f"After filtering: {len(val_filtered_imgs)} kept")

# Combined visualization: training (top two rows) and validation (bottom two rows)
n_show = 3
n_train = len(train_filtered_imgs)
n_val = len(val_filtered_imgs)
if n_train == 0 and n_val == 0:
    print("No patches to display.")
else:
    # Determine number of columns to show (at least 1)
    n_cols = max(1, min(n_show, max(n_train, n_val)))

    # Safe sampling (if not enough examples, sample with replacement)
    def safe_sample(n, k):
        if n == 0:
            return [None] * k
        if n >= k:
            return random.sample(range(n), k)
        else:
            return [random.choice(range(n)) for _ in range(k)]

    train_idxs = safe_sample(n_train, n_cols)
    val_idxs = safe_sample(n_val, n_cols)

    fig, axes = plt.subplots(4, n_cols, figsize=(2 * n_cols, 8))
    # Ensure axes is 2D
    if n_cols == 1:
        axes = axes.reshape(4, 1)

    for c in range(n_cols):
        t_idx = train_idxs[c]
        v_idx = val_idxs[c]

        # Train image (row 0) and mask (row 1)
        if t_idx is not None:
            axes[0, c].imshow(train_filtered_imgs[t_idx], cmap='gray')
            axes[0, c].set_title(f"Train img {t_idx}")
            axes[1, c].imshow(train_filtered_msks[t_idx], cmap='gray')
            axes[1, c].set_title("Train Mask")
        else:
            axes[0, c].axis('off')
            axes[1, c].axis('off')

        # Val image (row 2) and mask (row 3)
        if v_idx is not None:
            axes[2, c].imshow(val_filtered_imgs[v_idx], cmap='gray')
            axes[2, c].set_title(f"Val image {v_idx}")
            axes[3, c].imshow(val_filtered_msks[v_idx], cmap='gray')
            axes[3, c].set_title("Val Mask")
        else:
            axes[2, c].axis('off')
            axes[3, c].axis('off')

        # Turn off axes ticks for all shown subplots
        for r in range(4):
            axes[r, c].axis('off')

    fig.suptitle("Sample_patches")
    plt.tight_layout()
    plt.show()





# -------------------------------------------------------------------------
# Augment the dataset
# -------------------------------------------------------------------------

# ## Now prepare Augmentation

# - HorizontalFlip
# - VerticalFlip
# - RandomRotate90
# - ShiftScaleRotate
# - ElasticTransform
# - GaussNoise
# - GaussianBlur
# - PoisionNoise
# - zoom in
# - zoom out (fill with repreatuion)
# - random crop
# - random brightness


import albumentations as A
import numpy as np # Ensure numpy is imported
import cv2 # Import OpenCV for border_mode

def augment_data(image, mask):
    """
    Applies a series of albumentations transformations to an image and its corresponding mask,
    returning the original image/mask pair and three additional augmented pairs.
    Geometric transformations are applied to both image and mask.
    Non-geometric transformations are applied only to the image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The input mask.

    Returns:
        list: A list of four tuples, where each tuple contains (transformed_image, transformed_mask).
              The first tuple is the original (image, mask).
    """

    geometric_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT), # Added border_mode
        A.ElasticTransform(alpha=1, sigma=20, p=0.5), # Removed alpha_affine as it's not a valid argument
        # A.RandomCrop(height=256, width=256, p=0.5), # Included RandomCrop as requested
    ])

    non_geometric_transform = A.Compose([
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(1,2), p=0.2),
        # A.PoissonNoise(p=0.5), # Removed PoissonNoise due to AttributeError
        A.RandomBrightnessContrast(p=0.2)
    ])

    augmented_data_list = [(image, mask)] # Start with the original image and mask

    for _ in range(3): # Generate 3 additional augmented samples
        # Apply geometric transformations to both image and mask
        geometric_augmented = geometric_transform(image=image, mask=mask)
        geo_image = geometric_augmented['image']
        geo_mask = geometric_augmented['mask']

        # Apply non-geometric transformations ONLY to the geometrically augmented image
        # The mask remains unchanged by these transformations
        non_geometric_augmented_image = non_geometric_transform(image=geo_image)['image']

        augmented_data_list.append((non_geometric_augmented_image, geo_mask))

    return augmented_data_list

print("Augmentation function 'augment_data' modified to return original and 3 augmented image-mask pairs with selective transformations.")




# Pick a sample image and mask from the filtered training set
# Using the first available patch for demonstration
sample_image = train_filtered_imgs[0]
sample_mask = train_filtered_msks[0]

# Augment the data
augmented_samples = augment_data(sample_image, sample_mask)

# Visualize the original and augmented samples
# Create subplots with 2 rows (for image and mask) and N columns (for each sample)
fig, axes = plt.subplots(2, len(augmented_samples), figsize=(2 * len(augmented_samples), 6))

for i, (img, msk) in enumerate(augmented_samples):
    title_prefix = "Original" if i == 0 else f"Augmented {i}"

    # Top row: images
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f"{title_prefix} Image")
    axes[0, i].axis('off')

    # Bottom row: masks
    axes[1, i].imshow(msk, cmap='gray')
    axes[1, i].set_title(f"{title_prefix} Mask")
    axes[1, i].axis('off')

fig.suptitle("Original and Augmented Samples", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
plt.show()

### Apply augmentation to tarining patches only. Valdiation doesnt need augmenation. 

train_augmented_imgs = []
train_augmented_msks = []

# val_augmented_imgs = []
# val_augmented_msks = []

# val_augmented_imgs = val_filtered_imgs
# val_augmented_msks = val_filtered_msks

# Augment training data
for img, msk in zip(train_filtered_imgs, train_filtered_msks):
    augmented_pairs = augment_data(img, msk)
    for a_img, a_msk in augmented_pairs:
        train_augmented_imgs.append(a_img)
        train_augmented_msks.append(a_msk)

# # Augment validation data
# for img, msk in zip(val_filtered_imgs, val_filtered_msks):
#     augmented_pairs = augment_data(img, msk)
#     for a_img, a_msk in augmented_pairs:
#         val_augmented_imgs.append(a_img)
#         val_augmented_msks.append(a_msk)

print(f"Original train patches: {len(train_filtered_imgs)}, Augmented train samples: {len(train_augmented_imgs)}")
# print(f"Original val patches: {len(val_filtered_imgs)}, Augmented val samples: {len(val_augmented_imgs)}")


# Number of augmented samples to display
n_show = 4
# 
print(f"Displaying {n_show} augmented training samples...")
idxs = random.sample(range(len(train_augmented_imgs)), n_show)

# 2 rows: top = images, bottom = masks
fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 6))

for col, idx in enumerate(idxs):
    img = train_augmented_imgs[idx]
    msk = train_augmented_msks[idx]

    # Top row: images
    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(f"Image {idx}")
    axes[0, col].axis('off')

    # Bottom row: masks
    axes[1, col].imshow(msk, cmap='gray')
    axes[1, col].set_title(f"Mask {idx}")
    axes[1, col].axis('off')



# Use the helper to save the current figure (uses default path unless overriden)
fname = f"augmented_patches_sample_{len(train_augmented_imgs)}_samples.png"
out_path = save_figure_png(fig, fname)  # pass overwrite=True if you want to replace existing file
plt.show()

def percentile_normalize(image, mask, pmin=1.0, pmax=99.0, out_range=(0.0, 1.0), eps=1e-8):
    """
    Percentile-based normalization for an image while returning the mask unchanged.

    Args:
        image (np.ndarray): Input image (H,W) or (H,W,C).
        mask (np.ndarray): Corresponding mask (returned unchanged).
        pmin (float): Lower percentile (0-100).
        pmax (float): Upper percentile (0-100).
        out_range (tuple): Desired output value range (min, max).
        eps (float): Small value to avoid division by zero.

    Returns:
        norm_image (np.ndarray): Normalized image as float32 in out_range.
        mask (np.ndarray): Unmodified mask (same object forwarded).
    """
    img = np.asarray(image).astype(np.float32)

    # Compute percentile bounds
    lo = np.percentile(img, pmin)
    hi = np.percentile(img, pmax)

    # Fallback to min/max if percentiles are degenerate
    if hi - lo < eps:
        lo = float(img.min())
        hi = float(img.max())

    # Avoid zero-range
    if hi - lo < eps:
        # image is essentially constant; return a constant array in the middle of out_range
        mid = 0.5 * (out_range[0] + out_range[1])
        norm = np.full_like(img, fill_value=mid, dtype=np.float32)
        return norm, mask

    # Clip to percentile range and scale to [0,1]
    clipped = np.clip(img, lo, hi)
    norm01 = (clipped - lo) / (hi - lo)

    # Map to desired output range
    out_min, out_max = out_range
    norm = norm01 * (out_max - out_min) + out_min

    return norm.astype(np.float32), mask



train_data_norm_aug, train_mask_norm_au = percentile_normalize(train_augmented_imgs, train_augmented_msks)
# val data is not augmented
val_data_norm_aug, val_mask_norm_au = percentile_normalize(val_filtered_imgs,val_filtered_msks)


print("Data normalization applied and new variables created:")
print(f"  train_data_nor_aug: {len(train_data_norm_aug)} images, dtype: {train_data_norm_aug[0].dtype}")
print(f"  train_mask_norm_au: {len(train_mask_norm_au)} masks, dtype: {train_mask_norm_au[0].dtype}")
print(f"  val_data_nor_aug: {len(val_data_norm_aug)} images, dtype: {val_data_norm_aug[0].dtype}")
print(f"  val_mask_norm_au: {len(val_mask_norm_au)} masks, dtype: {val_mask_norm_au[0].dtype}")


# Select a sample image before and after normalization
sample_idx = 0 # Using the first image for demonstration

original_augmented_image = train_augmented_imgs[sample_idx]
original_augmented_mask = train_augmented_msks[sample_idx]
normalized_image = train_data_norm_aug[sample_idx]
normalized_mask = train_mask_norm_au[sample_idx] # Mask is not normalized, just passed through

print(f"--- Data Differences Before and After Normalization (Sample Index: {sample_idx}) ---")

print("\nOriginal Augmented Image:")
print(f"  Data Type: {original_augmented_image.dtype}")
print(f"  Min Value: {original_augmented_image.min()}")
print(f"  Max Value: {original_augmented_image.max()}")
print(f"  Mean Value: {original_augmented_image.mean():.4f}")

print("\nOriginal Augmented Mask:")
print(f"  Data Type: {original_augmented_mask.dtype}")
print(f"  Min Value: {original_augmented_mask.min()}")
print(f"  Max Value: {original_augmented_mask.max()}")
print(f"  Mean Value: {original_augmented_mask.mean():.4f}")
print("  (Note: Masks are typically binary and not normalized like images)")

print("\nNormalized Image:")
print(f"  Data Type: {normalized_image.dtype}")
print(f"  Min Value: {normalized_image.min():.4f}")
print(f"  Max Value: {normalized_image.max():.4f}")
print(f"  Mean Value: {normalized_image.mean():.4f}")

print("\nNormalized Mask (unchanged from original augmented mask):")
print(f"  Data Type: {normalized_mask.dtype}")
print(f"  Min Value: {normalized_mask.min()}")
print(f"  Max Value: {normalized_mask.max()}")
print(f"  Mean Value: {normalized_mask.mean():.4f}")

# Visualize them side-by-side to see visual impact
fig, axes = plt.subplots(1, 4, figsize=(18, 5)) # Changed to 4 columns

axes[0].imshow(original_augmented_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(original_augmented_mask, cmap='gray')
axes[1].set_title('Original Mask')
axes[1].axis('off')

axes[2].imshow(normalized_image, cmap='gray')
axes[2].set_title('Normalized Image')
axes[2].axis('off')

axes[3].imshow(normalized_mask, cmap='gray')
axes[3].set_title('Normalized Mask (Unchanged)')
axes[3].axis('off')

plt.tight_layout()
plt.show()





#-------------------------------------------------------------------------
# Save the processed data for later use
#-------------------------------------------------------------------------





# Define the base directory where the original data was loaded from
# This assumes '/content/drive/MyDrive/Electron_Microscope_Practice_Projects/Mitochondria_segmentation/'
# is the correct path where the original .tif files reside.
base_dir = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/processed/"

# Convert lists of images/masks to single NumPy arrays for saving
# Stack them to create a 4D array if they are 2D patches
# For images, stack them directly
train_images_to_save = np.array(train_data_norm_aug)
train_masks_to_save  = np.array(train_mask_norm_au)

val_images_to_save = np.array(val_data_norm_aug)
val_masks_to_save  = np.array(val_mask_norm_au)

# Define filenames for saving
train_output_filename = os.path.join(base_dir, 'train_data_processed.npz')
val_output_filename   = os.path.join(base_dir, 'val_data_processed.npz')

# Save the training data
np.savez_compressed(train_output_filename,
                    images=train_images_to_save,
                    masks=train_masks_to_save)
print(f"Training data saved to: {train_output_filename}")
print(f"  Images shape: {train_images_to_save.shape}, Masks shape: {train_masks_to_save.shape}")

# Save the validation data
np.savez_compressed(val_output_filename,
                    images=val_images_to_save,
                    masks=val_masks_to_save)
print(f"Validation data saved to: {val_output_filename}")
print(f"  Images shape: {val_images_to_save.shape}, Masks shape: {val_masks_to_save.shape}")

print("Data saved successfully for later use.")
