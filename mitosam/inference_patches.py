
import os
import numpy as np
import torch
from PIL import Image
from tifffile import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import SamProcessor, SamModel
from peft import LoraConfig, get_peft_model, TaskType



MODEL_CKPT_PATH = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/models/sam_lora_best.pth"   # <-- Replace this
TEST_IMAGE_PATH = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/raw/Dataset/testing.tif"        # <-- Replace this
TEST_IMAGE_GT = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/data/raw/Dataset/testing_groundtruth.tif"


REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 256
OVERLAP = 32
THRESHOLD = 0.7
GRID_SIZE = 10


# === Helper Visualization Functions ===
def visualize_chunks_grid(chunks, coords, full_shape, save_path):
    layout = np.zeros(full_shape, dtype=np.float32)
    for patch, (y, x) in zip(chunks, coords):
        layout[y:y+patch.shape[0], x:x+patch.shape[1]] = patch
    plt.figure(figsize=(10, 10))
    plt.imshow(layout, cmap="gray")
    plt.title("Spatial Layout of Chunks")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_chunk_predictions(chunks, preds, probs, gts=None, save_path=None):
    N = len(chunks)
    fig, axes = plt.subplots(N, 4, figsize=(12, 3 * N))
    for i in range(N):
        for ax in axes[i]:
            ax.axis("off")
        axes[i][0].imshow(chunks[i], cmap="gray")
        axes[i][0].set_title("Input Patch")

        if gts:
            axes[i][1].imshow(gts[i], cmap="gray")
            axes[i][1].set_title("GT Mask")

        axes[i][2].imshow(preds[i], cmap="gray")
        axes[i][2].set_title("Predicted Mask")

        axes[i][3].imshow(probs[i], cmap="viridis")
        axes[i][3].set_title("Probability Map")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()

def visualize_full_result(image, merged_mask, prob_map, gt=None, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[1].imshow(merged_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    if gt is not None:
        axes[2].imshow(gt, cmap="gray")
        axes[2].set_title("Ground Truth")
    else:
        axes[2].axis("off")
    im = axes[3].imshow(prob_map, cmap="viridis")
    axes[3].set_title("Probability Map")
    fig.colorbar(im, ax=axes[3])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


# === Model + Inference Utilities ===
def load_trained_sam_peft(model_ckpt_path):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    base_model = SamModel.from_pretrained("facebook/sam-vit-base")

    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base_model, lora_config)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=DEVICE))
    return model.to(DEVICE).eval(), processor

def chunk_image_with_overlap(image, chunk_size=256, overlap=32):
    H, W = image.shape[:2]
    step = chunk_size - overlap
    y_starts = list(range(0, H - chunk_size + 1, step)) + ([H - chunk_size] if (H - chunk_size) % step else [])
    x_starts = list(range(0, W - chunk_size + 1, step)) + ([W - chunk_size] if (W - chunk_size) % step else [])
    coords = [(y, x) for y in y_starts for x in x_starts]
    chunks = [image[y:y+chunk_size, x:x+chunk_size] for (y, x) in coords]
    return chunks, coords

def make_grid_prompt(size=256, grid_size=10):
    lin = np.linspace(0, size-1, grid_size)
    xv, yv = np.meshgrid(lin, lin)
    pts = [[[int(x), int(y)] for x, y in zip(xv.ravel(), yv.ravel())]]
    return torch.tensor(pts).unsqueeze(0)

def predict_patch(patch, model, processor, input_points):
    inputs = processor(Image.fromarray(patch), input_points=input_points, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs, multimask_output=False)
    prob = torch.sigmoid(output.pred_masks.squeeze(1)).cpu().numpy().squeeze()
    return (prob > THRESHOLD).astype(np.uint8), prob

def stitch_chunks(chunks, coords, full_shape):
    final = np.zeros(full_shape, dtype=np.float32)
    weight = np.zeros(full_shape, dtype=np.float32)
    window = np.outer(np.hanning(chunks[0].shape[0]), np.hanning(chunks[0].shape[1]))
    for patch, (y, x) in zip(chunks, coords):
        h, w = patch.shape
        final[y:y+h, x:x+w] += patch * window[:h, :w]
        weight[y:y+h, x:x+w] += window[:h, :w]
    return final / (weight + 1e-8)

# === RUN INFERENCE PIPELINE ===
if __name__ == "__main__":
    model, processor = load_trained_sam_peft(MODEL_CKPT_PATH)
    image = imread(TEST_IMAGE_PATH)
    if image.ndim == 3: image = image[0]
    full_shape = image.shape

    gt = None
    if os.path.exists(TEST_IMAGE_GT):
        gt = imread(TEST_IMAGE_GT)
        if gt.ndim == 3: gt = gt[0]

    chunks, coords = chunk_image_with_overlap(image, CHUNK_SIZE, OVERLAP)
    visualize_chunks_grid(chunks, coords, full_shape, save_path=f"{REPORT_DIR}/chunks_layout_grid.png")

    prompt = make_grid_prompt(size=CHUNK_SIZE, grid_size=GRID_SIZE)
    preds, probs = [], []
    for chunk in chunks:
        m, p = predict_patch(chunk, model, processor, input_points=prompt)
        preds.append(m)
        probs.append(p)

    gt_chunks = [gt[y:y+CHUNK_SIZE, x:x+CHUNK_SIZE] for (y, x) in coords] if gt is not None else None
    visualize_chunk_predictions(chunks, preds, probs, gts=gt_chunks, save_path=f"{REPORT_DIR}/chunk_predictions_grid.png")

    stitched_mask = stitch_chunks(preds, coords, full_shape)
    stitched_prob = stitch_chunks(probs, coords, full_shape)

    visualize_full_result(image, stitched_mask, stitched_prob, gt=gt, save_path=f"{REPORT_DIR}/full_image_predictions_summary.png")
