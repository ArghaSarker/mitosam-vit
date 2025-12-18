import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PolyCollection

root = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/reports"
out_dir = "/share/klab/argha/SAM_mitochondria/MitoSAM-ViT/reports/figures"
os.makedirs(out_dir, exist_ok=True)

paths = {
    "bbox": {
        "fine_tuned": os.path.join(root, "SAM_ViT_Peft_rank64", "inference_metrics_bbox_prompt.csv"),
        "base":      os.path.join(root, "BASE_SAM_BBOX", "inference_metrics_BASE_SAM_BBOX.csv"),
    },
    "point 2x2": {
        "fine_tuned": os.path.join(root, "SAM_ViT_Peft_rank64", "inference_metrics_points_grid_2x2.csv"),
        "base":      os.path.join(root, "BASE_SAM_POINTS_GRID", "inference_metrics_BASE_SAM_points_grid_2x2.csv"),
    },
    "point 4x4": {
        "fine_tuned": os.path.join(root, "SAM_ViT_Peft_rank64", "inference_metrics_points_grid_4x4.csv"),
        "base":      os.path.join(root, "BASE_SAM_POINTS_GRID", "inference_metrics_BASE_SAM_points_grid_4x4.csv"),
    },
    "point 8x8": {
        "fine_tuned": os.path.join(root, "SAM_ViT_Peft_rank64", "inference_metrics_points_grid_8x8.csv"),
        "base":      os.path.join(root, "BASE_SAM_POINTS_GRID", "inference_metrics_BASE_SAM_points_grid_8x8.csv"),
    },
}

order = list(paths.keys())
metrics = ["dice", "iou"]
yt01 = np.linspace(0, 1, 11)
min_val = 0.6

def mean_dice_iou(p):
    df = pd.read_csv(p, usecols=metrics)
    return df.mean(numeric_only=True)

rows = []
for prompt in order:
    for model in ["base", "fine_tuned"]:
        p = paths[prompt][model]
        if not os.path.exists(p):
            raise SystemExit(f"missing: {prompt} {model} -> {p}")
        m = mean_dice_iou(p)
        rows.append({"prompt": prompt, "model": model, "dice": float(m["dice"]), "iou": float(m["iou"])})

means = pd.DataFrame(rows)
means.to_csv(os.path.join(out_dir, "dice_iou_means_promptwise.csv"), index=False)

wide = means.pivot(index="prompt", columns="model", values=metrics).reindex(order)
x = np.arange(len(order))
w = 0.38

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
for ax, metric in zip(axes, metrics):
    b = wide[(metric, "base")].to_numpy()
    f = wide[(metric, "fine_tuned")].to_numpy()
    ax.bar(x - w/2, b, width=w, label="base")
    ax.bar(x + w/2, f, width=w, label="fine_tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title(metric.upper())
    ax.set_ylim(0, 1)
    ax.set_yticks(yt01)
    ax.set_yticklabels([f"{v:.1f}" for v in yt01])

axes[0].set_ylabel("mean")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "dice_iou_base_vs_finetuned_by_prompt.png"), dpi=250)
plt.close()

ft_frames = []
for prompt in order:
    p = paths[prompt]["fine_tuned"]
    df = pd.read_csv(p, usecols=metrics)
    df["prompt"] = prompt
    ft_frames.append(df)

ft = pd.concat(ft_frames, ignore_index=True)
ft_long = ft.melt(id_vars=["prompt"], value_vars=metrics, var_name="metric", value_name="value")
ft_long = ft_long[ft_long["value"] >= min_val].copy()

palette = dict(zip(order, sns.color_palette("tab10", n_colors=len(order))))
yt = np.linspace(min_val, 1.0, int(round((1.0 - min_val) / 0.05)) + 1)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
for ax, metric in zip(axes, metrics):
    d = ft_long[ft_long["metric"] == metric]
    sns.violinplot(
        data=d, x="prompt", y="value",
        order=order, cut=0, inner="box",
        palette=palette, ax=ax, linewidth=1
    )
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(0.45)
    ax.set_title(metric.upper())
    ax.set_xlabel("")
    ax.set_ylabel("value" if ax is axes[0] else "")
    ax.set_ylim(min_val, 1.0)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.2f}" for v in yt])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"dice_iou_violin_box_finetuned_only_ge_{min_val}.png"), dpi=250)
plt.close()

print("saved:", os.path.join(out_dir, "dice_iou_means_promptwise.csv"))
print("saved:", os.path.join(out_dir, "dice_iou_base_vs_finetuned_by_prompt.png"))
print("saved:", os.path.join(out_dir, f"dice_iou_violin_box_finetuned_only_ge_{min_val}.png"))
