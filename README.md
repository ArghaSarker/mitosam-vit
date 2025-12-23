# 🧬 MitoSAM-ViT

Parameter-Efficient Fine-Tuning (PEFT) of **Segment Anything (SAM)** for **mitochondria segmentation** in 2D EM using **LoRA**.  
This repo compares **fine-tuned vs. vanilla SAM** and studies how **prompt type** (bbox vs. point grids) affects segmentation quality.

---

## Project

### Segment Anything Model (SAM)
SAM is a vision transformer–based, promptable segmentation model developed by Meta in 2023.

The model has two main components:

- **Image encoder (ViT)**: converts an input image into a latent embedding space.
- **Prompt encoder + mask decoder**: encodes prompts (points/boxes) and decodes masks conditioned on the image embedding.

Model used: **SAM ViT-B** (~91M parameters), pretrained on **SA-1B** (≈11M images, 1.1B masks).

<img src="reports/figures/sam_paper.png" width="900">

*Figure from Kirillov et al., "Segment Anything" (2023)*

---

### What was fine-tuned (and why)

Since SAM is a foundational model trained on the SA-1B dataset, it generalizes well to many segmentation tasks in natural images. However, it performs poorly on specialized tasks (e.g., cell segmentation, crack segmentation) that it has not seen during training.

To use this model for a specialized domain and achieve strong accuracy, it needs to be fine-tuned on domain-specific data. This introduces two main challenges: collecting a sufficiently large, training-eligible dataset and having enough hardware to train a large model. Even the smaller SAM variant has ~91 million parameters.

For this reason, to fine-tune on a small dataset and reduce GPU requirements, **parameter-efficient fine-tuning (PEFT)** methods such as **LoRA** make a lot of sense. Compared to full fine-tuning, LoRA typically causes only a small drop in accuracy while being much more efficient in compute and data requirements. In practice, it allows you to reuse SAM’s pretrained knowledge and learn domain-specific behavior on top of it.

---

### Fine-tuning strategy

In this experiment, I applied **LoRA** to the **ViT image encoder** and **fully fine-tuned the mask decoder**, while keeping the prompt encoder and the original image encoder weights frozen.


<img src="reports/figures/sam_Lora_finetuning.png" width="900">




While there are methods that apply LoRA to both the image encoder and the mask decoder to further reduce trainable parameters (e.g., CellSeg1; Zhou et al., 2024), or approaches that fine-tune only the mask decoder, I chose this strategy for the following reasons:

- **Image encoder**: This is the part of the architecture with most of the parameters. It encodes image understanding into a latent embedding space. It makes sense to reuse the pretrained knowledge and adapt it with **LoRA**, so only a small fraction of parameters are updated for my domain-specific dataset (roughly **6%** in this experiment).
- **Mask decoder**: I **fully fine-tuned** the final mask decoding component (≈4M parameters) to improve prompt-to-mask decoding for the mitochondria domain.

---

### Project outline

- Fine-tune SAM for mitochondria segmentation with LoRA.
- Quantify how **prompt strategies** (bbox vs. point grids) change performance.
- Analyze model predictions using **Integrated Gradients / saliency** and **Occlusion Sensitivity**.

---

## Training curves

<table>
  <tr>
    <td align="center">
      <b>Train vs validation loss</b><br>
      <img src="reports/SAM_ViT_Peft_rank64/sam_training_validation_loss_curve_metricies_added.png" width="440">
    </td>
    <td align="center">
      <b>Learning rate schedule</b><br>
      <img src="reports/SAM_ViT_Peft_rank64/sam_learning_rate_curve_metricies_added.png" width="440">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Train vs validation Dice</b><br>
      <img src="reports/SAM_ViT_Peft_rank64/sam_training_validation_dice_curve_metricies_added.png" width="440">
    </td>
    <td align="center">
      <b>Train vs validation IoU</b><br>
      <img src="reports/SAM_ViT_Peft_rank64/sam_training_validation_iou_curve_metricies_added.png" width="440">
    </td>
  </tr>
</table>

---

## Training details

- **Hardware**: NVIDIA H100 (80GB)
- **Training length**: 21 epochs
- **Input tiling**: 256×256 patches with 32 px overlap (to reduce boundary artifacts and improve stitching smoothness)
- **Augmentations**: random rotations, flips, brightness changes, elastic deformations, and random shifts
- **Optimizer**: Adam
- **Learning rate**: 1e-3 with a learning-rate scheduler
- **Loss**: Dice-Focal (0.8 Dice, 0.2 Focal)

**Prompt setup**
- **Box prompts**: derived from ground-truth bounding boxes
- **Point prompts**: uniform point grids evenly distributed over the input image (2×2 / 4×4 / 8×8)

**LoRA setup**
- **Rank**: 64
- **Dropout**: 0.1

---

## Results

### Qualitative visualizations

**Fine-tuned SAM**
- **BBox prompt (derived from GT)**  
  <img src="reports/SAM_ViT_Peft_rank64/inference_visualisations/val_sample_013.png" width="800">

- **Point grid 8×8**  
  <img src="reports/SAM_ViT_Peft_rank64/inference_visualisations/val_sample_013_points_grid_8x8.png" width="800">

- **Point grid 4×4**  
  <img src="reports/SAM_ViT_Peft_rank64/inference_visualisations/val_sample_013_points_grid_4x4.png" width="800">

- **Point grid 2×2**  
  <img src="reports/SAM_ViT_Peft_rank64/inference_visualisations/val_sample_013_points_grid_2x2.png" width="800">

**Vanilla SAM**
- **BBox prompt (derived from GT)**  
  <img src="reports/BASE_SAM_BBOX/inference_visualisations/val_sample_013_BASE_SAM_BBOX.png" width="800">

- **Point grid 4×4**  
  <img src="reports/BASE_SAM_POINTS_GRID/inference_visualisations/val_sample_013_BASE_SAM_points_grid_4x4.png" width="800">

- **Point grid 2×2**  
  <img src="reports/BASE_SAM_POINTS_GRID/inference_visualisations/val_sample_013_BASE_SAM_points_grid_2x2.png" width="800">

---

### TrustifAI: Can we trust the output?

To observe model behavior and understand predictions, I relied on two well-known XAI methods:

- **Saliency / Integrated Gradients (IG)**: gradient-based attributions that highlight pixels where changing the input would most change the model’s output (for a chosen target). IG is typically more stable than raw gradients because it integrates gradients along a path from a baseline image.
- **Occlusion sensitivity**: occludes (masks) small patches of the image and measures how much the prediction quality changes (e.g., Dice/IoU or mask confidence). Regions where occlusion hurts performance most are interpreted as “evidence” the model relied on.

Based on these maps, I observed that the fine-tuned model (with bounding-box prompts) shows stronger gradient activation on the object and localizes better, suggesting that the model learned mitochondria-specific features.

**Note:** Outputs from Occlusion Sensitivity and Integrated Gradients were normalized for visualization purposes.

---

### Quantitative comparison (Dice / IoU)

- **Prompt-wise mean Dice/IoU (base vs fine-tuned)**  
  <img src="reports/figures/dice_iou_base_vs_finetuned_by_prompt.png" width="900">

- **Score distribution (fine-tuned only)**  
  <img src="reports/figures/dice_iou_violin_box_finetuned_only_ge_0.6.png" width="900">

  Note: Values are filtered with a threshold of 0.6 to focus on the distribution and readability.

Summary on this dataset:
- **Fine-tuned SAM** is consistently strong across prompt types (Dice/IoU).
- **Vanilla SAM** is substantially weaker for **point-grid prompts**; bbox prompting is relatively better but still below the fine-tuned model.

---

### Findings

- For **fine-tuned SAM**, **bbox** and **denser point grids (4×4 / 8×8)** produce similar-looking masks; **2×2** can underperform because prompts are too sparse to constrain the object well. However, the violin plot shows that point prompts have a wider distribution, suggesting cases where they can underperform.
- For **vanilla SAM**, point-grid prompts can fail (often producing overly large or poorly localized masks), suggesting the pretrained model is not well aligned with mitochondria EM appearance without adaptation (i.e., it lacks domain-specific knowledge).
- When predictions collapse to trivial masks (too large / too small), attribution maps can become less informative; they may reflect model instability rather than meaningful localization.

---

## Conclusion

SAM is a powerful model that has been proven on many general tasks. Using LoRA to adapt the **image encoder** and fully tuning the **mask decoder** yields a SAM variant that segments mitochondria reliably in EM images and is more robust to prompt choice than vanilla SAM.

Using LoRA also makes training much more efficient: in my setup, only ~**6.1%** of the model parameters were trainable, which substantially reduces compute and makes adaptation feasible on **consumer-grade GPUs**.

---

## Discussion

- **Context collapsing**: Fine-tuning yields a large improvement in accuracy and reduces **prompt sensitivity**: after adaptation, bbox and point-grid prompts yield similar overall performance, suggesting the model learned mitochondria-specific features and decoding behavior. However, I observed a wider distribution with larger point-prompt grids. Could this suggest context collapsing in this scenario?
- **Can point prompts perform as well as bbox prompts** if placed where the object exists? Possibly. This raises the question of whether prompt location matters (e.g., centered on the object vs. near the boundary).
- **TrustifAI**: From Integrated Gradients and saliency maps, can we trust the segmentation blindly on unseen data without a human in the loop in sensitive domains (e.g., medical diagnosis)?

---

## Future work

- Add **YOLO** mitochondria detection to generate bbox prompts automatically.
- Build an end-to-end pipeline: **YOLO bbox → SAM prompt → instance masks**.
- Extend it for cell tracking and compare SAM v2 vs prompt-guided tracking.

---

## References

- Kirillov, A. *et al.* Segment Anything. *ICCV* (2023). arXiv:2304.02643
- Archit, A. *et al.* Segment Anything for Microscopy (μSAM). *Nature Methods* (2025).
- EPFL CVLAB. Electron Microscopy Dataset (CA1 hippocampus; mitochondria annotations).
- Zhou, P., Du, B., & Xu, Y. CellSeg1: Robust Cell Segmentation with One Training Image. *arXiv* (2024). doi:10.48550/arxiv.2412.01410

**Implementation resources**
- Reenu (Python for Microscopists), fine-tuning SAM for mitochondria (notebook): https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb  
- MathieuNlp, *Sam_LoRA* (LoRA for SAM): https://github.com/MathieuNlp/Sam_LoRA  
- computational-cell-analytics, *micro-sam* (PEFT SAM): https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/models/peft_sam.py  
- JamesQFreeman, *Sam_LoRA* (LoRA variant): https://github.com/JamesQFreeman/Sam_LoRA/blob/main/sam_lora.py
