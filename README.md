# 🧬 MitoSAM-ViT

Fine-tuning **Segment Anything (SAM)** for **mitochondria segmentation** in 2D EM using **LoRA**.  
This repo compares **fine-tuned vs. vanilla SAM** and studies how **prompt type** (bbox vs. point grids) affects segmentation quality.

---

## Project

### Segment Anything Model (SAM) 
SAM is a promptable segmentation model with two main components:

- **Image encoder (ViT)**: converts an input image into a latent embedding.
- **Prompt encoder + mask decoder**: encodes prompts (points/boxes) and decodes masks conditioned on the image embedding.

Model used: **SAM ViT-B** (~91M parameters), pretrained on **SA-1B** (≈11M images, 1.1B masks).

<img src="reports/figures/sam_paper.png" width="900">

*Figure from Kirillov et al., "Segment Anything" (2023)*


### What was fine-tuned (and why)
Goal: adapt SAM to mitochondria while keeping training efficient.

- **Image encoder**: adapted with **LoRA** so only a small fraction of parameters are updated.
- **Mask decoder**: **fully fine-tuned** (≈4M parameters) to improve prompt-to-mask decoding for the mitochondria domain.

### Project goals
- Fine-tune SAM for mitochondria segmentation with LoRA.
- Quantify how **prompt strategies** (bbox vs. point grids) change performance.
- Analyze model sensitivity using **Integrated Gradients / saliency** and **Occlusion Sensitivity** (as diagnostic tools for explainability, not ground-truth explanations).

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

### Can we explain the output? 


- **Saliency / Integrated Gradients (IG)**: gradient-based attributions that highlight pixels where changing the input would most change the model’s output (for a chosen target/output). IG is typically more stable than raw gradients because it integrates gradients along a path from a baseline image.

- **Occlusion sensitivity**: occludes (masks) small patches of the image and measures how much the prediction quality changes (e.g., Dice/IoU or mask confidence). Regions where occlusion hurts performance most are interpreted as “evidence” the model relied on.

Note: the output of the Occlusion Sensitivity and Integrated Gradiants has been normalized for visualization purpose. 

### Findings
- For **fine-tuned SAM**, **bbox** and **denser point grids (4×4 / 8×8)** produce similar-looking masks; **2×2** can underperform because prompts are too sparse to constrain the object well.
- For **vanilla SAM**, point-grid prompts can fail (often producing overly large or poorly localized masks), suggesting the pretrained model is not well aligned with mitochondria EM appearance without adaptation.
- When predictions collapse to trivial masks (too large / too small), attribution maps can become less informative; they may reflect model instability rather than meaningful localization.

### Quantitative comparison (Dice / IoU)


- **Prompt-wise mean Dice/IoU (base vs fine-tuned)**  
  <img src="reports/figures/dice_iou_base_vs_finetuned_by_prompt.png" width="900">

- **Score distribution (fine-tuned only, values < 0.6 filtered for readability)**  
  <img src="reports/figures/dice_iou_violin_box_finetuned_only_ge_0.6.png" width="900">

Summary on this dataset:
- **Fine-tuned SAM** is consistently strong across prompt types (Dice/IoU).
- **Vanilla SAM** is substantially weaker for **point-grid prompts**; bbox prompting is relatively better but still below the fine-tuned model.

---

## Discussion
- Fine-tuning reduces **prompt sensitivity**: after adaptation, bbox and point-grid prompts yield similar performance, suggesting the model learned mitochondria-specific features and decoding behavior.
- Vanilla SAM is not mitochondria-specialized: sparse prompts are often insufficient to guide correct masks in EM imagery.
- The fine-tuned violin plots show tighter score distributions for bbox prompts compared to point grids, indicating slightly more consistent behavior under bbox prompting.


---

## Conclusion
LoRA-adapting the **image encoder** and fully tuning the **mask decoder** yields a SAM variant that segments mitochondria reliably in EM images and is more robust to prompt choice than vanilla SAM.  

Using LoRA also makes training much more efficient: in my setup, only ~**6.1%** of the model parameters were trainable, which substantially reduces compute and makes adaptation feasible on **consumer-grade GPUs** .

---

## Future work
- Add **YOLO** mitochondria detection to generate bbox prompts automatically.
- Build an end-to-end pipeline: **YOLO bbox → SAM prompt → instance masks**.
- Extend it for cell tracking and compare with SAM_v2 vs prompt guided tracking.
---

## References

- Kirillov, A. *et al.* Segment Anything. *ICCV* (2023). arXiv:2304.02643  
- Archit, A. *et al.* Segment Anything for Microscopy (μSAM). *Nature Methods* (2025).  
- EPFL CVLAB. Electron Microscopy Dataset (CA1 hippocampus; mitochondria annotations).

**Implementation resources**
- Reenu (Python for Microscopists), fine-tuning SAM for mitochondria (notebook): https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb  
- MathieuNlp, *Sam_LoRA* (LoRA for SAM): https://github.com/MathieuNlp/Sam_LoRA  
- computational-cell-analytics, *micro-sam* (PEFT SAM): https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/models/peft_sam.py  
- JamesQFreeman, *Sam_LoRA* (LoRA variant): https://github.com/JamesQFreeman/Sam_LoRA/blob/main/sam_lora.py  
