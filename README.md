# 🧬 MitoSAM-ViT

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<img src="https://img.shields.io/badge/status-work%20in%20progress-orange" />

A work-in-progress project to adapt the **Segment Anything Model (SAM)** to **mitochondria segmentation** in 3D electron microscopy (EM) data using **LoRA** and **PEFT adapters**.

---

## 🔬 Project overview

**MitoSAM-ViT** explores whether a general-purpose segmentation foundation model, **Segment Anything** (SAM) (Kirillov *et al.*, 2023), can be efficiently specialized for **mitochondria** in EM images using **parameter-efficient fine-tuning**.

Current focus:

- Fine-tuning a **SAM ViT backbone** with **LoRA adapters** using the **PEFT** library.
- Training on a **mitochondria EM dataset from EPFL**, with voxel-level annotations.

The long-term goal is a **fully automated mitochondria segmentation pipeline** that combines:

- **YOLOv8**-based object detection for mitochondria; and  
- **SAM**, prompted with YOLO bounding boxes, for refined instance segmentation.

This project is conceptually related to:
- **Segment Anything (SAM)** – general promptable segmentation.  
- **Segment Anything for Microscopy (MicroSAM / μSAM)** – SAM-based tools for microscopy segmentation and tracking.  
- **Ultralytics YOLOv8** – modern real-time object detector used here for mitochondria localization.

---

## 🧠 Method (current stage)

1. **Base model**
   - Segment Anything Model (SAM) image encoder + mask decoder.
   - Promptable segmentation via points / boxes / masks.

2. **Parameter-efficient fine-tuning**
   - Insert **LoRA** adapters into selected Transformer layers of SAM’s ViT encoder.
   - Keep original SAM weights frozen; train only the low-rank adapter parameters.
   - Use **Hugging Face PEFT** to manage LoRA adapters (config, loading/saving, experiments).

3. **Data**
   - Mitochondria EM dataset from **EPFL** (3D volume EM with expert mitochondria annotations).
   - Standard split into training / validation volumes, with pre-processing handled in the `data/` and `mitosam/dataset.py` pipeline.

---

## 🗺️ Planned roadmap

This repository is **under active development**. Planned next steps:

1. **Mitochondria detection with YOLOv8**
   - Train a **YOLOv8** detector on EM images to predict **bounding boxes** for mitochondria.
   - Optimize for high recall to ensure most mitochondria are detected.

2. **Detection-guided SAM segmentation**
   - Use YOLOv8 **bounding boxes as prompts** for SAM.
   - For each detected box, SAM refines the region into a high-quality segmentation mask.
   - This forms a **detection + promptable segmentation pipeline** for automated mitochondria instance segmentation.

3. **Evaluation & analysis**
   - Compare:
     - Zero-shot SAM vs. LoRA-fine-tuned SAM.
     - Bounding-box-only detection vs. YOLOv8 + SAM masks.
   - Report metrics such as IoU / Dice for both semantic and instance-level mitochondria segmentation.

---

## 📁 Project Organization

This project follows the [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/) layout:

```text
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mitosam and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mitosam   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mitosam a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations



```
## 🚧 Status

This is **research code** and is **actively evolving**.  
Interfaces, training scripts, and experiment configurations may change as the project progresses.



## 📚 References

- **Segment Anything (SAM)**  
  A. Kirillov *et al.* “Segment Anything.” *ICCV*, 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

- **Segment Anything for Microscopy (MicroSAM / μSAM)**  
  A. Archit *et al.* “Segment Anything for Microscopy (μSAM).” *Nature Methods*, 2025. [Article](https://www.nature.com/articles/s41592-024-02580-4)

- **Ultralytics YOLOv8**  
  Ultralytics. “Ultralytics YOLOv8: State-of-the-Art Computer Vision Model.” [Documentation](https://docs.ultralytics.com/models/yolov8/)

- **EPFL Electron Microscopy (CA1 Hippocampus) Dataset**  
  EPFL CVLAB. “Electron Microscopy Dataset (CA1 hippocampus, mitochondria annotations).” [Dataset page](https://www.epfl.ch/labs/cvlab/data/data-em/)

