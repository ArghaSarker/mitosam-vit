from pathlib import Path

# This file lives in: MitoSAM-ViT/mitosam/paths.py
# Project root is the directory that contains "mitosam", "data", "notebooks", etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Your original dataset folder (now inside raw/)
DATASET_DIR = RAW_DATA_DIR / "Dataset"

# Individual volumes
TRAINING_VOLUME = DATASET_DIR / "training.tif"
TRAINING_GT_VOLUME = DATASET_DIR / "training_groundtruth.tif"
TESTING_VOLUME = DATASET_DIR / "testing.tif"
TESTING_GT_VOLUME = DATASET_DIR / "testing_groundtruth.tif"
VOLUMEDATA = DATASET_DIR / "volumedata.tif"
