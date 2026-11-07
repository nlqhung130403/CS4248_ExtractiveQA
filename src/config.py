"""Paths and settings shared across the pipeline. Later phases add their hyperparameters here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "train-v1.1.json"
DEV_FILE = DATA_DIR / "dev-v1.1.json"          # test set: never tune on it
SPLITS_DIR = DATA_DIR / "splits"
TRAIN_SPLIT_FILE = SPLITS_DIR / "train_split.json"
VAL_SPLIT_FILE = SPLITS_DIR / "val_split.json"

# Validation split (by article)
VAL_FRACTION = 0.05
SEED = 42

# Model and tokenisation
MODEL_NAME = "google/electra-base-discriminator"
MAX_LENGTH = 384    # tokens per window: [CLS] question [SEP] passage [SEP]
DOC_STRIDE = 128    # passage tokens shared by consecutive windows of the same example
