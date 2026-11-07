"""Paths and settings shared across the pipeline. Later phases add their hyperparameters here."""

from dataclasses import dataclass, fields
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

# Outputs
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"  # git-ignored: each model is ~420 MB
LOGS_DIR = OUTPUTS_DIR / "logs"                # small per-run config + loss history, safe to commit


# Training (Phase 3). One TrainConfig fully describes a run; train.py saves it with the run.
@dataclass
class TrainConfig:
    run_name: str = ""                 # empty: train.py names the run <preset>-<timestamp>
    preset: str = "electra-paper"
    model_name: str = MODEL_NAME
    max_length: int = MAX_LENGTH
    doc_stride: int = DOC_STRIDE

    # ELECTRA paper (Clark et al., 2020), SQuAD fine-tuning settings for the Base model.
    learning_rate: float = 1e-4        # LR of the QA head; lower layers get less with LLRD
    llrd_decay: float = 0.8            # layer-wise LR decay; 1.0 turns LLRD off
    num_epochs: float = 2
    batch_size: int = 32               # effective batch: per_device_batch_size x n_gpus x accumulation
    per_device_batch_size: int = 16    # windows per forward pass; lower it if the GPU runs out of memory
    warmup_ratio: float = 0.1          # LR rises linearly over this fraction of steps, then decays to 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.0
    # Dropout stays at the model config's default of 0.1.
    precision: str = "auto"            # auto (bf16 if supported, else fp16; fp32 on CPU) | bf16 | fp16 | fp32
    seed: int = SEED

    logging_steps: int = 100
    save_strategy: str = "epoch"       # "no" for debug runs: skip the ~1.3 GB per-epoch checkpoints
    # Debugging: train on the first N examples only, and/or stop after N optimizer steps (-1 = no limit).
    max_train_examples: int | None = None
    max_steps: int = -1


# Named starting points for TrainConfig. Command-line flags in train.py override them.
PRESETS = {
    "electra-paper": {},  # the defaults above
    # Section 8.1 baseline: the common BERT-style recipe without layer-wise decay.
    "baseline": {"learning_rate": 3e-5, "llrd_decay": 1.0},
}


def make_train_config(preset="electra-paper", **overrides):
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown TrainConfig fields: {sorted(unknown)}")
    return TrainConfig(**{**PRESETS[preset], **overrides, "preset": preset})
