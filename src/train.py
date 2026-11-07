"""Phase 3: fine-tune ELECTRA-base with a QA head on the training split.

Run from the repo root:

    python src/train.py                                   # ELECTRA paper recipe (LR 1e-4, LLRD 0.8)
    python src/train.py --preset baseline                 # LR 3e-5, no LLRD
    python src/train.py --run_name lr2e-4 --learning_rate 2e-4
    python src/train.py --run_name smoke --max_train_examples 200 --max_steps 5   # quick CPU check

Any TrainConfig field (src/config.py) can be overridden with --<field> <value>.

The run is written to outputs/checkpoints/<run_name>/: one checkpoint per epoch, and the final
model and tokenizer at the top level, ready for predict.py. A copy of the config and the loss
history goes to outputs/logs/<run_name>/ (small, safe to commit).

To continue a run that was interrupted (e.g. a Colab disconnect), rerun the same command with
--resume: it picks up from the last epoch checkpoint in the run's directory.
"""

import argparse
import dataclasses
import json
import time
import typing

import torch
import transformers
from transformers import (AutoModelForQuestionAnswering, Trainer, TrainingArguments,
                          default_data_collator, set_seed)

import config
import data
import preprocess


# ---------------------------------------------------------------------------------------------
# Layer-wise learning-rate decay (section 5.3)
# ---------------------------------------------------------------------------------------------

def layer_depth(name, n_layers):
    """Depth of a parameter: 0 for the embeddings, i + 1 for encoder layer i, n_layers + 1 for the
    QA head. Lower depth means further from the output and a smaller learning rate."""
    if name.startswith("electra.embeddings"):  # also covers embeddings_project in small models
        return 0
    if name.startswith("electra.encoder.layer."):
        return int(name.split(".")[3]) + 1
    if name.startswith("qa_outputs."):
        return n_layers + 1
    raise ValueError(f"don't know which layer parameter {name!r} belongs to")


def llrd_param_groups(model, base_lr, decay, weight_decay=0.0, no_decay_names=()):
    """AdamW parameter groups with lr = base_lr * decay ** (n_layers + 1 - depth).

    The QA head gets base_lr, encoder layer 11 gets base_lr * decay, ..., the embeddings get
    base_lr * decay ** 13. decay = 1.0 gives every parameter base_lr (no LLRD).
    Parameters in no_decay_names (biases, LayerNorm) get no weight decay.
    Groups are ordered from the head down, so the LR the Trainer logs is the head's (the peak).
    """
    n_layers = model.config.num_hidden_layers
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        depth = layer_depth(name, n_layers)
        wd = 0.0 if name in no_decay_names else weight_decay
        key = (depth, wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * decay ** (n_layers + 1 - depth),
                           "weight_decay": wd, "depth": depth}
        groups[key]["params"].append(param)
    return [groups[k] for k in sorted(groups, key=lambda k: (-k[0], k[1]))]


def describe_param_groups(groups, n_layers):
    names = {0: "embeddings", n_layers + 1: "QA head"}
    seen = set()
    for g in groups:
        if g["depth"] in seen:
            continue
        seen.add(g["depth"])
        n = sum(p.numel() for h in groups if h["depth"] == g["depth"] for p in h["params"])
        label = names.get(g["depth"], f"layer {g['depth'] - 1}")
        print(f"  {label:<11} lr {g['lr']:.3e}  ({n:,} params)")


class LLRDTrainer(Trainer):
    """Trainer whose AdamW optimizer uses layer-wise learning-rate decay.

    Only the optimizer is replaced. The Trainer still builds the scheduler (linear warmup, then
    linear decay to 0), which scales every group's LR by the same factor, so the layer ratios hold
    throughout training.
    """

    def __init__(self, *args, llrd_decay=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.llrd_decay = llrd_decay

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            model = self.model if model is None else model
            decay_names = set(self.get_decay_parameter_names(model))
            no_decay = {n for n, _ in model.named_parameters() if n not in decay_names}
            groups = llrd_param_groups(model, self.args.learning_rate, self.llrd_decay,
                                       self.args.weight_decay, no_decay)
            describe_param_groups(groups, model.config.num_hidden_layers)
            self.optimizer = torch.optim.AdamW(
                groups, lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon)
        return self.optimizer


# ---------------------------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------------------------

def resolve_precision(precision):
    """'auto' -> bf16 on GPUs that support it (A100, L4, ...), fp16 on other GPUs (T4, P100), fp32 on CPU."""
    if precision == "auto":
        if not torch.cuda.is_available():
            return "fp32"
        # Without including_emulation=False, T4s report bf16 support but run it slowly in emulation.
        return "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
    if precision in ("fp16", "bf16") and not torch.cuda.is_available():
        raise ValueError(f"{precision} needs a GPU; use --precision fp32 on CPU")
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"unknown precision {precision!r}")
    return precision


def gradient_accumulation(cfg):
    """Accumulation steps so that per_device_batch_size x n_gpus x steps == batch_size (section 5.4)."""
    per_step = cfg.per_device_batch_size * max(1, torch.cuda.device_count())
    if cfg.batch_size % per_step:
        raise ValueError(f"batch_size {cfg.batch_size} is not a multiple of "
                         f"per_device_batch_size x n_gpus = {per_step}")
    return cfg.batch_size // per_step


def warmup_kwargs(ratio):
    """transformers >= 5 takes a fraction of the total steps as warmup_steps; 4.x needs warmup_ratio
    and would read warmup_steps=0.1 as 0.1 steps, i.e. no warmup at all."""
    if int(transformers.__version__.split(".")[0]) >= 5:
        return {"warmup_steps": ratio}
    return {"warmup_ratio": ratio}


def training_arguments(cfg, output_dir):
    precision = resolve_precision(cfg.precision)
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation(cfg),
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,  # the QA head's LR; LLRDTrainer derives the others
        lr_scheduler_type="linear",
        **warmup_kwargs(cfg.warmup_ratio),  # LR rises over this fraction of the total steps
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_strategy=cfg.save_strategy,  # the final model is saved either way
        eval_strategy="no",               # Trainer eval only reports loss; we score EM/F1 with predict.py
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
    )


def build_train_features(cfg, tokenizer):
    examples = data.load_split(config.TRAIN_SPLIT_FILE)
    if cfg.max_train_examples is not None:
        examples = examples.select(range(min(cfg.max_train_examples, len(examples))))
    features = examples.map(preprocess.prepare_train_features, batched=True,
                            fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length,
                                       "doc_stride": cfg.doc_stride},
                            remove_columns=examples.column_names, desc="tokenising")
    print(f"training data: {len(examples)} examples -> {len(features)} windows")
    return features


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train(cfg, resume=False):
    if not cfg.run_name:
        cfg.run_name = f"{cfg.preset}-{time.strftime('%Y%m%d-%H%M%S')}"
    if cfg.llrd_decay == 1.0 and cfg.learning_rate > 5e-5:
        print(f"WARNING: LR {cfg.learning_rate} without layer-wise decay may diverge; ~3e-5 is the usual choice")

    output_dir = config.CHECKPOINTS_DIR / cfg.run_name
    log_dir = config.LOGS_DIR / cfg.run_name
    args = training_arguments(cfg, output_dir)

    run_info = {"config": dataclasses.asdict(cfg),
                "derived": {"gradient_accumulation_steps": args.gradient_accumulation_steps,
                            "n_gpus": torch.cuda.device_count(),
                            "precision": resolve_precision(cfg.precision),
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}
    print(json.dumps(run_info, indent=2))
    save_json(run_info, output_dir / "run_config.json")
    save_json(run_info, log_dir / "run_config.json")

    set_seed(cfg.seed)  # before loading the model, so the new QA head is initialised reproducibly
    tokenizer = preprocess.load_tokenizer(cfg.model_name)
    train_features = build_train_features(cfg, tokenizer)
    # The warning about newly initialised qa_outputs weights is expected: that is the new QA head.
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    trainer = LLRDTrainer(
        model=model,
        args=args,
        train_dataset=train_features,
        data_collator=default_data_collator,  # windows are already padded to max_length
        processing_class=tokenizer,           # saved with every checkpoint, so each one is loadable alone
        llrd_decay=cfg.llrd_decay,
    )
    start = time.time()
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    minutes = (time.time() - start) / 60

    trainer.save_model(str(output_dir))  # final model + tokenizer, for predict.py
    run_info["result"] = {**result.metrics, "train_minutes": round(minutes, 1)}
    save_json(run_info, output_dir / "run_config.json")
    save_json(run_info, log_dir / "run_config.json")
    save_json(trainer.state.log_history, log_dir / "log_history.json")
    last_loss = next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), float("nan"))
    print(f"\ntrained in {minutes:.1f} min; last logged loss {last_loss:.4f}, "
          f"mean over the run {result.training_loss:.4f}")
    print(f"model saved to {output_dir.relative_to(config.ROOT)}, logs to {log_dir.relative_to(config.ROOT)}")
    return trainer


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="electra-paper", choices=sorted(config.PRESETS))
    parser.add_argument("--resume", action="store_true", help="continue from the run's last checkpoint")
    hints = typing.get_type_hints(config.TrainConfig)
    for f in dataclasses.fields(config.TrainConfig):
        if f.name == "preset":
            continue
        # "int | None" -> int; everything else is a plain type.
        kind = next((t for t in typing.get_args(hints[f.name]) if t is not type(None)), hints[f.name])
        parser.add_argument(f"--{f.name}", type=kind, default=None, help=f"default: {f.default}")
    args = vars(parser.parse_args())
    preset, resume = args.pop("preset"), args.pop("resume")
    overrides = {k: v for k, v in args.items() if v is not None}
    if resume and not overrides.get("run_name"):
        parser.error("--resume needs the --run_name of the run to continue")
    return config.make_train_config(preset, **overrides), resume


if __name__ == "__main__":
    train(*parse_args())
