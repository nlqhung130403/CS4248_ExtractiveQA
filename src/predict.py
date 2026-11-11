"""Phase 4: run a trained model on a SQuAD-format file and save its answers.

    python src/predict.py --model outputs/checkpoints/final --data data/splits/val_split.json
    python src/predict.py --model outputs/checkpoints/final --data data/dev-v1.1.json

This writes two files to --out_dir (default outputs/predictions/<model folder name>/):

    <data name>_logits.npz         raw start/end scores of every window (input to postprocess.py)
    <data name>_predictions.json   {question_id: answer} with the default post-processing settings

and prints EM/F1. Run it on a GPU (about a minute for dev on a T4; 15-45 minutes on a laptop CPU).
To try other post-processing settings later, rerun only postprocess.py on the logits file:
no model or GPU needed.

Quick check on a few questions (works on CPU):

    python src/predict.py --model outputs/checkpoints/final --data data/splits/val_split.json --max_examples 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering

import config
import data
import postprocess
import preprocess

MODEL_INPUTS = ["input_ids", "token_type_ids", "attention_mask"]


def window_settings(model_path):
    """max_length and doc_stride the model was trained with (from run_config.json), else the defaults.
    Evaluation windows should be built the same way as training windows."""
    run_config = Path(model_path) / "run_config.json"
    if run_config.exists():
        cfg = json.loads(run_config.read_text(encoding="utf-8"))["config"]
        return cfg["max_length"], cfg["doc_stride"]
    return config.MAX_LENGTH, config.DOC_STRIDE


@torch.no_grad()
def run_model(model, features, batch_size, device):
    """Start and end logits, each a float32 array of shape (windows, max_length)."""
    model.to(device).eval()
    inputs = {k: torch.tensor(features[k]) for k in MODEL_INPUTS if k in features.column_names}
    n = len(features)
    starts, ends = [], []
    start_time = time.time()
    for i in range(0, n, batch_size):
        batch = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
        # fp16 on GPU like in training; full precision on CPU.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**batch)
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
        done = min(i + batch_size, n)
        if (i // batch_size) % 20 == 0 or done == n:
            print(f"  {done}/{n} windows ({time.time() - start_time:.0f}s)", flush=True)
    return np.concatenate(starts), np.concatenate(ends)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="checkpoint folder (e.g. outputs/checkpoints/final) or a Hugging Face model name")
    parser.add_argument("--data", type=Path, required=True, help="SQuAD-format JSON file")
    parser.add_argument("--out_dir", type=Path, help="default: outputs/predictions/<model folder name>")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_examples", type=int, help="only the first N questions (for quick checks)")
    args = parser.parse_args()

    model_name = Path(args.model).name
    out_dir = args.out_dir or config.OUTPUTS_DIR / "predictions" / model_name
    stem = args.data.stem
    if args.max_examples is not None:
        stem += f"_first{args.max_examples}"
    logits_path = out_dir / f"{stem}_logits.npz"

    squad = data.load_squad(args.data)
    examples = data.load_split(args.data)
    if args.max_examples is not None:
        examples = examples.select(range(min(args.max_examples, len(examples))))
        keep = set(examples["id"])
        # Keep only those questions so the scorer does not count the others as missing.
        for article in squad["data"]:
            for paragraph in article["paragraphs"]:
                paragraph["qas"] = [qa for qa in paragraph["qas"] if qa["id"] in keep]

    max_length, doc_stride = window_settings(args.model)
    tokenizer = preprocess.load_tokenizer(args.model)
    features = examples.map(preprocess.prepare_eval_features, batched=True,
                            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length,
                                       "doc_stride": doc_stride},
                            remove_columns=examples.column_names, desc="tokenising")
    preprocess.check_eval_features(examples, features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(examples)} questions -> {len(features)} windows (max_length {max_length}, "
          f"doc_stride {doc_stride}); running {args.model} on {device}")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    start_time = time.time()
    start_logits, end_logits = run_model(model, features, args.batch_size, device)
    seconds = time.time() - start_time

    meta = {"model": str(args.model), "data": args.data.name, "max_length": max_length,
            "doc_stride": doc_stride, "max_examples": args.max_examples, "device": str(device),
            "inference_seconds": round(seconds, 1)}
    offsets = postprocess.offsets_array(features["offset_mapping"])
    postprocess.save_logits(logits_path, start_logits, end_logits, offsets, features["example_id"], meta)
    print(f"inference took {seconds:.0f}s; wrote {logits_path}")

    logits = postprocess.load_logits(logits_path)
    postprocess.run(logits, squad, [postprocess.N_BEST], [postprocess.MAX_ANSWER_LENGTH],
                    [postprocess.SCORE], postprocess.predictions_path(logits_path))


if __name__ == "__main__":
    main()
