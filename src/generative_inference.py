import argparse
import json
import random
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, set_seed

import data


ROOT = Path(__file__).resolve().parent.parent

INSTRUCTIONS = {
    "basic": "Answer the question using the context.",
    "strict": (
        "Answer the question using one contiguous span copied exactly from the context. "
        "Return only that span, without explanations or additional text."
    ),
}


def choose_demos(training, evaluation, shots, seed):
    if shots == 0:
        return []

    train_titles = {row["title"] for row in training}
    if train_titles & {row["title"] for row in evaluation}:
        raise ValueError("Training and evaluation articles overlap.")

    candidates = [
        row for row in training
        if len(row["context"].split()) <= 120
    ]
    random.Random(seed).shuffle(candidates)

    demos, contexts = [], set()
    for row in candidates:
        if row["context"] not in contexts:
            demos.append(row)
            contexts.add(row["context"])

        if len(demos) == shots:
            return demos

    raise ValueError("Not enough distinct short passages for demonstrations.")


def build_prompt(tokenizer, example, demos, variant):
    def question_text(row):
        return f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}"

    messages = [{"role": "system", "content": INSTRUCTIONS[variant]}]

    for demo in demos:
        messages.extend([
            {"role": "user", "content": question_text(demo)},
            {"role": "assistant", "content": demo["answers"]["text"][0]},
        ])

    messages.append({"role": "user", "content": question_text(example)})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def score_predictions(examples, predictions):
    scorer = data.load_official_eval()
    exact, f1, non_extractive, empty = 0, 0, 0, 0

    for row in examples:
        prediction = predictions[row["id"]]
        answers = row["answers"]["text"]

        exact += max(
            scorer.compute_exact(answer, prediction)
            for answer in answers
        )
        f1 += max(
            scorer.compute_f1(answer, prediction)
            for answer in answers
        )
        empty += not prediction
        non_extractive += (
            bool(prediction) and prediction not in row["context"]
        )

    count = len(examples)
    return {
        "exact": 100 * exact / count,
        "f1": 100 * f1 / count,
        "total": count,
        "non_extractive_rate": 100 * non_extractive / count,
        "empty_answer_rate": 100 * empty / count,
    }


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/splits/val_split.json",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=ROOT / "data/splits/train_split.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", choices=INSTRUCTIONS, default="strict")
    parser.add_argument("--shots", type=int, choices=[0, 1, 3], default=0)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    for name in [
        "batch_size", "max_examples", "max_input_tokens", "max_new_tokens"
    ]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    return args


def main():
    args = parse_args()

    if (args.output_dir / "predictions.json").exists():
        raise FileExistsError(
            "This run already has predictions. Choose another output directory."
        )

    examples = data.flatten(data.load_squad(args.data))
    training = (
        data.flatten(data.load_squad(args.train_data))
        if args.shots else []
    )
    demos = choose_demos(training, examples, args.shots, args.seed)

    if args.max_examples:
        examples = random.Random(args.seed).sample(
            examples, min(args.max_examples, len(examples))
        )

    if not examples:
        raise ValueError("The evaluation file contains no questions.")

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        build_prompt(tokenizer, row, demos, args.prompt)
        for row in examples
    ]
    lengths = [
        len(ids)
        for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]
    ]
    longest = max(lengths)

    if longest > args.max_input_tokens:
        raise ValueError(
            f"Longest prompt has {longest} tokens. Increase --max-input-tokens "
            "for all compared runs; no inputs have been truncated."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    settings.update(
        demo_ids=[row["id"] for row in demos],
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )

    save_json(args.output_dir / "config.json", settings)
    (args.output_dir / "prompt_preview.txt").write_text(
        prompts[0], encoding="utf-8"
    )

    print(
        f"Prepared {len(examples)} questions; "
        f"longest prompt: {longest} tokens."
    )

    if args.preview:
        print(f"Prompt saved to {args.output_dir / 'prompt_preview.txt'}")
        return

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use --device cpu or check your GPU allocation."
        )

    dtype = torch.float32
    if device == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
    ).to(device).eval()

    if longest + args.max_new_tokens > model.config.text_config.max_position_embeddings:
        raise ValueError(
            "The requested prompt and answer exceed the model's context limit."
        )

    settings.update(
        resolved_device=device,
        dtype=str(dtype),
        model_commit=getattr(model.config, "_commit_hash", None),
        gpu=torch.cuda.get_device_name() if device == "cuda" else None,
    )
    save_json(args.output_dir / "config.json", settings)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    predictions = {}

    with (args.output_dir / "generations.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]

            inputs = tokenizer(
                prompts[start:start + args.batch_size],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            texts = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            for row, raw in zip(batch, texts):
                prediction = raw.strip()
                predictions[row["id"]] = prediction

                record = {
                    "id": row["id"],
                    "raw_output": raw,
                    "prediction": prediction,
                    "is_extractive": (
                        bool(prediction) and prediction in row["context"]
                    ),
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            file.flush()
            print(
                f"Completed {len(predictions)}/{len(examples)}",
                flush=True,
            )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    save_json(args.output_dir / "predictions.json", predictions)

    metrics = score_predictions(examples, predictions)
    metrics.update(
        elapsed_seconds=elapsed,
        questions_per_second=len(examples) / elapsed,
        peak_gpu_memory_gb=(
            torch.cuda.max_memory_allocated() / 1024**3
            if device == "cuda" else None
        ),
    )

    save_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
