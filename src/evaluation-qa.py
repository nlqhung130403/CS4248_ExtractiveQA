"""Evaluate SQuAD predictions using the supplied official scorer.

Run `python3 src/evaluation-qa.py --help` for usage.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path


def read_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object
    )


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("official_squad", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def flatten_dataset(data):
    questions = {}
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                if qid in questions:
                    raise ValueError(f"Duplicate question ID: {qid}")
                questions[qid] = {**qa, "context": paragraph["context"]}
    if not questions:
        raise ValueError("Dataset contains no questions.")
    return questions


def load_predictions(path):
    """Accept a plain ID-to-answer map or the LLM evaluator's result bundle."""
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("raw_predictions"), dict):
        return payload["raw_predictions"], "raw_predictions"
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), dict):
        return payload["predictions"], "predictions (already cleaned by producer)"
    return payload, "plain prediction map"


def validate_predictions(predictions, questions):
    if not isinstance(predictions, dict) or not all(
        isinstance(v, str) for v in predictions.values()
    ):
        raise ValueError(
            'Predictions must be a JSON object: {"question_id": "answer text"}.'
        )
    missing = questions.keys() - predictions.keys()
    extra = predictions.keys() - questions.keys()
    if missing or extra:
        raise ValueError(
            f"Prediction IDs do not match dataset: {len(missing)} missing, "
            f"{len(extra)} extra. Examples: {sorted(missing)[:3]}, {sorted(extra)[:3]}"
        )


def map_to_span(prediction, context):
    """Return (span, method, start, end); end is exclusive.

    Clean the answer, use the first matching passage span, or return empty
    if none matches. No gold answers are used.
    """
    candidate = prediction.strip()
    if not candidate:
        return "", "empty", None, None
    candidates = [candidate]
    cleaned = re.sub(
        r"^(?:answer|the answer is)\s*:\s*", "", candidate, flags=re.I
    ).strip()
    if len(cleaned) >= 2 and (cleaned[0], cleaned[-1]) in [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
    ]:
        cleaned = cleaned[1:-1].strip()
    if cleaned and cleaned != candidate:
        candidates.append(cleaned)
    for text in candidates:
        start = context.find(text)
        if start != -1:
            return (
                context[start : start + len(text)],
                "exact" if text == candidate else "cleaned",
                start,
                start + len(text),
            )
        # Word boundaries avoid mapping 'US' into the middle of 'Russia'.
        pattern = r"\s+".join(re.escape(token) for token in text.split())
        if text[0].isalnum():
            pattern = r"(?<!\w)" + pattern
        if text[-1].isalnum():
            pattern += r"(?!\w)"
        match = re.search(pattern, context, flags=re.I)
        if match:
            return (
                context[match.start() : match.end()],
                "case_whitespace",
                match.start(),
                match.end(),
            )
    return "", "unmatched", None, None


def question_type(question):
    words = re.findall(r"\b\w+\b", question.lower())
    if words[:2] in [["how", "many"], ["how", "much"]]:
        return " ".join(words[:2])
    return (
        words[0]
        if words
        and words[0]
        in {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"}
        else "other"
    )


def length_bucket(answers):
    # Minimum whitespace-token length among references; fixed before evaluation.
    length = min((len(a["text"].split()) for a in answers), default=0)
    return (
        "0"
        if length == 0
        else (
            "1"
            if length == 1
            else "2-3" if length <= 3 else "4-7" if length <= 7 else "8+"
        )
    )


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(data_path, prediction_paths, scorer_path, output_dir):
    data = read_json(data_path)
    questions = flatten_dataset(data)
    scorer = load_scorer(scorer_path)
    names = [Path(path).stem for path in prediction_paths]
    if len(set(names)) != len(names):
        raise ValueError("Prediction files must have distinct basenames (model names).")
    # Validate every input before writing results, avoiding silently partial comparisons.
    loaded = [load_predictions(path) for path in prediction_paths]
    inputs = [predictions for predictions, _ in loaded]
    for predictions in inputs:
        validate_predictions(predictions, questions)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, all_groups = [], []
    for name, raw in zip(names, inputs):
        directory = output_dir / name
        directory.mkdir(exist_ok=True)
        mapping = {
            qid: map_to_span(raw[qid], qa["context"]) for qid, qa in questions.items()
        }
        mapped = {qid: result[0] for qid, result in mapping.items()}
        write_json(directory / "predictions.raw.json", raw)
        write_json(directory / "predictions.mapped.json", mapped)
        # Delegate metric definitions and reference-answer handling to the official code.
        raw_em, raw_f1 = scorer.get_raw_scores(data["data"], raw)
        mapped_em, mapped_f1 = scorer.get_raw_scores(data["data"], mapped)
        rows = []
        for qid, qa in questions.items():
            span, method, start, end = mapping[qid]
            rows.append(
                {
                    "id": qid,
                    "question": qa["question"],
                    "context": qa["context"],
                    "gold_answers": json.dumps(
                        [a["text"] for a in qa["answers"]], ensure_ascii=False
                    ),
                    "question_type": question_type(qa["question"]),
                    "answer_length": length_bucket(qa["answers"]),
                    "answerability": "answerable" if qa["answers"] else "unanswerable",
                    "raw_prediction": raw[qid],
                    "mapped_prediction": span,
                    "mapping_method": method,
                    "span_start": start,
                    "span_end": end,
                    "raw_non_extractive": bool(raw[qid].strip())
                    and raw[qid].strip() not in qa["context"],
                    "raw_empty": not bool(raw[qid].strip()),
                    "raw_em": raw_em[qid],
                    "raw_f1": raw_f1[qid],
                    "mapped_em": mapped_em[qid],
                    "mapped_f1": mapped_f1[qid],
                }
            )
        write_csv(directory / "per_question.csv", rows)
        nonempty_count = sum(not row["raw_empty"] for row in rows)
        nonextractive_count = sum(row["raw_non_extractive"] for row in rows)
        for variant, em, f1 in [
            ("raw", raw_em, raw_f1),
            ("mapped", mapped_em, mapped_f1),
        ]:
            metrics = dict(scorer.make_eval_dict(em, f1))
            write_json(directory / f"metrics.{variant}.json", metrics)
            summary.append(
                {
                    "model": name,
                    "variant": variant,
                    **metrics,
                    "raw_non_extractive_count": nonextractive_count,
                    "raw_nonempty_count": nonempty_count,
                    "raw_non_extractive_pct_of_nonempty": (
                        100 * nonextractive_count / nonempty_count
                        if nonempty_count
                        else ""
                    ),
                    "mapping_failure_count": sum(
                        r["mapping_method"] == "unmatched" for r in rows
                    ),
                }
            )
            for dimension in [
                "question_type",
                "answer_length",
                "answerability",
                "raw_non_extractive",
                "mapping_method",
            ]:
                groups = defaultdict(list)
                for row in rows:
                    groups[str(row[dimension])].append(row["id"])
                for label, ids in sorted(groups.items()):
                    all_groups.append(
                        {
                            "model": name,
                            "variant": variant,
                            "dimension": dimension,
                            "group": label,
                            **scorer.make_eval_dict(em, f1, ids),
                        }
                    )
    write_csv(output_dir / "comparison.csv", summary)
    write_csv(output_dir / "breakdowns.csv", all_groups)
    files = [data_path, scorer_path, *prediction_paths]
    write_json(
        output_dir / "run_manifest.json",
        {
            "inputs_sha256": {
                str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in files
            },
            "mapping_policy": "conservative-v1; first match; unmatched -> empty string",
            "no_answer_policy": "Use final answer strings; apply any learned abstention threshold upstream on held-out training data.",
            "prediction_sources": {
                name: source for name, (_, source) in zip(names, loaded)
            },
            "dataset_version": data.get("version", "unspecified"),
            "question_count": len(questions),
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path, help="SQuAD-format evaluation JSON"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        nargs="+",
        type=Path,
        help="One JSON file per model/prompt variant",
    )
    parser.add_argument(
        "--scorer",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluate-v2.0.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/evaluation",
    )
    args = parser.parse_args()
    try:
        rows = evaluate(args.data, args.predictions, args.scorer, args.output)
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    for row in rows:
        print(
            f"{row['model']:20} {row['variant']:6} EM={row['exact']:.2f} F1={row['f1']:.2f} n={row['total']}"
        )
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
