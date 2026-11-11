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


def load_annotations(path, questions):
    """Manual labels affect analysis only, never predictions or official scores."""
    if path is None:
        return {}
    annotations = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"id", "answer_type", "reasoning_type"}.issubset(reader.fieldnames):
            raise ValueError("Annotation CSV needs id, answer_type and reasoning_type columns.")
        for row in reader:
            qid = row["id"].strip()
            if qid not in questions or qid in annotations:
                raise ValueError(f"Unknown or duplicate annotation ID: {qid}")
            annotations[qid] = {key: (row.get(key) or "").strip()
                                for key in ("answer_type", "reasoning_type")}
    return annotations


def classify_answer(qa, entities=()):
    """Heuristic label of the first reference; uncertain cases stay unclassified."""
    if not qa["answers"]:
        return "no_answer", "reference"
    answer = qa["answers"][0]
    text = answer["text"].strip()
    start = answer.get("answer_start", -1)
    end = start + len(answer["text"])
    labels = {"PERSON": "person", "GPE": "location", "LOC": "location",
              "FAC": "location", "DATE": "date", "TIME": "time",
              "CARDINAL": "number", "ORDINAL": "number", "QUANTITY": "number",
              "PERCENT": "number", "MONEY": "number"}
    for entity in entities:
        if entity.start_char == start and entity.end_char == end:
            return labels.get(entity.label_, "other_entity"), "spacy"
    # Restrict dates to numeric date formats and month + day/year expressions.
    # Bare years remain numbers without contextual entity tagging or manual labels.
    months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}", text) or re.fullmatch(
        rf"(?:{months} \d{{1,2}}(?:,? \d{{4}})?|\d{{1,2}} {months}(?: \d{{4}})?|{months} \d{{4}})", text, re.I
    ):
        return "date", "rule"
    if re.fullmatch(r"[$£€]?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|st|nd|rd|th)?", text, re.I):
        return "number", "rule"
    return "unclassified", "rule"


def analysis_labels(questions, annotations, spacy_model=None):
    entities_by_context = {}
    if spacy_model:
        # Optional: the user installs a spaCy English model separately.
        import spacy
        nlp = spacy.load(spacy_model)
        if "ner" not in nlp.pipe_names:
            raise ValueError("The selected spaCy model must include an NER component.")
        contexts = list(dict.fromkeys(qa["context"] for qa in questions.values()))
        for context, doc in zip(contexts, nlp.pipe(contexts)):
            entities_by_context[context] = list(doc.ents)
    result = {}
    for qid, qa in questions.items():
        label, source = classify_answer(qa, entities_by_context.get(qa["context"], ()))
        manual = annotations.get(qid, {})
        if manual.get("answer_type"):
            label, source = manual["answer_type"], "manual"
        result[qid] = {"answer_type": label, "answer_type_source": source,
                       "reasoning_type": manual.get("reasoning_type") or "unannotated"}
    return result


def boundary_error(prediction, qa):
    """Potential character-span boundary error, not a causal diagnosis.

    Compare every literal prediction occurrence with all valid reference spans.
    Exact reference text wins. Otherwise select the relation with greatest
    character intersection-over-union; ties follow passage/reference order.
    """
    prediction = prediction.strip()
    if not prediction:
        return "empty"
    context = qa["context"]
    if prediction not in context:
        return "non_extractive"
    if not qa["answers"]:
        return "no_gold_span"
    if any(prediction == a["text"] for a in qa["answers"]):
        return "exact_span"
    references = []
    for answer in qa["answers"]:
        start = answer.get("answer_start", -1)
        end = start + len(answer["text"])
        if start >= 0 and end > start and context[start:end] == answer["text"]:
            references.append((start, end))
    if not references:
        return "invalid_reference_offsets"
    best_overlap, label = 0, "disjoint"
    for match in re.finditer(r"(?=" + re.escape(prediction) + r")", context):
        start, end = match.start(), match.start() + len(prediction)
        for gold_start, gold_end in references:
            overlap = max(0, min(end, gold_end) - max(start, gold_start))
            ratio = overlap / (max(end, gold_end) - min(start, gold_start))
            if ratio > best_overlap:
                best_overlap = ratio
                if start <= gold_start and end >= gold_end:
                    label = "potential_extra_text"
                elif start >= gold_start and end <= gold_end:
                    label = "potential_missing_text"
                else:
                    label = "potential_partial_overlap"
    return label


def evaluate(data_path, prediction_paths, scorer_path, output_dir, annotations_path=None, spacy_model=None):
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
    annotations = load_annotations(annotations_path, questions)
    labels = analysis_labels(questions, annotations, spacy_model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = output_dir / "annotation_template.csv"
    # Never overwrite a template that may already contain the user's annotations.
    if not template.exists():
        write_csv(template, [{"id": qid, "question": qa["question"],
                             "context": qa["context"],
                             "gold_answers": json.dumps([a["text"] for a in qa["answers"]]),
                             "suggested_answer_type": labels[qid]["answer_type"],
                             "answer_type": "", "reasoning_type": ""}
                            for qid, qa in questions.items()])
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
                    **labels[qid],
                    "raw_boundary_error": boundary_error(raw[qid], qa),
                    "mapped_boundary_error": boundary_error(span, qa),
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
                "answer_type",
                "reasoning_type",
                f"{variant}_boundary_error",
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
    if annotations_path is not None:
        files.append(annotations_path)
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
            "answer_type_policy": "first reference; manual override > optional exact NER span > date/number rules > unclassified",
            "spacy_model": spacy_model,
            "reasoning_annotated_count": sum(row["reasoning_type"] != "unannotated" for row in labels.values()),
            "boundary_policy": "literal spans; exact text first, then maximum character IoU across occurrences/references; potential errors only",
            "syntactic_divergence": "not implemented",
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
    parser.add_argument("--annotations", type=Path,
                        help="Optional CSV with id, answer_type, reasoning_type (blank labels allowed)")
    parser.add_argument("--spacy-model", help="Optional installed English spaCy NER model, e.g. en_core_web_sm")
    args = parser.parse_args()
    try:
        rows = evaluate(args.data, args.predictions, args.scorer, args.output, args.annotations, args.spacy_model)
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    for row in rows:
        print(
            f"{row['model']:20} {row['variant']:6} EM={row['exact']:.2f} F1={row['f1']:.2f} n={row['total']}"
        )
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
