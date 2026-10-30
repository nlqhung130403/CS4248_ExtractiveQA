"""Phase 1: load SQuAD, split train into train/val by article, and flatten to one row per question.

Run from the repo root to create data/splits/ and run the Phase 1 checks:

    python src/data.py
"""

import json
import random
from pathlib import Path

import config


def load_squad(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_squad(squad, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep ensure_ascii=True (the default), like the original files: evaluate-v2.0.py opens
    # files without an encoding, which on Windows means cp1252 and fails on raw UTF-8.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(squad, f)


def split_by_article(squad, val_fraction=config.VAL_FRACTION, seed=config.SEED):
    """Randomly hold out whole articles. Returns (train_squad, val_squad), both in SQuAD format."""
    articles = squad["data"]
    n_val = round(len(articles) * val_fraction)
    val_idx = set(random.Random(seed).sample(range(len(articles)), n_val))

    # Keep the original article order inside each split.
    train = [a for i, a in enumerate(articles) if i not in val_idx]
    val = [a for i, a in enumerate(articles) if i in val_idx]
    return ({"version": squad["version"], "data": train},
            {"version": squad["version"], "data": val})


def flatten(squad):
    """One row per question, with answers in the Hugging Face SQuAD layout."""
    rows = []
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append({
                    "id": qa["id"],
                    "title": article["title"],
                    "context": paragraph["context"],
                    "question": qa["question"].strip(),  # some questions have long leading whitespace
                    "answers": {
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    },
                })
    return rows


def to_dataset(rows):
    from datasets import Dataset  # imported lazily so the split/check code needs only the stdlib
    return Dataset.from_list(rows)


def load_split(path):
    """Load a SQuAD-format file as a flattened datasets.Dataset."""
    return to_dataset(flatten(load_squad(path)))


def answer_mismatches(rows):
    """Rows where some answer text is not at its answer_start in the context."""
    bad = []
    for r in rows:
        for text, start in zip(r["answers"]["text"], r["answers"]["answer_start"]):
            if r["context"][start:start + len(text)] != text:
                bad.append((r["id"], text, start))
    return bad


def run_checks(full_train, train_split, val_split, dev):
    """The checks of section 3.6 of implementation_plan.md. Raises AssertionError on failure."""
    train_rows, val_rows, dev_rows = flatten(train_split), flatten(val_split), flatten(dev)

    n_full = len(flatten(full_train))
    assert len(train_rows) + len(val_rows) == n_full == 87_599, \
        f"train + val = {len(train_rows)} + {len(val_rows)}, expected {n_full} (87,599)"
    assert len(dev_rows) == 10_570, f"dev has {len(dev_rows)} questions, expected 10,570"

    train_titles = {a["title"] for a in train_split["data"]}
    val_titles = {a["title"] for a in val_split["data"]}
    overlap = train_titles & val_titles
    assert not overlap, f"articles in both train and val: {sorted(overlap)}"

    ids = [r["id"] for r in train_rows + val_rows + dev_rows]
    assert len(ids) == len(set(ids)), "duplicate question ids"

    for name, rows in [("train", train_rows), ("val", val_rows), ("dev", dev_rows)]:
        bad = answer_mismatches(rows)
        assert not bad, f"{name}: {len(bad)} answers not found at answer_start, e.g. {bad[:3]}"


def describe(name, squad):
    rows = flatten(squad)
    n_paragraphs = sum(len(a["paragraphs"]) for a in squad["data"])
    print(f"{name:<6} {len(squad['data']):>4} articles  {n_paragraphs:>6} paragraphs  {len(rows):>6} questions")


def main():
    full_train = load_squad(config.TRAIN_FILE)
    dev = load_squad(config.DEV_FILE)

    train_split, val_split = split_by_article(full_train)
    save_squad(train_split, config.TRAIN_SPLIT_FILE)
    save_squad(val_split, config.VAL_SPLIT_FILE)

    describe("train", train_split)
    describe("val", val_split)
    describe("dev", dev)
    print(f"wrote {config.TRAIN_SPLIT_FILE.relative_to(config.ROOT)} and "
          f"{config.VAL_SPLIT_FILE.relative_to(config.ROOT)}")

    run_checks(full_train, train_split, val_split, dev)
    print("all Phase 1 checks passed")


if __name__ == "__main__":
    main()
