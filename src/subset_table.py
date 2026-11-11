"""Exact match (and F1) of one or more models on selected subsets of the dev set.

    python src/subset_table.py --data data/dev-v1.1.json \\
        --predictions ELECTRA=outputs/predictions/final/dev-v1.1_predictions.json [NAME=FILE ...]

Subsets:
- Question type: the first word of the question (what / how / when / why). "How many" and
  "how much" count as "how".
- Answer length: the shortest gold answer, in words after the official SQuAD normalisation
  (lowercase, no punctuation or articles), since EM/F1 are computed on normalised text and
  credit the best-matching reference.
"""

import argparse
import json
import re
from pathlib import Path

import data

QUESTION_WORDS = ["what", "how", "when", "why"]


def question_word(question):
    words = re.findall(r"\w+", question.lower())
    return words[0] if words else ""


def answer_length(answers, normalize):
    return min(len(normalize(a["text"]).split()) for a in answers)


def subsets(rows, normalize):
    """[(label, [question ids])] in table order."""
    groups = [(f"{w.capitalize()} questions", [r["id"] for r in rows if question_word(r["question"]) == w])
              for w in QUESTION_WORDS]
    lengths = {r["id"]: answer_length([{"text": t} for t in r["answers"]["text"]], normalize) for r in rows}
    groups.append(("One-word answers", [q for q, n in lengths.items() if n == 1]))
    groups.append(("Answers >5 words", [q for q, n in lengths.items() if n > 5]))
    groups.append(("All questions", [r["id"] for r in rows]))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/dev-v1.1.json"))
    parser.add_argument("--predictions", nargs="+", required=True, metavar="NAME=FILE")
    args = parser.parse_args()

    official = data.load_official_eval()
    squad = data.load_squad(args.data)
    rows = data.flatten(squad)
    models = []
    for item in args.predictions:
        name, _, path = item.partition("=")
        with open(path, encoding="utf-8") as f:
            exact, f1 = official.get_raw_scores(squad["data"], json.load(f))
        models.append((name, exact, f1))

    header = f"| {'Group':<20} | {'n':>6} |" + "".join(f" {name} EM | {name} F1 |" for name, _, _ in models)
    print(header)
    print("|" + "|".join("---" for _ in range(2 + 2 * len(models))) + "|")
    for label, ids in subsets(rows, official.normalize_answer):
        cells = "".join(f" {100 * sum(exact[q] for q in ids) / len(ids):.1f} |"
                        f" {100 * sum(f1[q] for q in ids) / len(ids):.1f} |" for _, exact, f1 in models)
        print(f"| {label:<20} | {len(ids):>6,} |{cells}")


if __name__ == "__main__":
    main()
