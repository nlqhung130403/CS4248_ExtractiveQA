"""Phase 4: turn the model's start/end scores into one answer string per question.

predict.py runs the model and saves its raw scores to <name>_logits.npz. This script reads that
file, picks the best span for every question, writes {question_id: answer} JSON and prints EM/F1.
It needs no model and no GPU, so trying different settings takes seconds:

    python src/postprocess.py outputs/predictions/final/val_split_logits.npz --data data/splits/val_split.json
    # Several values per setting: score every combination and print a table (no file written)
    python src/postprocess.py outputs/predictions/final/val_split_logits.npz --data data/splits/val_split.json \\
        --max_answer_length 15 30 50 --n_best 10 20 --score logits softmax

Check the selection and offset code without a model: put the highest score on the gold answer's
tokens and confirm the gold text comes back (EM should be close to 100):

    python src/postprocess.py --oracle --data data/splits/val_split.json
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
import data

N_BEST = 20             # top start and end positions considered per window
MAX_ANSWER_LENGTH = 30  # longest allowed answer, in tokens
SCORE = "logits"        # "logits": start + end logit; "softmax": the same in log-probabilities per window
NO_OFFSET = -1          # offsets entry of a question, special or padding token


# ---------------------------------------------------------------------------------------------
# Saved model outputs
# ---------------------------------------------------------------------------------------------

def offsets_array(offset_mapping):
    """Eval-feature offset_mapping (None for non-passage tokens) -> int16 array (windows, length, 2),
    NO_OFFSET for non-passage tokens. SQuAD passages are at most ~4,100 characters, so int16 fits."""
    arr = np.array([[o if o is not None else (NO_OFFSET, NO_OFFSET) for o in window]
                    for window in offset_mapping], dtype=np.int32)
    assert arr.max() < np.iinfo(np.int16).max, "passage too long for int16 offsets"
    return arr.astype(np.int16)


def save_logits(path, start_logits, end_logits, offsets, example_ids, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, start_logits=start_logits.astype(np.float32),
                        end_logits=end_logits.astype(np.float32), offsets=offsets,
                        example_id=np.array(example_ids), meta=json.dumps(meta))


def load_logits(path):
    f = np.load(path)
    return {"start_logits": f["start_logits"], "end_logits": f["end_logits"], "offsets": f["offsets"],
            "example_id": f["example_id"].tolist(), "meta": json.loads(str(f["meta"]))}


# ---------------------------------------------------------------------------------------------
# Span selection (section 6 of implementation_plan.md)
# ---------------------------------------------------------------------------------------------

def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def best_span_in_window(start, end, offsets, n_best=N_BEST, max_answer_length=MAX_ANSWER_LENGTH,
                        score=SCORE):
    """Best (score, start_token, end_token) of one window, or None if it has no valid span.

    Only passage tokens can start or end an answer: the others are masked out before taking the
    top n_best start and end positions, so they never use up candidate slots.
    """
    passage = offsets[:, 0] != NO_OFFSET
    if score == "softmax":
        # Normalise over the passage tokens plus [CLS] (position 0). [CLS] is the "answer is not in
        # this window" label from training, so a window that does not contain the answer gives its
        # spans low probability. This makes scores comparable across windows.
        keep = passage.copy()
        keep[0] = True
        start, end = start.astype(np.float64), end.astype(np.float64)
        start[keep], end[keep] = log_softmax(start[keep]), log_softmax(end[keep])
    elif score != "logits":
        raise ValueError(f"unknown score {score!r}")
    start = np.where(passage, start, -np.inf)
    end = np.where(passage, end, -np.inf)

    k = min(n_best, int(passage.sum()))
    if k == 0:
        return None
    # Stable sort: equal scores (common with fp16 logits) keep token order, so ties go to the earlier
    # token and the answer does not depend on the machine's numpy version.
    starts = np.argsort(-start, kind="stable")[:k]
    ends = np.argsort(-end, kind="stable")[:k]
    scores = start[starts][:, None] + end[ends][None, :]
    length = ends[None, :] - starts[:, None] + 1
    scores = np.where((length >= 1) & (length <= max_answer_length), scores, -np.inf)

    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    if not np.isfinite(scores[i, j]):
        return None
    return float(scores[i, j]), int(starts[i]), int(ends[j])


def select_answers(logits, contexts, n_best=N_BEST, max_answer_length=MAX_ANSWER_LENGTH, score=SCORE):
    """{question_id: answer text}, taking the best span across all windows of each question.

    logits is the dict from load_logits; contexts maps question id -> passage. A question with no
    valid span in any window (should not happen) gets "".
    """
    windows = defaultdict(list)
    for w, qid in enumerate(logits["example_id"]):
        windows[qid].append(w)

    answers = {}
    for qid, ws in windows.items():
        best = None  # (score, window, start_token, end_token)
        for w in ws:
            span = best_span_in_window(logits["start_logits"][w], logits["end_logits"][w],
                                       logits["offsets"][w], n_best, max_answer_length, score)
            if span is not None and (best is None or span[0] > best[0]):
                best = (span[0], w, span[1], span[2])
        if best is None:
            answers[qid] = ""
            continue
        _, w, s, e = best
        offsets = logits["offsets"][w]
        answers[qid] = contexts[qid][offsets[s][0]:offsets[e][1]]
    return answers


# ---------------------------------------------------------------------------------------------
# Scoring and checks
# ---------------------------------------------------------------------------------------------

def evaluate(squad, predictions):
    """Official SQuAD EM and F1 (0-100) of {question_id: answer} on a SQuAD-format dict."""
    official = data.load_official_eval()
    exact, f1 = official.get_raw_scores(squad["data"], predictions)
    result = official.make_eval_dict(exact, f1)
    return result["exact"], result["f1"]


def check_predictions(predictions, rows):
    """One non-empty answer for every question in the data file. Returns the number of empty answers."""
    ids = {r["id"] for r in rows}
    missing, extra = ids - predictions.keys(), predictions.keys() - ids
    assert not missing, f"{len(missing)} questions have no prediction, e.g. {sorted(missing)[:3]}"
    assert not extra, f"{len(extra)} predictions are for unknown questions, e.g. {sorted(extra)[:3]}"
    return sum(not a.strip() for a in predictions.values())


def oracle_logits(examples, features):
    """Fake model outputs that put a high score on the first gold answer's tokens (or [CLS] in
    windows that do not contain it), in the same layout as predict.py's output."""
    import preprocess

    offsets = offsets_array(features["offset_mapping"])
    example_ids = features["example_id"]
    gold = {r["id"]: (r["answers"]["answer_start"][0], r["answers"]["text"][0]) for r in examples}
    start_logits = np.zeros(offsets.shape[:2], dtype=np.float32)
    end_logits = np.zeros(offsets.shape[:2], dtype=np.float32)
    for w, qid in enumerate(example_ids):
        start_char, text = gold[qid]
        sequence_ids = [1 if o != NO_OFFSET else None for o in offsets[w, :, 0]]
        span = preprocess.char_to_token_span(offsets[w].tolist(), sequence_ids, start_char,
                                             start_char + len(text))
        s, e = span if span is not None else (preprocess.CLS_INDEX, preprocess.CLS_INDEX)
        start_logits[w, s] = end_logits[w, e] = 10.0
    return {"start_logits": start_logits, "end_logits": end_logits, "offsets": offsets,
            "example_id": example_ids, "meta": {"model": "oracle"}}


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii (the default) keeps the file readable by evaluate-v2.0.py on Windows (see data.save_squad).
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def predictions_path(logits_path):
    """outputs/predictions/final/val_split_logits.npz -> .../val_split_predictions.json"""
    logits_path = Path(logits_path)
    return logits_path.with_name(logits_path.name.removesuffix("_logits.npz") + "_predictions.json")


def run(logits, squad, n_bests, max_lengths, scores, out=None):
    """Score every combination of settings; write predictions if there is only one combination."""
    rows = data.flatten(squad)
    assert set(logits["example_id"]) == {r["id"] for r in rows}, \
        "the logits file and --data have different questions"
    contexts = {r["id"]: r["context"] for r in rows}

    combos = list(itertools.product(n_bests, max_lengths, scores))
    if len(combos) > 1:
        print(f"{'n_best':>6} {'max_len':>7} {'score':>8} {'EM':>6} {'F1':>6}")
    for n_best, max_len, score in combos:
        predictions = select_answers(logits, contexts, n_best, max_len, score)
        n_empty = check_predictions(predictions, rows)
        em, f1 = evaluate(squad, predictions)
        if len(combos) > 1:
            print(f"{n_best:>6} {max_len:>7} {score:>8} {em:6.2f} {f1:6.2f}")
            continue
        print(f"n_best {n_best}, max_answer_length {max_len}, score {score}: EM {em:.2f}  F1 {f1:.2f}"
              f"  ({len(predictions)} questions, {n_empty} empty answers)")
        if out is not None:
            write_json(predictions, out)
            print(f"wrote {out}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logits", nargs="?", type=Path, help="<name>_logits.npz written by predict.py")
    parser.add_argument("--data", type=Path, required=True, help="SQuAD-format file the logits were computed on")
    parser.add_argument("--out", type=Path, help="predictions JSON (default: next to the logits file)")
    parser.add_argument("--n_best", type=int, nargs="+", default=[N_BEST])
    parser.add_argument("--max_answer_length", type=int, nargs="+", default=[MAX_ANSWER_LENGTH])
    parser.add_argument("--score", nargs="+", default=[SCORE], choices=["logits", "softmax"])
    parser.add_argument("--oracle", action="store_true",
                        help="use fake logits that point at the gold answer instead of a logits file")
    args = parser.parse_args()

    squad = data.load_squad(args.data)
    if args.oracle:
        import preprocess
        examples = data.load_split(args.data)
        features = preprocess.build_features(examples, preprocess.load_tokenizer(), train=False)
        logits, out = oracle_logits(examples, features), args.out
    else:
        if args.logits is None:
            parser.error("give a logits file, or --oracle")
        logits, out = load_logits(args.logits), args.out or predictions_path(args.logits)
    run(logits, squad, args.n_best, args.max_answer_length, args.score, out)


if __name__ == "__main__":
    main()
