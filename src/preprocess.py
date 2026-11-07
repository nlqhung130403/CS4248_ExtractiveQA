"""Phase 2: turn (question, passage) pairs into fixed-length token windows.

Run from the repo root to tokenise the training split and check the windows:

    python src/preprocess.py
"""

from collections import Counter

from transformers import AutoTokenizer

import config
import data


def load_tokenizer(name=config.MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Only fast tokenizers return offset mappings, which we need to go between characters and tokens.
    assert tokenizer.is_fast, f"{name} did not load a fast tokenizer"
    return tokenizer


def tokenize_windows(tokenizer, questions, contexts,
                     max_length=config.MAX_LENGTH, doc_stride=config.DOC_STRIDE):
    """Tokenise question/passage pairs into overlapping windows of exactly max_length tokens.

    Each window is [CLS] question [SEP] passage-chunk [SEP] [PAD]...; the full question is repeated
    in every window and only the passage is split. One example can produce several windows:
    enc["overflow_to_sample_mapping"][i] is the index of the example that window i came from.
    """
    return tokenizer(
        questions, contexts,
        truncation="only_second",        # only ever truncate the passage, never the question
        max_length=max_length,
        stride=doc_stride,               # overlap between consecutive windows, in passage tokens
        return_overflowing_tokens=True,  # produce multiple windows per example
        return_offsets_mapping=True,     # (start_char, end_char) of each token
        padding="max_length",
    )


def passage_offsets(enc, i):
    """Character offsets of the passage tokens in window i."""
    return [off for off, seq in zip(enc["offset_mapping"][i], enc.sequence_ids(i)) if seq == 1]


def char_to_token_span(offsets, sequence_ids, start_char, end_char):
    """Token positions (start, end) of the passage tokens covering characters [start_char, end_char).

    offsets and sequence_ids are for one window. Returns None if the span is not fully inside
    this window's part of the passage.
    """
    passage = [i for i, seq in enumerate(sequence_ids) if seq == 1]
    first, last = passage[0], passage[-1]
    if offsets[first][0] > start_char or offsets[last][1] < end_char:
        return None

    start = first
    while offsets[start][1] <= start_char:  # first token that ends after start_char
        start += 1
    end = last
    while offsets[end][0] >= end_char:      # last token that starts before end_char
        end -= 1
    return start, end


def token_to_char_span(offsets, start, end):
    """Characters [start_char, end_char) covered by tokens start..end (inclusive) of one window."""
    return offsets[start][0], offsets[end][1]


CLS_INDEX = 0  # label for "the answer is not (fully) in this window"


def label_windows(enc, answer_starts, answer_texts):
    """Start/end token labels for every window. answer_starts[j] and answer_texts[j] are the
    (single) answer of example j; windows that do not contain the whole answer are labelled CLS_INDEX."""
    start_positions, end_positions = [], []
    for i, j in enumerate(enc["overflow_to_sample_mapping"]):
        start_char = answer_starts[j]
        end_char = start_char + len(answer_texts[j])  # exclusive
        span = char_to_token_span(enc["offset_mapping"][i], enc.sequence_ids(i), start_char, end_char)
        start, end = span if span is not None else (CLS_INDEX, CLS_INDEX)
        start_positions.append(start)
        end_positions.append(end)
    return start_positions, end_positions


def prepare_train_features(examples, tokenizer,
                           max_length=config.MAX_LENGTH, doc_stride=config.DOC_STRIDE):
    """Batched function for datasets.map: examples (columns of flattened rows) -> training windows
    with input_ids, token_type_ids, attention_mask, start_positions, end_positions."""
    enc = tokenize_windows(tokenizer, examples["question"], examples["context"], max_length, doc_stride)
    # Training examples have exactly one answer.
    enc["start_positions"], enc["end_positions"] = label_windows(
        enc,
        [a["answer_start"][0] for a in examples["answers"]],
        [a["text"][0] for a in examples["answers"]],
    )
    # The model does not take these as inputs.
    enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping")
    return enc


def prepare_eval_features(examples, tokenizer,
                          max_length=config.MAX_LENGTH, doc_stride=config.DOC_STRIDE):
    """Batched function for datasets.map: examples -> evaluation windows without labels.

    Besides the model inputs, each window keeps example_id (the SQuAD question id) and an
    offset_mapping in which every non-passage token is None, so post-processing can map
    predictions back to the passage and can never pick a question or special token.
    Drop these two columns before passing the windows to the model.
    """
    enc = tokenize_windows(tokenizer, examples["question"], examples["context"], max_length, doc_stride)
    sample_map = enc.pop("overflow_to_sample_mapping")
    enc["example_id"] = [examples["id"][j] for j in sample_map]
    enc["offset_mapping"] = [
        [off if seq == 1 else None for off, seq in zip(offsets, enc.sequence_ids(i))]
        for i, offsets in enumerate(enc["offset_mapping"])
    ]
    return enc


def build_features(dataset, tokenizer, train):
    """Apply prepare_train_features or prepare_eval_features to a flattened datasets.Dataset.

    The number of rows changes (one example can become several windows), so the example
    columns are removed.
    """
    fn = prepare_train_features if train else prepare_eval_features
    return dataset.map(fn, batched=True, fn_kwargs={"tokenizer": tokenizer},
                       remove_columns=dataset.column_names)


def check_windows(enc, max_length=config.MAX_LENGTH, doc_stride=config.DOC_STRIDE):
    """Every window has the right length and layout, and consecutive windows overlap by doc_stride."""
    sample_map = enc["overflow_to_sample_mapping"]
    for i, ids in enumerate(enc["input_ids"]):
        assert len(ids) == max_length, f"window {i} has {len(ids)} tokens"
        seq = enc.sequence_ids(i)
        assert seq[0] is None and 0 in seq and 1 in seq, f"window {i} is not [CLS] question [SEP] passage [SEP]"

        if i > 0 and sample_map[i] == sample_map[i - 1]:
            prev, cur = passage_offsets(enc, i - 1), passage_offsets(enc, i)
            assert prev[-doc_stride:] == cur[:doc_stride], f"windows {i - 1} and {i} do not overlap by {doc_stride}"


def check_labels(enc, batch, normalize):
    """Label every window with label_windows, then map the labels back to text.

    Returns (n_windows_with_answer, ids of examples whose answer is in no window,
    (id, gold, recovered) for mismatches after normalising, number of exact character matches).
    """
    starts, ends = label_windows(enc, [r["answers"]["answer_start"][0] for r in batch],
                                 [r["answers"]["text"][0] for r in batch])
    n_with_answer, n_exact, mismatches = 0, 0, []
    found = set()
    for i, j in enumerate(enc["overflow_to_sample_mapping"]):
        start, end = starts[i], ends[i]
        if start == end == CLS_INDEX:
            continue
        seq = enc.sequence_ids(i)
        assert start <= end and seq[start] == seq[end] == 1, f"window {i}: bad labels ({start}, {end})"
        n_with_answer += 1
        found.add(j)
        s, e = token_to_char_span(enc["offset_mapping"][i], start, end)
        r = batch[j]
        recovered, text = r["context"][s:e], r["answers"]["text"][0]
        n_exact += recovered == text
        if normalize(recovered) != normalize(text):
            mismatches.append((r["id"], text, recovered))
    missing = [batch[j]["id"] for j in range(len(batch)) if j not in found]
    return n_with_answer, missing, mismatches, n_exact


def main(batch_size=1000):
    rows = data.flatten(data.load_squad(config.TRAIN_SPLIT_FILE))
    tokenizer = load_tokenizer()
    normalize = data.load_official_eval().normalize_answer

    windows_per_example = Counter()
    question_lengths = []
    n_with_answer, n_exact, missing, mismatches = 0, 0, [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        enc = tokenize_windows(tokenizer, [r["question"] for r in batch], [r["context"] for r in batch])
        check_windows(enc)
        windows_per_example.update(start + j for j in enc["overflow_to_sample_mapping"])
        question_lengths += [len(ids) for ids in tokenizer([r["question"] for r in batch],
                                                           add_special_tokens=False)["input_ids"]]
        w, m, mm, ex = check_labels(enc, batch, normalize)
        n_with_answer, missing, mismatches, n_exact = n_with_answer + w, missing + m, mismatches + mm, n_exact + ex

    n_windows = sum(windows_per_example.values())
    print(f"{len(rows)} examples -> {n_windows} windows ({n_windows / len(rows):.3f} per example)")
    print("windows per example:", dict(sorted(Counter(windows_per_example.values()).items())))
    print(f"question length: max {max(question_lengths)} tokens, "
          f"mean {sum(question_lengths) / len(question_lengths):.1f}")
    print("all windows are", config.MAX_LENGTH, "tokens and consecutive windows overlap by",
          config.DOC_STRIDE, "passage tokens")

    print(f"\ntraining labels: {n_with_answer} windows labelled with the answer, "
          f"{n_windows - n_with_answer} labelled [CLS] ({100 * (n_windows - n_with_answer) / n_windows:.2f}%)")
    print(f"  examples whose answer is in no window: {len(missing)}")
    print("  labels mapped back to text through the offsets:")
    print(f"  recovered text identical to gold: {n_exact}/{n_with_answer} ({100 * n_exact / n_with_answer:.2f}%)")
    print(f"  same after official normalisation: {n_with_answer - len(mismatches)}/{n_with_answer} "
          f"({100 * (1 - len(mismatches) / n_with_answer):.2f}%)")
    for id_, gold, recovered in mismatches[:10]:
        print(f"    [{id_}] gold {gold!r} -> recovered {recovered!r}")

    # Show how the longest passage is split.
    longest = max(windows_per_example, key=windows_per_example.get)
    r = rows[longest]
    enc = tokenize_windows(tokenizer, [r["question"]], [r["context"]])
    print(f"\nexample {r['id']} ({len(r['context'])} chars) -> {len(enc['input_ids'])} windows")
    for i in range(len(enc["input_ids"])):
        offs = passage_offsets(enc, i)
        print(f"  window {i}: {len(offs)} passage tokens, chars {offs[0][0]}-{offs[-1][1]}, "
              f"starts '{tokenizer.decode(enc['input_ids'][i][:12])} ...'")

    # Build the actual features with datasets.map (4.6).
    print()
    train_features = build_features(data.to_dataset(rows), tokenizer, train=True)
    assert len(train_features) == n_windows
    n_cls = sum(s == CLS_INDEX for s in train_features["start_positions"])
    assert n_cls == n_windows - n_with_answer
    print(f"train features: {len(train_features)} rows, columns {train_features.column_names}")

    for name, path in [("val", config.VAL_SPLIT_FILE), ("dev", config.DEV_FILE)]:
        examples = data.load_split(path)
        features = build_features(examples, tokenizer, train=False)
        check_eval_features(examples, features)
        print(f"{name} features: {len(examples)} examples -> {len(features)} rows, "
              f"columns {features.column_names}")


def check_eval_features(examples, features):
    """Every example has at least one window, and only passage tokens keep an offset."""
    assert set(features["example_id"]) == set(examples["id"]), "some examples have no window"
    for i, (offsets, type_ids) in enumerate(zip(features["offset_mapping"], features["token_type_ids"])):
        # [CLS] and the question have token_type_id 0, so they must have no offset. Passage tokens
        # (type 1, except the final [SEP]) must keep one.
        assert offsets[0] is None, f"window {i}: [CLS] has an offset"
        assert all(o is None for o, t in zip(offsets, type_ids) if t == 0), f"window {i}: question token has an offset"
        assert any(o is not None for o in offsets), f"window {i}: no passage offsets"


if __name__ == "__main__":
    main()
