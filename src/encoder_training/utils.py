# utils.py
import importlib.util
import json
import re

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_squad_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []

    for article in squad_dict["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                for ans in qa["answers"]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(
                        {
                            "text": ans["text"],
                            "answer_start": ans["answer_start"],
                            "answer_end": ans["answer_start"]
                            + len(ans["text"]),
                        }
                    )

    return contexts, questions, answers


def preprocess_squad_examples(
    contexts, questions, answers, tokenizer, max_length=384, stride=128
):
    encodings = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",  # Truncate context only
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    offset_mapping = encodings.pop("offset_mapping")
    sample_mapping = encodings.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i].item()
        ans = answers[sample_idx]
        start_char = ans["answer_start"]
        end_char = start_char + len(ans["text"])

        # sequence_ids: None (special tokens), 0 (question), 1 (context)
        sequence_ids = encodings.sequence_ids(i)

        # Locate context boundaries inside input_ids
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        ctx_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        ctx_end = idx - 1

        # Check if answer is outside this sliding window
        if not (
            offsets[ctx_start][0] <= start_char and offsets[ctx_end][1] >= end_char
        ):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Map start_char to token index
            cur_start = ctx_start
            while cur_start <= ctx_end and offsets[cur_start][0] <= start_char:
                cur_start += 1
            start_positions.append(cur_start - 1)

            # Map end_char to token index
            cur_end = ctx_end
            while cur_end >= ctx_start and offsets[cur_end][1] >= end_char:
                cur_end -= 1
            end_positions.append(cur_end + 1)

    encodings["start_positions"] = torch.tensor(start_positions)
    encodings["end_positions"] = torch.tensor(end_positions)

    return encodings


class UniversalQADataset(Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)




def load_squad_eval_data(filepath: str):
    """Loads SQuAD JSON and returns raw data alongside a list of test examples."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)["data"]

    examples = []
    for article in raw_data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                })
    return raw_data, examples


def clean_prediction(text: str) -> str:
    """Strips thinking blocks (<think>...</think>), leading prefixes, and extra lines."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^(Answer:|The answer is:?)\s*", "", text, flags=re.IGNORECASE)
    return text.strip().split("\n")[0].strip()


def load_official_eval_module(filepath: str):
    """Dynamically imports the official SQuAD evaluate-v2.0.py script."""
    spec = importlib.util.spec_from_file_location("squad_eval", filepath)
    squad_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(squad_eval)
    return squad_eval
