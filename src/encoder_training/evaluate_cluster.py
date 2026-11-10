import numpy as np
import torch
import collections
import importlib.util
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DATASET_EVAL_PATH = "/path/to/SQUAD/dev-v1.1.json"
EVAL_CODE_PATH = "/path/to/SQUAD/evaluate-v2.0.py"
MODEL_NAME = "bert-base-cased"

def load_official_eval_module(filepath):
    """Dynamically imports the official SQuAD evaluation script from a file path."""
    spec = importlib.util.spec_from_file_location("squad_eval", filepath)
    squad_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(squad_eval)
    return squad_eval


def predict_answer(context, question, model, tokenizer):
    device = "cpu"
    if torch.cuda.is_available() :
        device = "cuda"

    model.eval()

    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=384,
    )

    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    sequence_ids = inputs.sequence_ids(0)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities via Softmax
    start_probs = (
        torch.softmax(outputs["start_logits"], dim=-1)[0].cpu().numpy()
    )
    end_probs = torch.softmax(outputs["end_logits"], dim=-1)[0].cpu().numpy()

    # Mask non-context tokens (question, [CLS], [SEP], padding)
    for idx, seq_id in enumerate(sequence_ids):
        if seq_id != 1:  # 1 indicates context tokens
            start_probs[idx] = 0.0
            end_probs[idx] = 0.0

    # Find the best valid (start, end) pair where start <= end and span_len < 30
    best_score = 0.0
    best_start, best_end = 0, 0

    # Look at top-20 start and end positions
    start_candidates = np.argsort(start_probs)[-20:][::-1]
    end_candidates = np.argsort(end_probs)[-20:][::-1]

    for s in start_candidates:
        for e in end_candidates:
            if s <= e and (e - s) < 30:  # answers are rarely > 30 tokens
                score = start_probs[s] * end_probs[e]
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e

    # Map token positions directly to original context character spans
    char_start = offset_mapping[best_start][0]
    char_end = offset_mapping[best_end][1]

    # Slice the original raw text string
    predicted_text = context[char_start:char_end]
    return predicted_text


def evaluate_with_official_script(
    dataset, preds, na_probs=None, na_prob_thresh=1.0
):
    """Calls the official evaluation functions imported from EVAL_CODE_PATH."""
    if na_probs is None:
        na_probs = {k: 0.0 for k in preds}

    qid_to_has_ans = squad_eval.make_qid_to_has_ans(dataset)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

    # Official raw score computation
    exact_raw, f1_raw = squad_eval.get_raw_scores(dataset, preds)

    # Official threshold application
    exact_thresh = squad_eval.apply_no_ans_threshold(
        exact_raw, na_probs, qid_to_has_ans, na_prob_thresh
    )
    f1_thresh = squad_eval.apply_no_ans_threshold(
        f1_raw, na_probs, qid_to_has_ans, na_prob_thresh
    )

    out_eval = squad_eval.make_eval_dict(exact_thresh, f1_thresh)

    if has_ans_qids:
        has_ans_eval = squad_eval.make_eval_dict(
            exact_thresh, f1_thresh, qid_list=has_ans_qids
        )
        squad_eval.merge_eval(out_eval, has_ans_eval, "HasAns")
    if no_ans_qids:
        no_ans_eval = squad_eval.make_eval_dict(
            exact_thresh, f1_thresh, qid_list=no_ans_qids
        )
        squad_eval.merge_eval(out_eval, no_ans_eval, "NoAns")

    # Official optimal threshold calculation
    squad_eval.find_all_best_thresh(
        out_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans
    )

    return out_eval


# =========================================================================
# 3. POST-PROCESSING (Convert Logits to Answers & NA-Probabilities)
# =========================================================================
def postprocess_qa_predictions(
    examples,
    features,
    all_start_logits,
    all_end_logits,
    n_best_size=20,
    max_answer_length=30,
):
    """Extracts best span and calculates unanswerability probability (na_prob)."""
    sample_mapping = features["overflow_to_sample_mapping"].numpy()
    offset_mapping = features["offset_mapping"].numpy()

    example_to_features = collections.defaultdict(list)
    for idx, sample_idx in enumerate(sample_mapping):
        example_to_features[sample_idx].append(idx)

    predictions = collections.OrderedDict()
    na_probs = collections.OrderedDict()

    for example_idx, example in enumerate(tqdm(examples, desc="Post-processing")):
        feature_indices = example_to_features[example_idx]
        context = example["context"]
        qid = example["id"]

        min_null_score = float("inf")
        prelim_predictions = []

        for f_idx in feature_indices:
            start_logits = all_start_logits[f_idx]
            end_logits = all_end_logits[f_idx]
            offsets = offset_mapping[f_idx]
            seq_ids = features.sequence_ids(f_idx)

            # Logits for null-answer at CLS index (0)
            null_score = float(start_logits[0] + end_logits[0])
            if null_score < min_null_score:
                min_null_score = null_score

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]

            for s_idx in start_indexes:
                for e_idx in end_indexes:
                    if s_idx >= len(offsets) or e_idx >= len(offsets):
                        continue
                    if seq_ids[s_idx] != 1 or seq_ids[e_idx] != 1:
                        continue
                    if e_idx < s_idx or (e_idx - s_idx + 1) > max_answer_length:
                        continue

                    start_char = offsets[s_idx][0]
                    end_char = offsets[e_idx][1]
                    pred_text = context[start_char:end_char].strip()
                    score = float(start_logits[s_idx] + end_logits[e_idx])

                    prelim_predictions.append(
                        {"score": score, "text": pred_text}
                    )

        if prelim_predictions:
            best_pred = max(prelim_predictions, key=lambda x: x["score"])
            best_non_null_score = best_pred["score"]
            best_text = best_pred["text"]
        else:
            best_non_null_score = 0.0
            best_text = ""

        # Difference between null score and best candidate span
        score_diff = min_null_score - best_non_null_score

        # Map to [0, 1] probability using Sigmoid
        na_prob = 1.0 / (1.0 + math.exp(max(-700.0, min(700.0, -score_diff))))
        na_probs[qid] = na_prob

        # If null is more probable (score_diff > 0), set prediction to empty string
        predictions[qid] = best_text

    return predictions, na_probs


# =========================================================================
# 4. FULL EVALUATION PIPELINE
# =========================================================================
class EvalDataset(Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: self.encodings[key][idx] for key in ("input_ids", "attention_mask", "token_type_ids") if key in self.encodings}

    def __len__(self):
        return len(self.encodings["input_ids"])


def run_evaluation(model, tokenizer, dev_json_path, device="cuda", batch_size=32):
    # 1. Load data
    with open(dev_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)["data"]

    examples = []
    for article in raw_data:
        for p in article["paragraphs"]:
            ctx = p["context"]
            for qa in p["qas"]:
                examples.append(
                    {"id": qa["id"], "question": qa["question"], "context": ctx}
                )

    # 2. Tokenize with sliding windows
    print(f"Tokenizing {len(examples)} examples...")
    features = tokenizer(
        [e["question"].strip() for e in examples],
        [e["context"] for e in examples],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    loader = DataLoader(
        EvalDataset(features), batch_size=batch_size, shuffle=False
    )

    # 3. Model Inference
    model.to(device)
    model.eval()
    start_logits, end_logits = [], []

    print("Running model inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(start_logits, axis=0)
    all_end_logits = np.concatenate(end_logits, axis=0)

    # 4. Extract predictions
    preds, na_probs = postprocess_qa_predictions(
        examples, features, all_start_logits, all_end_logits
    )

    # 5. Evaluate using the imported official script functions
    print("Evaluating with official script...")
    results = evaluate_with_official_script(raw_data, preds, na_probs)

    print("\n========== Official SQuAD 2.0 Results ==========")
    print(json.dumps(results, indent=2))
    return results, preds, na_probs

if __name__ == "__main__" : 
    device = "cpu"
    if torch.cuda.is_available() :
        device = "cuda"
        
    squad_eval = load_official_eval_module(EVAL_CODE_PATH)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


    results, preds, na_probs = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        dev_json_path=DATASET_EVAL_PATH,
        device=device,
        batch_size=32,
    )

    