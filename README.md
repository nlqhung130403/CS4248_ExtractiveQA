# Extractive Question Answering on SQuAD

**CS4248 Introduction to Large Language Models — AY 2026/27 Semester 1, Final Project**

This project builds an **extractive question answering (QA) system**: given an English passage and a question about it, the system returns the answer as a contiguous segment of text (a _span_) copied directly from the passage. The system is trained and evaluated on the **Stanford Question Answering Dataset (SQuAD) v1.1** (Rajpurkar et al., 2016).

---

## 1. Problem definition

Reading comprehension is the ability to read a text and then answer questions about it. For a machine, this requires both an understanding of natural language and some knowledge about the world.

Formally, each example consists of:

| Input            | Description                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| **Context** _P_  | A paragraph from an English Wikipedia article (at least 500 characters) |
| **Question** _q_ | A natural-language question written by a crowdworker about _P_          |

| Output         | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| **Answer** _a_ | A span `P[i : j]` of the context, identified by its start and end positions |

Unlike multiple-choice reading comprehension datasets, SQuAD provides **no list of candidate answers**. The system must choose among _all possible spans_ in the passage, which for a passage of _L_ tokens is O(*L*²) candidates.

### Example (from Figure 1 of the SQuAD paper)

> **Context:** In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of precipitation include drizzle, rain, sleet, snow, graupel and hail... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud. Short, intense periods of rain in scattered locations are called "showers".

| Question                                                                                | Answer span      |
| --------------------------------------------------------------------------------------- | ---------------- |
| What causes precipitation to fall?                                                      | `gravity`        |
| What is another main form of precipitation besides drizzle, rain, snow, sleet and hail? | `graupel`        |
| Where do water droplets collide with ice crystals to form precipitation?                | `within a cloud` |

To answer the first question, a system has to locate the relevant clause ("…falls under gravity"), recognise that "under" expresses a cause rather than a location, and return `gravity`.

## 2. Objective

The goal is to **implement a QA system that, for each (context, question) pair, predicts the answer span that best matches the human-annotated answers**, and to maximise two metrics on the SQuAD v1.1 development set (used as our test set):

1. **Exact Match (EM)**
2. **Macro-averaged F1**

As reference points, the original paper reports a gap between a feature-based model and human performance, which later neural models have steadily narrowed. This project aims to build a modern system that substantially closes that gap, and to analyse where it still falls short.

## 3. Data

Both files are provided in the course's Canvas Project folder.

| Split | File              | Use in this project                                                 |
| ----- | ----------------- | ------------------------------------------------------------------- |
| Train | `train-v1.1.json` | Model training (and any held-out validation split we carve from it) |
| Test  | `dev-v1.1.json`   | Final evaluation only                                               |

**About SQuAD.** SQuAD v1.0 contains 107,785 question–answer pairs over 536 Wikipedia articles, drawn by random sampling from the top 10,000 English Wikipedia articles by PageRank. Articles were randomly partitioned 80% / 10% / 10% into train / dev / test, so **no article appears in more than one split**. Each question in the dev set has **at least three reference answers** (the original plus at least two additional crowdsourced answers), which makes evaluation more robust to boundary disagreements.

**File format.** Each JSON file has the structure:

```
{
  "version": "1.1",
  "data": [
    {
      "title": "...",
      "paragraphs": [
        {
          "context": "...",
          "qas": [
            {
              "id": "...",
              "question": "...",
              "answers": [ { "text": "...", "answer_start": 123 }, ... ]
            }
          ]
        }
      ]
    }
  ]
}
```

`answer_start` is a **character offset** into `context`. Training examples have one answer; dev examples have several.

> ⚠️ Because `dev-v1.1.json` is our test set, it must not be used for model selection or hyperparameter tuning. Use a held-out portion of the training data for that.

## 4. Evaluation

Evaluation uses the official script **`evaluate-v2.0.py`**, available from the CodaLab link in the project brief. SQuAD v1.1 contains no unanswerable questions, so the v2.0 script scores it with the same EM and F1 definitions.

Both metrics normalise text first: lowercase, strip punctuation, remove the articles _a / an / the_, and collapse whitespace.

- **Exact Match (EM):** the percentage of predictions that exactly match _any one_ of the reference answers.
- **F1 (macro-averaged):** for each question, treat the prediction and each reference as bags of tokens, compute token-level F1, take the **maximum** over references, then average over all questions.

### Running the scorer

Predictions must be a JSON object mapping each question `id` to the predicted answer string:

```json
{
  "56be4db0acb8001400a502ec": "Denver Broncos",
  "56be4db0acb8001400a502ed": "Carolina Panthers"
}
```

```bash
python evaluate-v2.0.py data/dev-v1.1.json outputs/predictions.json
```

## 5. Reference results

Reported in the SQuAD paper (Table 5; numbers are on SQuAD v1.0, which the authors note give results similar to v1.1):

| Method                              | Dev EM   | Dev F1   | Test EM  | Test F1  |
| ----------------------------------- | -------- | -------- | -------- | -------- |
| Random guess                        | 1.1      | 4.1      | 1.3      | 4.3      |
| Sliding window                      | 13.2     | 20.2     | 12.5     | 19.7     |
| Sliding window + distance           | 13.3     | 20.2     | 13.0     | 20.0     |
| Logistic regression (feature-based) | 40.0     | 51.0     | 40.4     | 51.0     |
| **Human**                           | **80.3** | **90.5** | **77.0** | **86.8** |

The paper also mentions that a neural model (Match-LSTM with Answer Pointer; Wang and Jiang, 2016) had already reached 70.3% F1 on v1.1. Since our test set is the official dev set, the **human dev scores (80.3 EM / 90.5 F1)** are the most directly comparable ceiling.

### Our results

| Model                | EM  | F1  |
| -------------------- | --- | --- |
| _TODO: baseline_     |     |     |
| _TODO: final system_ |     |     |

## 6. Approach

We fine-tune a pre-trained ELECTRA-base encoder on the SQuAD v1.1 training set, starting from the raw pre-trained weights (no prior QA fine-tuning).

**Backbone: ELECTRA-base**

## 7. Repository structure

```
.
├── README.md
├── data/
│   ├── train-v1.1.json
│   └── dev-v1.1.json
├── src/                 # TODO: training, inference and preprocessing code
├── outputs/
│   └── predictions.json
├── evaluate-v2.0.py
└── requirements.txt
```

## 8. Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Split the data and train (writes outputs/checkpoints/<run_name>/)
python src/data.py
python src/train.py --run_name final                       # ELECTRA paper recipe: LR 1e-4, LLRD 0.8
python src/train.py --run_name baseline --preset baseline  # LR 3e-5, no LLRD

# 3. Predict (on a GPU). Writes outputs/predictions/<run_name>/<data>_logits.npz and
#    <data>_predictions.json, and prints EM/F1. Use the validation split for all decisions.
python src/predict.py --model outputs/checkpoints/final --data data/splits/val_split.json
python src/predict.py --model outputs/checkpoints/final --data data/dev-v1.1.json   # final numbers only

# Optional: try post-processing settings on the saved scores (no model or GPU needed)
python src/postprocess.py outputs/predictions/final/val_split_logits.npz --data data/splits/val_split.json \
    --max_answer_length 15 30 50 --n_best 10 20 --score logits softmax

# 4. Evaluate with the official script
python evaluate-v2.0.py data/dev-v1.1.json outputs/predictions/final/dev-v1.1_predictions.json
```

## 9. Error analysis (planned)

Following the analysis in the SQuAD paper, we plan to break down performance by:

- **Answer type** (date, number, person, location, other entity, noun phrase, verb phrase, clause, …)
- **Reasoning type** (synonymy, world knowledge, syntactic variation, multi-sentence reasoning)
- **Syntactic divergence** between the question and the answer sentence
- **Span boundary errors** (e.g., including or omitting non-essential modifiers)

## References

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264.pdf). In _Proceedings of EMNLP 2016_, pages 2383–2392.
- Shuohang Wang and Jing Jiang. 2016. Machine Comprehension Using Match-LSTM and Answer Pointer. _CoRR_, abs/1608.07905.
- SQuAD website: https://stanford-qa.com

## Shared evaluation and span mapping

Run the shared evaluator on the same dataset for all models:

```bash
python src/evaluation-qa.py --data data/dev-v1.1.json \
  --predictions outputs/predictions.json runs/qwen-squad-lora/eval_lora.json \
  --output outputs/evaluation
```

Replace the prediction paths with your actual files. Each filename is used as the model name, so filenames must be unique.

The script uses the existing `evaluate-v2.0.py` without changing its scoring logic. It only requires NumPy and does not need model weights or a GPU.

**Supported inputs:**

- Plain JSON maps: `{"question_id": "answer"}`.
- LLM bundles: `{"metrics": {...}, "predictions": {...}}`.
- Newer LLM bundles also include `raw_predictions`, which are used for raw scoring to preserve the original generated text.

Older bundles only contain cleaned predictions, so their original generations cannot be recovered. The `run_manifest.json` records which field was used. Do not describe cleaned predictions as untouched generations.

**Evaluation and outputs:**

The evaluator checks question IDs, calculates raw scores using the official scorer, and separately scores answers mapped back to passage spans. Mapping handles trimmed answers, recognised answer prefixes and quotation marks, with case and whitespace differences allowed. Unmatched answers become empty strings and are counted as mapping failures. It does not recover arbitrary explanations or thinking text.

The main outputs are:

- `comparison.csv`: Overall raw/mapped EM and F1, non-extractive output rates and mapping failures.
- `breakdowns.csv`: Scores by question type, answer length, answerability and mapping method.
- `<model>/per_question.csv`: Individual predictions and scores for error analysis.
- `<model>/predictions.*.json` and `metrics.*.json`: Raw and mapped predictions and metrics.
- `run_manifest.json`: Input/scorer hashes, source fields and mapping policy.

Overall scores are percentages; per-question scores range from 0 to 1. Non-extractive output refers to answers that do not appear as case-sensitive substrings in the passage.

Choose mapping rules using held-out training data and keep the dev set for final evaluation.

**Run the integration tests** (no model inference required):

```bash
python -m unittest discover -s tests -v
```
