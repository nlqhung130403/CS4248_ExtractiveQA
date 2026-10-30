# Extractive Question Answering on SQuAD

**CS4248 Introduction to Large Language Models — AY 2026/27 Semester 1, Final Project**

This project builds an **extractive question answering (QA) system**: given an English passage and a question about it, the system returns the answer as a contiguous segment of text (a *span*) copied directly from the passage. The system is trained and evaluated on the **Stanford Question Answering Dataset (SQuAD) v1.1** (Rajpurkar et al., 2016).

---

## 1. Problem definition

Reading comprehension is the ability to read a text and then answer questions about it. For a machine, this requires both an understanding of natural language and some knowledge about the world.

Formally, each example consists of:

| Input | Description |
|---|---|
| **Context** *P* | A paragraph from an English Wikipedia article (at least 500 characters) |
| **Question** *q* | A natural-language question written by a crowdworker about *P* |

| Output | Description |
|---|---|
| **Answer** *a* | A span `P[i : j]` of the context, identified by its start and end positions |

Unlike multiple-choice reading comprehension datasets, SQuAD provides **no list of candidate answers**. The system must choose among *all possible spans* in the passage, which for a passage of *L* tokens is O(*L*²) candidates.

### Example (from Figure 1 of the SQuAD paper)

> **Context:** In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of precipitation include drizzle, rain, sleet, snow, graupel and hail... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud. Short, intense periods of rain in scattered locations are called "showers".

| Question | Answer span |
|---|---|
| What causes precipitation to fall? | `gravity` |
| What is another main form of precipitation besides drizzle, rain, snow, sleet and hail? | `graupel` |
| Where do water droplets collide with ice crystals to form precipitation? | `within a cloud` |

To answer the first question, a system has to locate the relevant clause ("…falls under gravity"), recognise that "under" expresses a cause rather than a location, and return `gravity`.

## 2. Objective

The goal is to **implement a QA system that, for each (context, question) pair, predicts the answer span that best matches the human-annotated answers**, and to maximise two metrics on the SQuAD v1.1 development set (used as our test set):

1. **Exact Match (EM)**
2. **Macro-averaged F1**

As reference points, the original paper reports a gap between a feature-based model and human performance, which later neural models have steadily narrowed. This project aims to build a modern system that substantially closes that gap, and to analyse where it still falls short.

## 3. Data

Both files are provided in the course's Canvas Project folder.

| Split | File | Use in this project |
|---|---|---|
| Train | `train-v1.1.json` | Model training (and any held-out validation split we carve from it) |
| Test | `dev-v1.1.json` | Final evaluation only |

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

Both metrics normalise text first: lowercase, strip punctuation, remove the articles *a / an / the*, and collapse whitespace.

- **Exact Match (EM):** the percentage of predictions that exactly match *any one* of the reference answers.
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

| Method | Dev EM | Dev F1 | Test EM | Test F1 |
|---|---|---|---|---|
| Random guess | 1.1 | 4.1 | 1.3 | 4.3 |
| Sliding window | 13.2 | 20.2 | 12.5 | 19.7 |
| Sliding window + distance | 13.3 | 20.2 | 13.0 | 20.0 |
| Logistic regression (feature-based) | 40.0 | 51.0 | 40.4 | 51.0 |
| **Human** | **80.3** | **90.5** | **77.0** | **86.8** |

The paper also mentions that a neural model (Match-LSTM with Answer Pointer; Wang and Jiang, 2016) had already reached 70.3% F1 on v1.1. Since our test set is the official dev set, the **human dev scores (80.3 EM / 90.5 F1)** are the most directly comparable ceiling.

### Our results

| Model | EM | F1 |
|---|---|---|
| _TODO: baseline_ | | |
| _TODO: final system_ | | |

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

# 2. Train
# TODO: python src/train.py ...

# 3. Predict on the test set
# TODO: python src/predict.py --data data/dev-v1.1.json --out outputs/predictions.json

# 4. Evaluate
python evaluate-v2.0.py data/dev-v1.1.json outputs/predictions.json
```

## 9. Error analysis (planned)

Following the analysis in the SQuAD paper, we plan to break down performance by:

- **Answer type** (date, number, person, location, other entity, noun phrase, verb phrase, clause, …)
- **Reasoning type** (synonymy, world knowledge, syntactic variation, multi-sentence reasoning)
- **Syntactic divergence** between the question and the answer sentence
- **Span boundary errors** (e.g., including or omitting non-essential modifiers)

## References

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264.pdf). In *Proceedings of EMNLP 2016*, pages 2383–2392.
- Shuohang Wang and Jing Jiang. 2016. Machine Comprehension Using Match-LSTM and Answer Pointer. *CoRR*, abs/1608.07905.
- SQuAD website: https://stanford-qa.com