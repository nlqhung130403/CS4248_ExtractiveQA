# Extractive Question Answering on SQuAD

**CS4248 Introduction to Large Language Models — AY 2026/27 Semester 1, Final Project**

This project builds an extractive question answering (QA) system: given an English passage and a question about it, the task is to return the answer as a contiguous segment of text (a _span_) copied directly from the passage. We compare encoder-based models, generative prompting and LoRA adaptation on the **Stanford Question Answering Dataset (SQuAD) v1.1** (Rajpurkar et al., 2016).

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

We compare supervised encoders, generative prompting and LoRA fine-tuning under a shared evaluation. We also examine whether mapping generated answers to passage spans improves scores and how performance varies across question and answer categories.

## 3. Data

Both files are provided in the course's Canvas Project folder.

| Split | File              | Use in this project                                                 |
| ----- | ----------------- | ------------------------------------------------------------------- |
| Train | `train-v1.1.json` | Model training (and any held-out validation split we carve from it) |
| Test  | `dev-v1.1.json`   | Final evaluation only                                               |

**About SQuAD.** SQuAD v1.0 contains 107,785 question–answer pairs over 536 Wikipedia articles, drawn by random sampling from the top 10,000 English Wikipedia articles by PageRank. Articles were randomly partitioned 80% / 10% / 10% into train / dev / test, so **no article appears in more than one split**. Development questions may have multiple reference answers, making evaluation more robust to boundary disagreements.

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

`answer_start` is a **character offset** into `context`. Training examples have one answer; development examples may have multiple reference answers.

> ⚠️ Because `dev-v1.1.json` is our test set, it must not be used for model selection or hyperparameter tuning. Use a held-out portion of the training data for that.

## 4. Evaluation

We use the provided official **`evaluate-v2.0.py`** scorer:

- **Exact Match (EM):** percentage of predictions matching any reference after normalization.
- **Token F1:** best token-overlap F1 across references for each question, averaged over all questions.

Normalization lowercases text, removes punctuation and articles, and collapses whitespace.

### Shared evaluator

`src/evaluation-qa.py` accepts plain prediction maps (`{"question_id": "answer"}`) or bundles containing `predictions`. If present, `raw_predictions` is preferred. Each file must cover every dataset question with no extra IDs; filename stems must be unique.

The evaluator scores both supplied and span-mapped answers. Mapping cleans recognised prefixes and quotes, allows case/whitespace differences, and returns an empty answer when no passage match is found. It uses no gold answers and can lower scores by discarding partially correct outputs. **Raw** means before shared mapping, not necessarily before model-specific cleaning.

Main outputs:

- `comparison.csv`: overall raw/mapped scores and mapping diagnostics.
- `breakdowns.csv`: grouped scores and question counts.
- `<model>/per_question.csv`: predictions, references and individual scores.
- Prediction/metric JSON files, `annotation_template.csv` and `run_manifest.json` for reuse and reproducibility.

Overall scores are percentages; per-question scores range from 0 to 1. See Section 8 for evaluation commands and Section 9 for error analysis.

## 5. Reference results

Reported in the SQuAD paper (Table 5; numbers are on SQuAD v1.0, which the authors note give results similar to v1.1):

| Method                              | Dev EM   | Dev F1   | Test EM  | Test F1  |
| ----------------------------------- | -------- | -------- | -------- | -------- |
| Random guess                        | 1.1      | 4.1      | 1.3      | 4.3      |
| Sliding window                      | 13.2     | 20.2     | 12.5     | 19.7     |
| Sliding window + distance           | 13.3     | 20.2     | 13.0     | 20.0     |
| Logistic regression (feature-based) | 40.0     | 51.0     | 40.4     | 51.0     |
| **Human**                           | **80.3** | **90.5** | **77.0** | **86.8** |

The paper also mentions that a neural model (Match-LSTM with Answer Pointer; Wang and Jiang, 2016) had already reached 70.3% F1 on v1.1. These historical results provide context, not a strict performance ceiling or a directly controlled comparison with our SQuAD v1.1 runs.

### Our results

Scores on `dev-v1.1.json` (10,570 questions) before shared span mapping, using the official scorer. Supplied predictions may already include model-specific cleaning.

| Model                 | Setting                   |        EM |        F1 |
| --------------------- | ------------------------- | --------: | --------: |
| BERT-base-uncased     | Fine-tuned                |     78.27 |     86.19 |
| BERT-base-cased       | Fine-tuned                |     79.41 |     87.08 |
| DeBERTa-v3-base       | Fine-tuned                |     85.27 |     91.71 |
| ELECTRA-base          | Fine-tuned                | **85.87** |     92.24 |
| Qwen3.5-0.8B          | Zero-shot, 32-token setup |     61.24 |     77.30 |
| Qwen3.5-0.8B          | LoRA                      |     85.68 | **92.32** |
| Gemma 2 2B base       | Zero-shot                 |     41.84 |     59.32 |
| Gemma 2 2B base       | Two-shot                  |     68.19 |     79.37 |
| Gemma 2 2B Instruct   | Zero-shot                 |     71.33 |     85.89 |
| Llama 3.2 1B base     | Zero-shot                 |     38.58 |     52.32 |
| Llama 3.2 1B base     | Two-shot                  |     53.60 |     65.16 |
| Llama 3.2 1B Instruct | Zero-shot                 |     59.64 |     71.38 |

ELECTRA achieves the highest EM, while Qwen+LoRA achieves the highest F1. Their differences are small: 0.19 EM points in ELECTRA's favour and 0.08 F1 points in Qwen+LoRA's favour. No statistical significance tests are reported.

### Separate Qwen prompting experiment

This experiment uses different prompting, token budgets and post-processing from the Qwen zero-shot configuration above. Within this experiment, the zero-shot and three-shot runs share the same budgets. Three-shot prompting was selected on held-out validation data.

| Setting    |    EM |    F1 | Non-extractive rate (%) |
| ---------- | ----: | ----: | ----------------------: |
| Zero-shot  | 61.70 | 76.52 |                   22.40 |
| Three-shot | 67.81 | 80.79 |                   12.05 |

Only one fixed demonstration set was evaluated. Differences between the two separate Qwen zero-shot setups should not be attributed to prompt wording alone.

## 6. Approach

### Encoder models

We evaluate BERT, DeBERTa and ELECTRA models with a question-answering head that predicts start and end positions. Post-processing selects a valid span and copies it from the original passage.

The repository's training pipeline fine-tunes pretrained ELECTRA-base weights without prior QA fine-tuning. The final recipe uses a learning rate of `1e-4`, layer-wise learning-rate decay of `0.8`, and two epochs. Encoder training settings differ across runs.

### Generative prompting

Compact generative models receive an extraction instruction, the passage and the question, optionally preceded by fixed training demonstrations. Base and instruction-tuned Gemma/Llama variants are compared. A separate Qwen experiment evaluates zero-, one- and three-shot prompting on validation data before final evaluation of selected settings.

### LoRA adaptation

Qwen3.5-0.8B is adapted using LoRA with supervised question-answer examples. The model generates answers token by token while only its adapter parameters are updated. The project also uses a logit lens to inspect reference first-token information across intermediate layers.

## 7. Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── implementation_plan.md
├── evaluate-v2.0.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── postprocess.py
│   ├── subset_table.py
│   ├── generative_inference.py
│   ├── evaluation-qa.py
│   ├── annotations/
│   │   ├── annotations_100_draft.csv
│   │   ├── annotation_manifest.json
│   │   └── annotation_guide.txt
│   └── llm_training_peft/
│       ├── training.py
│       ├── evaluation.py
│       └── logit_lens_analysis.py
├── outputs/
│   ├── ELECTRA_log.txt
│   ├── comparison.csv
│   ├── breakdowns.csv
│   ├── annotation_template.csv
│   ├── predictions/
│   │   └── final_kaggle/
│   │       ├── val_split_predictions.json
│   │       └── dev-v1.1_predictions.json
│   ├── dev_gpu_strict_0shot/
│   ├── dev_gpu_strict_3shot/
│   ├── val_gpu_strict_0shot/
│   ├── val_gpu_strict_1shot/
│   └── val_gpu_strict_3shot/
└── tests/
    └── test_evaluation.py
```

Each `*_gpu_strict_*` directory contains `config.json`, `predictions.json`, `metrics.json`, `generations.jsonl` and `prompt_preview.txt`.

Dataset files and model checkpoints are not included in this listing. Place the Canvas files in `data/` to use the commands below, or adjust the paths to their actual locations.

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

# 4. Evaluate the new encoder predictions
python src/evaluation-qa.py --data data/dev-v1.1.json \
  --predictions outputs/predictions/final/dev-v1.1_predictions.json \
  --output outputs/evaluation
```

### Evaluate existing predictions with annotations

The following uses the included ELECTRA dev predictions and the repository annotation CSV:

```bash
python src/evaluation-qa.py \
  --data data/dev-v1.1.json \
  --predictions outputs/predictions/final_kaggle/dev-v1.1_predictions.json \
  --annotations src/annotations/annotations_100_draft.csv \
  --output outputs/evaluation_annotated
```

For a multi-model comparison, supply additional files after `--predictions`. Each file must cover the same questions and have a distinct filename stem.

### Optional spaCy answer-type tagging

If spaCy and an English NER model are installed in the active environment, add `--spacy-model en_core_web_sm` to the evaluation command. This changes answer-type labels and breakdowns, not overall EM/F1. The evaluator does not download a model automatically.

For generative prompting and LoRA training, refer to `src/generative_inference.py` and `src/llm_training_peft/`. Check their input, model and output paths before running.

### Integration tests

No model inference is required:

```bash
python -m unittest discover -s tests -v
```

## 9. Error analysis

The evaluator reports EM/F1 breakdowns by **question type, answer length, answer type and reasoning type**, alongside potential span-boundary errors, non-extractive outputs and mapping failures.

- **Answer types** use manual labels where available, otherwise optional spaCy tagging and date/number rules. Unclassified cases remain `unclassified`.
- **Reasoning types** use 100 randomly sampled development questions (seed 42), with AI-assisted labels reviewed by the team. Questions outside the labelled sample are `unannotated`.
- **Boundary flags** identify potential extra text, missing text or partial overlap and require manual interpretation.

Pass labels using `--annotations src/annotations/annotations_100_draft.csv`. The evaluator reads `id`, `answer_type` and `reasoning_type`; it does not filter by review status. Overall scores still cover the full dataset.

Results are saved in `breakdowns.csv` and `<model>/per_question.csv`. See [the annotation guide](src/annotations/annotation_guide.txt) for the rubric and review process.

Small reasoning groups support exploratory conclusions only. Syntactic divergence is not implemented.

## References

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264.pdf). In _Proceedings of EMNLP 2016_, pages 2383–2392.
- Shuohang Wang and Jing Jiang. 2016. Machine Comprehension Using Match-LSTM and Answer Pointer. _CoRR_, abs/1608.07905.
- [SQuAD website](https://stanford-qa.com)
