# Implementation Plan: Fine-tuning ELECTRA-base on SQuAD v1.1

This document explains **what** we will build, **why** each piece exists, and **in what order** to build it. It assumes you know basic ML (training/validation/test splits, loss functions, gradient descent, overfitting) but not necessarily Transformers or question answering. Terms in **bold** are explained in the [Glossary](#12-glossary) at the end.

---

## 0. The big picture (read this first)

We want a program that reads a passage and a question and returns the answer as a piece of text copied from the passage.

Instead of training a model from scratch, we take **ELECTRA-base**, a Transformer that Google already **pre-trained** on a large amount of English text. It already "understands" language reasonably well, but it has never seen a QA task. We add a tiny new layer on top of it (the **QA head**) and train the whole thing on SQuAD's ~88k question–answer pairs. This is called **fine-tuning**.

The model does not generate text. For every token in the passage, it outputs two numbers:

- a **start score**: how likely is this token to be the *first* token of the answer?
- an **end score**: how likely is this token to be the *last* token of the answer?

We then pick the (start, end) pair with the highest combined score and copy out the text between them.

```
           ┌──────────────────────────────────────────────────────────────┐
 question ─┤                                                              │
           │  [CLS] question tokens [SEP] passage tokens [SEP]            │
 passage ──┤                     │                                        │
           └─────────────────────┼────────────────────────────────────────┘
                                 ▼
                     ELECTRA-base encoder (12 layers, pre-trained)
                                 │  one 768-dim vector per token
                                 ▼
                     QA head: Linear(768 → 2)   ← the only brand-new part
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             start score per token     end score per token
                    └────────────┬────────────┘
                                 ▼
             pick best valid (start, end) pair → map back to characters
                                 ▼
                         "Denver Broncos"
```

The full pipeline we need to write:

| Step | What it does | File |
|---|---|---|
| 1. Load & split data | Read the JSON, flatten it, hold out a validation set | `src/data.py` |
| 2. Preprocess | Turn text into token IDs and turn character-level answers into token positions | `src/preprocess.py` |
| 3. Train | Fine-tune ELECTRA with the QA head | `src/train.py` |
| 4. Post-process | Turn start/end scores back into answer strings | `src/postprocess.py` |
| 5. Predict | Run the trained model on a dataset and write `predictions.json` | `src/predict.py` |
| 6. Evaluate | Score predictions with the official script | `evaluate-v2.0.py` (given) |
| 7. Analyse errors | Break down where the model fails | `src/analysis/` |

---

## 1. Environment and compute

### 1.1 Hardware

Fine-tuning ELECTRA-base needs a **GPU**. The machine this repo currently lives on has a CPU-only PyTorch install, so training there would take days. Options:

| Option | GPU | Rough time for 2 epochs | Notes |
|---|---|---|---|
| Google Colab (free/Pro) | T4 / L4 / A100 | ~2–4 h on T4, < 1 h on A100 | Sessions can disconnect, so **save checkpoints to Google Drive** |
| Kaggle Notebooks | T4 / P100 | ~2–4 h | 30 GPU-hours/week free |
| School GPU cluster (e.g. NUS SoC Compute Cluster) | varies | varies | Best for long or repeated runs; submit as batch jobs |

The times above are rough estimates. Measure your own speed on the first run.

Develop and debug **locally on CPU using a tiny subset** (e.g. 200 examples), then move to a GPU only for real runs. This saves a lot of GPU time.

### 1.2 Software

Create `requirements.txt`:

```
torch>=2.1
transformers>=4.40
datasets>=2.18
accelerate>=0.30      # required by the Hugging Face Trainer
numpy
pandas                # for error analysis tables
matplotlib            # for plots in the report
spacy                 # for answer-type analysis (optional, Phase 6)
tqdm
```

We use the **Hugging Face** libraries:
- `transformers`: gives us the pre-trained ELECTRA model, its **tokenizer**, and a `Trainer` class that runs the training loop for us.
- `datasets`: a convenient table-like container that can apply preprocessing quickly and cache the result.

### 1.3 Which checkpoint?

ELECTRA is released as two models: a small **generator** and the main **discriminator**. **Always fine-tune the discriminator:**

```
google/electra-base-discriminator
```

When you load it with `AutoModelForQuestionAnswering`, you will see a warning like *"Some weights of ElectraForQuestionAnswering were not initialized … qa_outputs.weight, qa_outputs.bias"*. **This is expected.** It means the QA head is new and random, which is exactly what we want. It matches the README's requirement of starting from raw pre-trained weights.

---

## 2. Target repository layout

```
.
├── README.md
├── implementation_plan.md
├── requirements.txt
├── data/
│   ├── train-v1.1.json          # given
│   ├── dev-v1.1.json            # given — TEST SET, do not tune on it
│   └── splits/                  # created by src/data.py
│       ├── train_split.json     # train minus held-out articles
│       └── val_split.json       # held-out articles, in SQuAD format
├── src/
│   ├── config.py                # all hyperparameters in one place
│   ├── data.py                  # load, flatten, split
│   ├── preprocess.py            # tokenisation + label creation
│   ├── postprocess.py           # logits → answer strings
│   ├── train.py                 # fine-tuning entry point
│   ├── predict.py               # inference entry point
│   └── analysis/
│       ├── error_analysis.py
│       └── annotate_sample.csv  # manual annotations (Phase 6)
├── outputs/
│   ├── checkpoints/             # saved models (git-ignored, they're ~420 MB each)
│   ├── predictions.json
│   └── logs/
├── evaluate-v2.0.py
└── .gitignore                   # ignore outputs/checkpoints, __pycache__, etc.
```

---

## 3. Phase 1: Data loading and the validation split

### 3.1 What the data looks like

Both files are a single JSON object, nested three levels deep: **article → paragraph → question**.

```
{
  "version": "1.1",
  "data": [                                   # list of articles (one per Wikipedia page)
    {
      "title": "University_of_Notre_Dame",
      "paragraphs": [                         # list of paragraphs from that article
        {
          "context": "Architecturally, the school has a Catholic character. ...",
          "qas": [                            # list of questions about this paragraph
            {
              "id": "5733be284776f41900661182",
              "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
              "answers": [                    # list of reference answers
                {"answer_start": 515, "text": "Saint Bernadette Soubirous"}
              ]
            },
            ...
          ]
        },
        ...
      ]
    },
    ...
  ]
}
```

| Field | Meaning |
|---|---|
| `title` | Wikipedia article name. Used to split by article (3.3). |
| `context` | The passage. The answer is always a substring of it. |
| `id` | Unique question id. Keys of `predictions.json` must be these ids. |
| `question` | The question text. |
| `answers[].text` | The answer, copied exactly from `context`. |
| `answers[].answer_start` | **Character** index in `context` where the answer begins. The end is `answer_start + len(text)` (exclusive). |

| File | Articles | Paragraphs | Questions | Answers per question |
|---|---|---|---|---|
| `train-v1.1.json` | 442 | 18,896 | 87,599 | always 1 |
| `dev-v1.1.json` | 48 | 2,067 | 10,570 | 1–6, usually 3 (may contain duplicates) |

Each paragraph has several questions (about 5 on average), and the same paragraph text is repeated for each of them once the data is flattened (3.5).

### 3.2 Why we need our own validation split

`dev-v1.1.json` is our **test set**. If we picked hyperparameters by looking at its score, our final number would be optimistically biased (we would have "trained on the test set" indirectly). So we carve a **validation set** out of the training data and use *that* for every decision.

### 3.3 Split by article, not by question

SQuAD groups many questions around the same Wikipedia article. If questions from one article landed in both train and validation, the model would have already read that passage during training, and validation scores would look better than they really are. The original SQuAD splits are by article, so we do the same:

- The training file has **442 articles / 87,599 questions**.
- Randomly pick **~5% of articles (about 22)** for validation. With seed `42` this gives **22 articles / 3,789 questions** for validation and 420 articles / 83,810 questions for training, which is enough for stable estimates.
- Use a fixed random seed (e.g. `42`) so everyone on the team gets the same split.

### 3.4 Save the validation split in SQuAD format

Write `val_split.json` with **exactly the same JSON structure** as the original files. Then the official `evaluate-v2.0.py` works on it unchanged, and we score validation and test with the same code.

### 3.5 Flattening

The nested JSON (article → paragraphs → questions) is awkward to work with. Flatten it into one row per question:

| id | title | context | question | answers |
|---|---|---|---|---|
| `5733be28…` | University_of_Notre_Dame | "Architecturally, the school…" | "To whom did the Virgin Mary…" | `{"text": ["Saint Bernadette Soubirous"], "answer_start": [515]}` |

Then wrap it in a `datasets.Dataset` (`Dataset.from_list(rows)`).

Small cleaning step: some SQuAD questions start with lots of spaces, which waste tokens. Apply `question.strip()`.

### 3.6 Checks before moving on
- [ ] Row counts: train + val = 87,599; dev = 10,570.
- [ ] No article title appears in both train_split and val_split.
- [ ] For every training row, `context[answer_start : answer_start + len(text)] == text`.

---

## 4. Phase 2: Preprocessing (the trickiest part)

Most bugs in QA projects happen here, so take it slowly and test it well.

### 4.1 Tokenisation

The model cannot read characters. The **tokenizer** splits text into **sub-word tokens** and maps each to an integer ID. For example, "graupel" might become `["gr", "##au", "##pel"]`. The question and passage are packed into a single input:

```
[CLS] what causes precipitation to fall ? [SEP] in meteorology , precipitation is … [SEP]
```

Load the tokenizer with `AutoTokenizer.from_pretrained("google/electra-base-discriminator")`. It must be a **fast** tokenizer (the default), because we need **offset mappings** (below).

### 4.2 Long passages: sliding windows

The model accepts a limited number of tokens. We use `max_length = 384`. Some passages are longer than that. We do **not** just cut off the end, because the answer might be there. Instead we split the passage into overlapping **windows** (also called *features* or *chunks*):

```
passage tokens:  |----------------------------------------------------|
window 1:        |=========================|
window 2:                      |=========================|
window 3:                                    |=====================|
                               ↑ overlap of 128 tokens (doc_stride) ↑
```

- `max_length = 384`: total tokens per window (question + passage + special tokens)
- `doc_stride = 128`: how many tokens consecutive windows **overlap**, so an answer near a boundary appears whole in at least one window

The tokenizer does this for us:

```python
enc = tokenizer(
    questions, contexts,
    truncation="only_second",       # only ever truncate the passage, never the question
    max_length=384,
    stride=128,
    return_overflowing_tokens=True, # produce multiple windows per example
    return_offsets_mapping=True,    # char positions of each token (see 4.3)
    padding="max_length",
)
# enc["overflow_to_sample_mapping"][i] tells you which original example window i came from
```

So **one example can become several features**. Keep this in mind everywhere after this point.

### 4.3 Offset mappings: going between characters and tokens

SQuAD gives answers as **character** positions (`answer_start = 515`). The model predicts **token** positions. The **offset mapping** is a list that says, for each token, which characters `(start_char, end_char)` of the original text it covers. We use it in both directions:

- **Training:** character answer → token start/end labels
- **Prediction:** predicted token start/end → character span → answer text

Use `enc.sequence_ids(i)` to tell which tokens belong to the question (`0`), the passage (`1`), or are special tokens (`None`). Only passage tokens can be part of an answer.

### 4.4 Creating training labels

For each training window:

1. Find the answer's character span: `[answer_start, answer_start + len(answer_text))`.
2. Find the first and last **passage** tokens in this window.
3. If the answer is **not fully inside** this window, set both labels to `0` (the `[CLS]` position). This tells the model "the answer isn't in this window".
4. Otherwise, walk through the offsets to find the token where the answer starts (`start_positions`) and the token where it ends (`end_positions`).

### 4.5 Creating evaluation features

For validation/test windows we do **not** create labels. Instead we keep what we need to rebuild the answer later:

- `example_id`: the SQuAD question id this window came from (via `overflow_to_sample_mapping`)
- `offset_mapping`: with question/special-token offsets set to `None`, so they can never be picked as an answer

### 4.6 Apply with `datasets.map`

```python
train_features = train_ds.map(prepare_train_features, batched=True,
                              remove_columns=train_ds.column_names)
```

`batched=True` matters for speed. `remove_columns` is needed because the number of rows changes (examples → windows).

### 4.7 Checks before moving on (do not skip)
- [ ] **Round-trip test:** for ~1,000 training windows that have a non-zero label, decode the tokens between `start_positions` and `end_positions` and compare to the gold answer after normalisation (lowercase, strip punctuation). **Over 99% should match.** The rare mismatches come from tokenisation quirks; look at a few to make sure that's the cause.
- [ ] Print the number of features: expect a bit more than the number of examples (most passages fit in one window).
- [ ] Print the fraction of training windows labelled `0`/`[CLS]`: it should be small (a few percent).

---

## 5. Phase 3: Training

### 5.1 The model and the loss

```python
model = AutoModelForQuestionAnswering.from_pretrained("google/electra-base-discriminator")
```

For each window the model outputs `start_logits` and `end_logits` (one score per token). The loss is:

```
loss = ( CrossEntropy(start_logits, start_positions)
       + CrossEntropy(end_logits,   end_positions) ) / 2
```

In plain words: treat "which token is the start?" as a classification over all 384 positions, do the same for the end, and average the two losses. The Hugging Face model computes this automatically when you pass `start_positions` and `end_positions`.

### 5.2 Hyperparameters

We take the ELECTRA paper's (Clark et al., 2020) SQuAD fine-tuning settings for the Base model as our starting point:

| Hyperparameter | Value | What it means |
|---|---|---|
| Learning rate | `1e-4` (top layer) | Step size for weight updates |
| **Layer-wise LR decay** | `0.8` | Lower layers get smaller learning rates (see 5.3) |
| Epochs | `2` | Passes over the training data |
| Batch size | `32` | Windows per update (use gradient accumulation if it doesn't fit, see 5.4) |
| Warmup | first `10%` of steps | LR rises from 0 to its peak, then decays linearly to 0 |
| Optimizer | AdamW, `eps=1e-6`, betas `(0.9, 0.999)` | |
| Weight decay | `0` | |
| Dropout | `0.1` | Default in the model config |
| `max_length` / `doc_stride` | `384` / `128` | See 4.2 |
| Mixed precision | `fp16=True` (or `bf16` on A100/L4) | Roughly 2× faster, less memory |
| Seed | `42` | For reproducibility |

Put all of these in `src/config.py` (or a YAML file) so every run is fully described by its config. **Log the config with every run.**

### 5.3 Layer-wise learning-rate decay (LLRD), explained

The lower layers of a pre-trained Transformer learn general language features that are already useful. The top layers are more task-specific. LLRD gives each layer its own learning rate: the QA head gets the full LR, and each layer below gets 0.8× the LR of the one above. This changes the bottom layers less, which makes fine-tuning more stable and usually more accurate. It is the main reason the ELECTRA paper can use a fairly high LR like `1e-4`.

```
QA head          lr = 1e-4
layer 11         lr = 1e-4 × 0.8
layer 10         lr = 1e-4 × 0.8²
...
layer 0          lr = 1e-4 × 0.8¹²
embeddings       lr = 1e-4 × 0.8¹³
```

Implementation sketch: build AdamW **parameter groups** by parameter name, then hand the optimizer to the `Trainer`:

```python
def llrd_param_groups(model, base_lr=1e-4, decay=0.8, n_layers=12):
    groups = []
    for name, param in model.named_parameters():
        if name.startswith("electra.embeddings"):
            depth = 0
        elif name.startswith("electra.encoder.layer."):
            depth = int(name.split(".")[3]) + 1          # layer i → depth i+1
        else:                                            # qa_outputs (the head)
            depth = n_layers + 1
        lr = base_lr * decay ** (n_layers + 1 - depth)
        groups.append({"params": [param], "lr": lr, "weight_decay": 0.0})
    return groups

optimizer = torch.optim.AdamW(llrd_param_groups(model), eps=1e-6)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
trainer = Trainer(model=model, args=args, train_dataset=train_features,
                  optimizers=(optimizer, scheduler))
```

**Warning:** `1e-4` *without* LLRD is too high for full fine-tuning and may diverge. If LLRD is off, use around `3e-5` (the common BERT-style default).

### 5.4 Fitting in GPU memory

A batch of 32 windows × 384 tokens may not fit on a 16 GB GPU. Use **gradient accumulation**: e.g. `per_device_train_batch_size=16, gradient_accumulation_steps=2` gives an effective batch of 32 with half the memory. The result is mathematically the same.

### 5.5 Using the Hugging Face `Trainer`

The `Trainer` runs the training loop for us (batching, backward pass, optimizer step, fp16, logging, checkpoints). Key `TrainingArguments`:

```python
TrainingArguments(
    output_dir="outputs/checkpoints/<run_name>",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
    seed=42,
    report_to="none",          # or "tensorboard" / "wandb"
)
```

The `Trainer`'s built-in evaluation only reports loss, which doesn't tell us EM/F1. So after each epoch (or at the end), run our own `predict.py` on the **validation split** and score it with the official script.

### 5.6 Development workflow
1. **Smoke test on CPU:** train on 200 examples for a few steps. It should run end-to-end without errors and the loss should go down.
2. **Overfit test:** train on ~100 examples for many steps. Training-set EM should approach 100%. If it doesn't, there's a bug in preprocessing or labels.
3. **Full run on GPU.** Watch the loss curve: it should drop quickly in the first few hundred steps, then slowly. A flat or exploding loss means the LR is wrong or the labels are broken.

---

## 6. Phase 4: Post-processing (scores → answer text)

The model gives scores per token, per window. We need **one answer string per question**.

For each question:
1. Collect all windows that came from it (match on `example_id`).
2. In each window, take the **top `n_best = 20`** start positions and the top 20 end positions by score.
3. Consider every (start, end) pair from those lists and **discard invalid ones**:
   - start or end falls on a question/special token (offset is `None`)
   - `end < start`
   - span longer than `max_answer_length = 30` tokens
4. Score each remaining pair as `start_logit + end_logit`.
5. Across **all windows** of the question, pick the highest-scoring pair.
6. Use the offset mapping to convert it to characters: `answer = context[offsets[start][0] : offsets[end][1]]`.

Write the result as `{question_id: answer_text}` to a JSON file. This is exactly the format `evaluate-v2.0.py` expects.

Why add logits rather than multiply probabilities? Logits are log-scale scores, so adding them is like multiplying probabilities, and it avoids a softmax per window. This is the standard approach.

**Check:** the output JSON has exactly one entry per question id in the input file, and no empty strings.

---

## 7. Phase 5: Evaluation

### 7.1 On the validation split (use for all decisions)

```bash
python src/predict.py --model outputs/checkpoints/<run> --data data/splits/val_split.json --out outputs/val_predictions.json
python evaluate-v2.0.py data/splits/val_split.json outputs/val_predictions.json
```

Scores on our validation split will be **lower** than on the dev set for the same model. The validation questions come from the training file and have only **one** reference answer, while dev questions usually have **three or more** (only 139 of 10,570 have fewer), and the metric takes the best match across references. Compare validation runs with other validation runs only.

### 7.2 On the test set (dev-v1.1.json): only for final numbers

```bash
python src/predict.py --model outputs/checkpoints/<final_run> --data data/dev-v1.1.json --out outputs/predictions.json
python evaluate-v2.0.py data/dev-v1.1.json outputs/predictions.json
```

Run this **once per finalised model** (baseline and final system), not while tuning.

### 7.3 What to expect

| Reference | Dev EM | Dev F1 |
|---|---|---|
| Logistic regression (SQuAD paper) | 40.0 | 51.0 |
| Human (SQuAD paper) | 80.3 | 90.5 |
| ELECTRA-Base (reported in the ELECTRA paper) | 86.8 | 92.7 |

The ELECTRA paper used a slightly more elaborate answer head (borrowed from XLNet) than the single linear layer we use, so expect to land **slightly below** its numbers with a correct implementation. A result far below that (e.g. F1 under 85) almost certainly means a bug, most often in preprocessing or post-processing.

---

## 8. Phase 6: Experiments

Change **one thing at a time**, judge on the **validation split**, and record every run in a results table (run name, config, val EM, val F1, training time).

### 8.1 Baseline vs. final system

The README's results table needs a baseline and a final system. Suggested pairing:

| Run | Description |
|---|---|
| **Baseline** | ELECTRA-base, "standard" recipe: LR `3e-5`, no LLRD, 2 epochs, batch 32 |
| **Final** | ELECTRA-base with the paper's recipe (LLRD 0.8, LR `1e-4`) plus whatever ablations below help on validation |

Optionally, also reimplement the SQuAD paper's **sliding-window** baseline (no learning at all) as a "how far have we come" reference point. It takes very little code.

### 8.2 Ablations worth running (in priority order)

1. **LLRD on/off** and LR in `{5e-5, 1e-4, 2e-4}`: likely the biggest effect.
2. **Epochs** `{2, 3}`
3. **Post-processing settings** `max_answer_length ∈ {15, 30, 50}`, `n_best ∈ {10, 20}`: cheap because they need **no retraining**. Just rerun prediction.
4. **Window settings** `max_length ∈ {384, 512}`, `doc_stride ∈ {64, 128}`
5. **Seeds:** rerun the final config with 3 seeds and report mean ± std. Fine-tuning results vary between seeds, so a difference smaller than the seed variation isn't meaningful.

### 8.3 Final model
After choosing hyperparameters on validation, either:
- use the best validation checkpoint as-is, **or**
- retrain on the **full** training file (train + val) with the chosen settings. This gives slightly more data but no validation check, so decide this upfront as a team.

Then evaluate once on `dev-v1.1.json` and fill in the README table.

---

## 9. Phase 7: Error analysis

This follows section 9 of the README and the analysis style of the SQuAD paper. Use the **dev-set predictions of the final model**. Since the model is now fixed, looking at test errors here doesn't bias the result.

Start by building one table (pandas DataFrame) with one row per question: `id, question, context, gold answers, prediction, EM, F1`. Recompute EM/F1 per question with the functions in `evaluate-v2.0.py` (`compute_exact`, `compute_f1`).

| Analysis | How | Automatic? |
|---|---|---|
| **Boundary errors** | Among wrong answers (EM = 0): F1 > 0 means *partially right* (boundary issue), F1 = 0 means *completely wrong*. Also check whether the prediction contains the gold answer or the gold contains the prediction (too long vs. too short). | Yes |
| **Question type** | Group by question word: what / who / when / where / why / how / which / other. Report EM/F1 per group. | Yes |
| **Answer type** | Classify the gold answer with spaCy: date, number, person, location, other entity (via NER), else noun phrase / verb phrase / clause (via POS/dependency). Mirrors Table 2 of the SQuAD paper. | Yes (heuristic) |
| **Answer & passage length** | EM/F1 bucketed by gold answer length (in words) and by passage length / number of windows | Yes |
| **Syntactic divergence** | Hard to compute exactly as in the paper. A simpler proxy: word overlap between question and answer sentence (lower overlap means more paraphrasing). | Yes (proxy) |
| **Reasoning type** | Manually label a random sample of ~100 errors with the SQuAD paper's categories (synonymy, world knowledge, syntactic variation, multi-sentence reasoning, ambiguous). Split the sample among teammates and record labels in `src/analysis/annotate_sample.csv`. | Manual |

Include several concrete examples of each error type in the report. Concrete examples make the analysis much more convincing than numbers alone.

---

## 10. Suggested order of work and milestones

| Phase | Deliverable | "Done" means |
|---|---|---|
| 0. Setup | `requirements.txt`, `.gitignore`, GPU access confirmed | `import transformers` works on the GPU machine and `torch.cuda.is_available()` is `True` |
| 1. Data | `src/data.py`, `data/splits/*` | Checks in 3.6 pass |
| 2. Preprocessing | `src/preprocess.py` | Round-trip test in 4.7 passes (> 99%) |
| 3. Post-processing | `src/postprocess.py`, `src/predict.py` | Untrained model produces a valid `predictions.json` that the scorer accepts (scores will be near 0, which is fine) |
| 4. Training | `src/train.py` | CPU smoke test and overfit test pass (5.6) |
| 5. Baseline | First full GPU run | Baseline val + dev scores recorded |
| 6. Final model | Ablations + final run | Results table complete, final dev scores in README |
| 7. Analysis | `src/analysis/*`, figures | All rows of the table in section 9 done |
| 8. Write-up | README usage section, report | Someone else can reproduce the results from the README |

Phases 2 and 3 can be developed **in parallel** by different teammates. Build post-processing *before* training is finished so it's ready the moment the first model is.

---

## 11. Common pitfalls checklist

- [ ] Using `electra-base-generator` instead of `electra-base-discriminator`
- [ ] Using a *slow* tokenizer (no `offset_mapping`)
- [ ] Truncating the question instead of the passage (`truncation="only_second"`)
- [ ] Forgetting that one example becomes several windows: mixing up example indices and feature indices
- [ ] Allowing answers that start or end inside the question tokens
- [ ] Off-by-one errors: SQuAD character ends are exclusive (`start + len(text)`)
- [ ] Tuning anything on `dev-v1.1.json`
- [ ] Splitting validation by question instead of by article
- [ ] Comparing validation scores to dev scores directly (different number of reference answers)
- [ ] Not saving checkpoints to persistent storage on Colab/Kaggle
- [ ] Committing model checkpoints (~420 MB) to git
- [ ] High LR (`1e-4`) without layer-wise decay
- [ ] Not fixing random seeds / not logging configs, so a good run can't be reproduced

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Pre-training** | Training a model on a huge unlabelled text corpus with a self-supervised task, so it learns general language knowledge. Already done for us by Google. |
| **ELECTRA** | A Transformer encoder pre-trained with *replaced token detection*: a small generator swaps some words, and the main model (the **discriminator**) learns to spot which words were replaced. This is more sample-efficient than BERT's masked-word prediction. "Base" = 12 layers, 768 hidden size, ~110M parameters. |
| **Fine-tuning** | Continuing to train a pre-trained model (with a small new layer on top) on a specific labelled task, here SQuAD. All weights are updated. |
| **QA head** | A single linear layer mapping each token's 768-dim vector to 2 numbers: start score and end score. |
| **Tokenizer / sub-word tokens** | Splits text into pieces from a fixed vocabulary (~30k for ELECTRA). Rare words are broken into several pieces. |
| **[CLS], [SEP]** | Special tokens. `[CLS]` is at position 0; `[SEP]` separates question from passage and marks the end. We use position 0 as the "no answer in this window" label. |
| **Window / feature / doc_stride** | A chunk of a long passage that fits in `max_length` tokens. Consecutive windows overlap by `doc_stride` tokens. |
| **Offset mapping** | For each token, the `(start_char, end_char)` positions in the original text it came from. |
| **Logits** | Raw, unnormalised scores output by the model. Softmax turns them into probabilities. |
| **Cross-entropy loss** | The standard classification loss. Here, "which of the 384 positions is the start (or end)?" is treated as a 384-way classification. |
| **Learning-rate warmup / linear decay** | The LR starts at 0, rises linearly over the first 10% of steps, then decreases linearly to 0. Warmup avoids destroying pre-trained weights with big early updates. |
| **Layer-wise LR decay (LLRD)** | Giving lower layers smaller learning rates than upper layers (section 5.3). |
| **Gradient accumulation** | Summing gradients over several small batches before one optimizer step, to simulate a larger batch with less memory. |
| **Mixed precision (fp16/bf16)** | Doing most computation in 16-bit floats for speed and memory savings. |
| **EM / F1** | Exact Match: the normalised prediction equals a reference exactly. F1: word-overlap score between prediction and reference, taking the best reference. Both are averaged over questions. |
| **Ablation** | An experiment that changes or removes one component to measure its effect. |

---

## References

- Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555). In *ICLR 2020*. (Fine-tuning hyperparameters are in the appendix.)
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264.pdf). In *EMNLP 2016*.
- Hugging Face course, chapter on question answering: https://huggingface.co/learn/nlp-course/chapter7/7
- Hugging Face `run_qa.py` example script (reference implementation of the pre- and post-processing above): https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
