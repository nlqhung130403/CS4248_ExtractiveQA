#!/usr/bin/env python3
"""Train one encoder QA model and evaluate its saved checkpoint on SQuAD 1.1."""
import json
import os
from pathlib import Path

# EDIT HERE: Slurm array selects the model; all runs share these settings.
MODELS = ['bert-base-cased', 'bert-base-uncased', 'microsoft/deberta-v3-base']
ROOT = Path('/path/to/project')  # Set to the directory containing SQUAD/ and runs/
EPOCHS = 2
LEARNING_RATE = 1e-5
MICROBATCH = 4
ACCUMULATION = 4
SEED = 42
DRY_RUN = os.environ.get('ENCODER_DRY_RUN') == '1'


def main():
    model_name = MODELS[int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))]
    output = ROOT / 'runs' / ('encoder_' + model_name.replace('/', '_'))
    for name in ['train-v1.1.json', 'dev-v1.1.json', 'evaluate-v2.0.py']:
        assert (ROOT / 'SQUAD' / name).is_file(), name
    metadata = dict(model=model_name, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                    microbatch=MICROBATCH, accumulation=ACCUMULATION, seed=SEED,
                    output=str(output), max_length=384, stride=128)
    print(json.dumps(metadata, indent=2), flush=True)
    if DRY_RUN:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, set_seed
    from utils import load_squad_data, preprocess_squad_examples, UniversalQADataset
    import evaluate_cluster as evaluation
    assert torch.cuda.is_available(), 'GPU required'
    set_seed(SEED)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_config.json').write_text(json.dumps(metadata, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    assert tokenizer.is_fast
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).float()
    model.config.use_cache = False
    contexts, questions, answers = load_squad_data(ROOT / 'SQUAD/train-v1.1.json')
    features = preprocess_squad_examples(contexts, questions, answers, tokenizer)
    dataset = UniversalQADataset(features)
    args = TrainingArguments(output_dir=str(output), num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE, per_device_train_batch_size=MICROBATCH,
        gradient_accumulation_steps=ACCUMULATION, fp16=True,
        gradient_checkpointing=True, logging_steps=100, save_strategy='epoch',
        save_total_limit=2, report_to='none', seed=SEED)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    trainer.save_model(str(output / 'final'))
    tokenizer.save_pretrained(output / 'final')
    evaluation.squad_eval = evaluation.load_official_eval_module(str(ROOT / 'SQUAD/evaluate-v2.0.py'))
    metrics, predictions, _ = evaluation.run_evaluation(model, tokenizer,
        str(ROOT / 'SQUAD/dev-v1.1.json'), device='cuda', batch_size=8)
    (output / 'evaluation.json').write_text(json.dumps(dict(metrics=metrics, predictions=predictions), indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
