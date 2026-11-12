# Encoder QA training code

`train_encoder.py`, `evaluate_cluster.py`, and `utils.py` are the SoC cluster scripts used for the reported BERT-base-cased, BERT-base-uncased, and DeBERTa-v3-base SQuAD 1.1 runs. `train_encoder.py` includes the float32 model-weight fix needed for DeBERTa mixed-precision training. `encoder.sbatch` is the Slurm array launcher.

`train.py` and `evaluate.py` are the original project scripts. `evaluate-v2.0.py` is the official SQuAD scoring code.

Set the `/path/to/...` placeholders before running. `train_encoder.py` expects a project root containing `SQUAD/train-v1.1.json`, `SQUAD/dev-v1.1.json`, and `SQUAD/evaluate-v2.0.py`; it writes to `runs/`. `encoder.sbatch` also needs paths for logs, scratch, dependencies, and the Python interpreter. Dataset JSON files, dependencies, checkpoints, and prediction files are separate from this code folder.
