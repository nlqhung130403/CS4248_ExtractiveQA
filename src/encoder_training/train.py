from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from utils import load_squad_data, preprocess_squad_examples, UniversalQADataset
import torch


DATASET_TRAIN_PATH = "/path/to/SQUAD/train-v1.1.json"
DATASET_EVAL_PATH = "/path/to/SQUAD/dev-v1.1.json"
EVAL_CODE_PATH = "/path/to/SQUAD/evaluate-v2.0.py"
MODEL_NAME = "bert-base-cased"

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available() :
        device = "cuda"

    # 1. Load standard model without custom classes or inheritance
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # 2. Tokenize dataset
    contexts, questions, answers = load_squad_data(DATASET_TRAIN_PATH)

    train_encodings = preprocess_squad_examples(
        # contexts[:5000], questions[:5000], answers[:5000], tokenizer
        contexts, questions, answers, tokenizer
    )
    train_dataset = UniversalQADataset(train_encodings)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # # 3. Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)



    # 4. Standard PyTorch training loop
    model.train()
    epochs = 2

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            optimizer.zero_grad()

            # Send all items (input_ids, attention_mask, start/end_positions, etc.) to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loop.set_postfix(loss=loss.item())

