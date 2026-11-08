# evaluate.py
import argparse
import json
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib

DATASET_EVAL_PATH = "/home/p/patrick4/CS4248/SQUAD/dev-v1.1.json"
EVAL_CODE_PATH = "/home/p/patrick4/CS4248/SQUAD/evaluate-v2.0.py"
LORA_PATH = [
    "/home/p/patrick4/CS4248/runs/qwen-squad-lora", 
    None 
][0]
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
EVAL_BATCH_SIZE = 4
MAX_NEW_TOKENS = 32
MAX_LENGTH = 1024
RESULTS_SAVE_PATH = [
    "/home/p/patrick4/CS4248/runs/qwen-squad-lora/eval_lora.json", 
    "/home/p/patrick4/CS4248/runs/qwen-squad-lora/eval_base.json"
][0]

def load_squad_eval_data(filepath) :
    with open(filepath, "r", encoding="utf-8") as f :
        raw_data = json.load(f)["data"]

    examples = []
    for article in raw_data:
        for paragraph in article["paragraphs"] :
            context = paragraph["context"]
            for qa in paragraph["qas"] :
                examples.append({
                    "id" : qa["id"],
                    "question" : qa["question"],
                    "context" : context,
                })
    return raw_data, examples


def clean_prediction(current_generated):
    if "</think>" in current_generated:
        current_generated = current_generated.split("</think>")[-1]

    current_generated = current_generated.strip()  
    lowered_current_generated = current_generated.lower()

    for prefix in ("the answer is:", "the answer is", "answer:") :
        if lowered_current_generated[:len(prefix)] == prefix :
            current_generated = current_generated[len(prefix):].lstrip()
            break

    return current_generated.strip().split("\n")[0].strip()


def load_official_eval_module(filepath) :
    spec = importlib.util.spec_from_file_location("squad_eval", filepath)
    squad_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(squad_eval)
    return squad_eval



if __name__ == "__main__":
    squad_eval = load_official_eval_module(EVAL_CODE_PATH)
    raw_data, examples = load_squad_eval_data(DATASET_EVAL_PATH)

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    if LORA_PATH == None :
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, LORA_PATH)

    model.eval()

    predictions = {}
    batch_size = EVAL_BATCH_SIZE

    for i in range(0, len(examples), batch_size):
        if i % 100 == 0 :
            print(i, "/", len(examples))
        batch_examples = examples[i : i + batch_size]

        prompts = []
        for ex in batch_examples : 
            prompt_msgs = [
                {"role": "system", "content": "You are an extractive QA system. Answer the question using ONLY the exact words from the context. Give ONLY the direct answer text, no explanations."},
                {"role": "user", "content": f"Context:\n{ex['context']}\n\nQuestion:\n{ex['question']}"}
            ]
            text = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        input_len = inputs.input_ids.shape[1]
        gen_texts = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for ex, gen_text in zip(batch_examples, gen_texts):
            predictions[ex["id"]] = clean_prediction(gen_text)

    exact_raw, f1_raw = squad_eval.get_raw_scores(raw_data, predictions)
    results = squad_eval.make_eval_dict(exact_raw, f1_raw)

    with open(RESULTS_SAVE_PATH, "w") as f:
        json.dump({"metrics": results, "predictions": predictions}, f)

