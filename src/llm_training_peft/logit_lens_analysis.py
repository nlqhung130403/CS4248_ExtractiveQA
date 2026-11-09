import torch
import json
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import jlens

# OUTPUT_DIR = "/home/p/patrick4/CS4248/runs/qwen-squad-lora"
DATASET_EVAL_PATH = "/home/p/patrick4/CS4248/SQUAD/dev-v1.1.json"
MAX_LENGTH = 1024
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LORA_PATH = [
    "/home/p/patrick4/CS4248/runs/qwen-squad-lora", 
    None 
][0]
OUTPUT_PATH = [
    '/home/p/patrick4/CS4248/runs/logit_lens/lora.jsonl', 
    '/home/p/patrick4/CS4248/runs/logit_lens/base.jsonl',  
][0]

# neuronpedia/jacobian-lens/blob/main/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = {
    "Qwen/Qwen3.5-0.8B" : "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    "Qwen/Qwen3.6-27B": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
}[MODEL_NAME]




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
                    "answers" :  [a["text"] for a in qa["answers"]],
                })
    return raw_data, examples

def top5(logits):
    return [tokenizer.decode([t]) for t in logits.topk(5).indices]



if __name__ == "__main__" : 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None :
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    if LORA_PATH :
        model = PeftModel.from_pretrained(model, str(LORA_PATH)).merge_and_unload()


    model_jlens = jlens.from_hf(model, tokenizer)

    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )

    raw_examples = load_squad_eval_data(DATASET_EVAL_PATH)  
    
    # prompt = "Fact: The currency used in the country shaped like a boot is"

    all_results_to_save = []
    for idx, current_ex in enumerate(raw_examples[1]) : 
        if idx % 100 == 0 : 
            print(idx, "/", len(raw_examples[1]))

        prompt_msgs = [
            {"role": "system", "content": "You are an extractive QA system. Answer the question using ONLY the exact words from the context. Give ONLY the direct answer text, no explanations."},
            {"role": "user", "content": f"Context:\n{current_ex['context']}\n\nQuestion:\n{current_ex['question']}"}
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        layers = [
            temp_idx for temp_idx in range(model_jlens.n_layers)
        ]

        if len(tokenizer(prompt)['input_ids']) <= MAX_LENGTH:

            # jlens_logits, model_logits, _ = lens.apply(model_jlens, prompt, layers=layers, positions=[-2])
            with torch.no_grad() : 
                logit_lens, model_logits, _ = lens.apply(
                    model_jlens, prompt, layers=layers, positions=[-1], use_jacobian=False
                )

            targets = []
            for ans in current_ex["answers"] :
                tokens_now = tokenizer.encode(ans, add_special_tokens=False)
                first_token_now =  tokenizer.decode([tokens_now[0]])
                if not (first_token_now in targets) : 
                    targets.append(first_token_now)

            result = {
                "id" : current_ex['id'],  
                "answers" : current_ex['answers'], 
                "target_tokens" : targets, 
                "layers" : {str(layer) : top5(logit_lens[layer][0]) for layer in layers}, 
                "model" : top5(model_logits[0]), 
                "position" : -1, 
                "lora" : True if LORA_PATH else False

            }

            all_results_to_save.append(json.dumps(result, ensure_ascii=False))

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f :
        for temp_my_res in all_results_to_save : 
            f.write(temp_my_res + "\n")
