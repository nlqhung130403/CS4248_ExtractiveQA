import torch
import json
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

OUTPUT_DIR = "/home/p/patrick4/CS4248/runs/qwen-squad-lora"
DATASET_TRAIN_PATH = "/home/p/patrick4/CS4248/SQUAD/train-v1.1.json"
MAX_LENGTH = 1024
MODEL_NAME = "Qwen/Qwen3.5-0.8B"


def load_squad_train_data() :
    with open(DATASET_TRAIN_PATH) as f :
        squad_data = json.load(f)["data"]

    examples = []
    for article in squad_data :
        for paragraph in article["paragraphs"] :
            context = paragraph["context"]
            for qa in paragraph["qas"] :
                if qa["answers"] :
                    examples.append({
                        "context" : context,
                        "question" : qa["question"],
                        "answer" : qa["answers"][0]["text"],
                    })
    return examples



def my_create_preprocess_function(tokenizer) :
    def preprocess_function(example) :
        prompt_msgs = [
            {"role": "system", "content": "You are an extractive QA system. Answer the question using ONLY the exact words from the context. Give ONLY the direct answer text, no explanations."},
            {"role": "user", "content": f"Context:\n{example['context']}\n\nQuestion:\n{example['question']}"}
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": example["answer"]}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, enable_thinking=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)["input_ids"]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids" : full_ids,
            "attention_mask" : [1] * len(full_ids),
            "labels" : labels
        }
    return preprocess_function



if __name__ == "__main__" :
    print("Loading training data...")
    raw_examples = load_squad_train_data()
    dataset = Dataset.from_list(raw_examples)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None :
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    
    preproc_funct = my_create_preprocess_function(tokenizer)
    tokenized_dataset = dataset.map(
        preproc_funct,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    tokenized_dataset = tokenized_dataset.filter(lambda ex: any(x != -100 for x in ex["labels"]))
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA weights to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    print("Training finished successfully.")


