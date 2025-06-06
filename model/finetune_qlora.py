from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import torch
import json
from pathlib import Path

# === Config ===
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
TOKENIZER_PATH = "./model/tokenizer"
DATA_PATH = "./data/lora_formatted/train.jsonl"
OUTPUT_DIR = "./model/qlora_adapters"

# === Load Dataset ===
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

data = load_jsonl(DATA_PATH)
dataset = Dataset.from_list(data)

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# === Tokenize Dataset ===
def tokenize(example):
    prompt = example["prompt"]
    response = json.dumps(example["response"])
    full_text = prompt + "\n\n" + response
    encoded = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

tokenized_data = dataset.map(tokenize)

# === Load Base Model (Quantized) ===
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# === LoRA Config ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === Training Args ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    save_total_limit=1,
    logging_steps=10,
    save_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data
)

# === Train ===
trainer.train()

# === Save Adapter Weights ===
model.save_pretrained(OUTPUT_DIR)
print(f"[✓] LoRA adapter weights saved to {OUTPUT_DIR}")
