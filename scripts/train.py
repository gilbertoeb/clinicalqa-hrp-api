import os
import json
import yaml

import torch
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from src.data_utils import prepare_train_features

# === Environment setup ===
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Debug info ===
print(transformers.__version__)
print(transformers.TrainingArguments.__module__)

# === Load training config ===
with open("configs/train_config_radiology.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Load tokenizer & model ===
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = AutoModelForQuestionAnswering.from_pretrained(config["model_name"])
model.to(device)

# === Load dataset ===
cached_path = config["data_path"]
print(f"Loading tokenized dataset from cache: {cached_path}")
tokenized_dataset = torch.load(cached_path, weights_only=False)

sample = tokenized_dataset[0]
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"{k}: type={type(v)} — SKIPPED")

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
)

dl = DataLoader(tokenized_dataset, batch_size=16)
first = next(iter(dl))
print({k: (v.shape, v.dtype) for k, v in first.items()})

first = tokenized_dataset[0]
for k, v in first.items():
    print(f"{k}: {type(v)}, shape={v.shape if isinstance(v, torch.Tensor) else 'n/a'}")

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="no",
    learning_rate=float(config["learning_rate"]),
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_eval_batch_size"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=config["weight_decay"],
    logging_dir=config["logging_dir"],
    logging_steps=config["logging_steps"],
    save_total_limit=config["save_total_limit"],
    report_to=config["report_to"],
    seed=config.get("seed", 42)
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# === Train ===
trainer.train()

# === Save final model ===
trainer.save_model(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])