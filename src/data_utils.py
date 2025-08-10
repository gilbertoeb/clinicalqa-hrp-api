# src/data_utils.py

# === Imports ===
import json
from datasets import load_dataset, Dataset

# === Functions ===

def load_qa_dataset(json_path: str) -> Dataset:
    """
    Load QA data from a JSONL file into a Hugging Face Dataset.
    """
    with open(json_path, "r") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def prepare_train_features(examples, tokenizer, config):
    """
    Tokenize and align answer spans for QA training.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=config["max_length"],
        stride=config["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_idx = sample_mapping[i]
        answer = examples["answer_text"][sample_idx]
        start_char = examples["answer_start"][sample_idx]
        end_char = start_char + len(answer)

        sequence_ids = tokenized_examples.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If answer is not fully inside the context, label as CLS
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            token_start = context_start
            token_end = context_end
            while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                token_start += 1
            while token_end >= 0 and offsets[token_end][1] >= end_char:
                token_end -= 1
            start_positions.append(token_start - 1)
            end_positions.append(token_end + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples