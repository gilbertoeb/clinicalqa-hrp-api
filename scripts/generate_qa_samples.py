# scripts/generate_qa_samples.py

# === Imports ===
import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# === Configuration ===
INPUT_PATH = Path("data/raw/mimic/discharge_notes_200.jsonl")
OUTPUT_PATH = Path("data/generated_qa_examples_200_0152bwixwixquixqubixuoqwibxuqwwdqhiwudh.jsonl")
NUM_SAMPLES = 100
MODEL_NAME = "claude-3-sonnet-20240229"

SYSTEM_PROMPT = """You are a medical QA dataset generator. Your job is to create training examples for extractive question answering from a clinical discharge summary.

For each note, extract 3–5 high-quality question-answer pairs. Each question should be clearly answerable from the note. Each answer must be a **verbatim span** from the note. For each answer, also report the `answer_start` index (the character position where the answer starts in the note).

Output as a JSON list like this:

[
  {
    "question": "What condition led to the patient's admission?",
    "answer_text": "shortness of breath",
    "answer_start": 289
  },
  ...
]

Rules:
- The answer must be a continuous span of text from the context.
- Compute 'answer_start' by finding the index of the answer string in the context.
- Choose clinically meaningful questions (e.g., diagnoses, meds, test results).
- Be diverse in question type and avoid repetition.

Only output the JSON list. Do not include explanations or formatting outside the JSON."""

# === Load API Key and Initialize Client ===
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# === Load Notes ===
with open(INPUT_PATH, "r") as f:
    notes = [json.loads(line) for line in f]

# === Sample Notes ===
samples = random.sample(notes, NUM_SAMPLES)

# === Generate QA Pairs and Write to Output ===
with open(OUTPUT_PATH, "a") as f:
    for i, sample in enumerate(samples):
        text = sample["text"].strip()
        print(f"\nNote {i+1}/{NUM_SAMPLES} — HADM_ID: {sample['hadm_id']}")

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        )

        try:
            qa_pairs = json.loads(response.choices[0].message.content)
            for qa in qa_pairs:
                item = {
                    "subject_id": sample["subject_id"],
                    "hadm_id": sample["hadm_id"],
                    "context": text,
                    **qa
                }
                f.write(json.dumps(item) + "\n")
        except Exception as e:
            print("JSON decode failed:", e)
            continue

print(f"\nSaved QA examples to {OUTPUT_PATH}")