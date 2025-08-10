# scripts/generate_radiology_qa_samples.py

# === Imports ===
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# === Configuration ===
INPUT_PATH = Path("data/raw/radiology/radiology_sampled_400.jsonl")
OUTPUT_PATH = Path("data/raw/radiology/generated_radiology_qa_400.jsonl")
MODEL_NAME = "claude-3-sonnet-20240229"

SYSTEM_PROMPT = """You are a dataset generator for radiology question answering.

Given a chest X-ray radiology report, extract 3–5 high-quality extractive question-answer pairs. Each question should be clearly answerable using a **verbatim span** from the report. Also include the `answer_start` character index of the answer string.

Only output a JSON list like this:

[
  {
    "question": "What condition is ruled out in the impression?",
    "answer_text": "No acute cardiopulmonary process identified.",
    "answer_start": 781
  },
  ...
]

Rules:
- The answer must be a continuous span of text from the report.
- Compute 'answer_start' by finding the index of the answer string in the report.
- Focus on medically relevant questions about findings, impressions, tubes/lines, diagnoses, etc.
- Be diverse in question types and avoid repetition.
- Only output the raw JSON list — no prose, headers, or explanations.
"""

# === Load API Key and Initialize Client ===
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# === Load Radiology Reports ===
with open(INPUT_PATH, "r") as f:
    notes = [json.loads(line) for line in f]

# === Generate QA Pairs and Write to Output ===
with open(OUTPUT_PATH, "a") as f_out:
    for i, sample in enumerate(notes):
        text = sample["report"].strip()
        print(f"\nRadiology Report {i+1}/{len(notes)}")

        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=512,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
            )

            qa_pairs = json.loads(response.choices[0].message.content)
            for qa in qa_pairs:
                item = {
                    "subject_id": sample.get("subject_id", ""),
                    "hadm_id": sample.get("study_id", ""),
                    "context": text,
                    **qa
                }
                f_out.write(json.dumps(item) + "\n")

        except Exception as e:
            print(f"Error for study_id={sample.get('study_id')}: {e}")
            continue

print(f"\nSaved radiology QA examples to {OUTPUT_PATH}")