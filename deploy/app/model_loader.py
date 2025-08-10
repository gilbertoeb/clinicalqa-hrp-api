# deploy/app/model_loader.py

# === Imports ===
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch

# === Constants ===
MODEL_PATH = "models/clinicalbert-qa-mixed-v3"

# === Model Loading Utilities ===

def load_model_and_tokenizer():
    """
    Load the tokenizer and QA model from the specified path, and initialize the QA pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    model.to(torch.device("cpu"))  # Cloud Run doesn't support GPU
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return tokenizer, model, qa_pipeline

# === QA Inference Utility ===

def answer_question(context, question, qa_pipeline):
    """
    Use the QA pipeline to answer a question given a context.
    """
    response = qa_pipeline({"context": context, "question": question})
    return response["answer"]