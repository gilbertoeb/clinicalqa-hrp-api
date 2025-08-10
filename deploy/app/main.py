# deploy/app/main.py

# === Imports ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model_and_tokenizer, answer_question

# === App Initialization ===
app = FastAPI()

# === Model Loading (Startup) ===
# Load model and tokenizer once at startup for efficiency
print("--- Loading model...")
tokenizer, model, qa_pipeline = load_model_and_tokenizer()
print("--- Model loaded successfully!")

# === Data Models ===
class QARequest(BaseModel):
    context: str
    question: str

# === API Endpoints ===
@app.post("/qa")
def get_answer(request: QARequest):
    """
    Endpoint to get an answer for a given context and question.
    """
    try:
        answer = answer_question(request.context, request.question, qa_pipeline)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))