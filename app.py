from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

# Path ke model IndoBERT
model_path = "Emolog-ML\checkpoint-1315"

# Load tokenizer dan model IndoBERT
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# FastAPI app instance
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JournalEntry(BaseModel):
    text: str

def get_mood_from_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class_idx

# API endpoint untuk mendapatkan suggestion berdasarkan mood
@app.post("/get_mood/")
async def get_mood(entry: JournalEntry):
    mood = get_mood_from_text(entry.text)
    return {"mood": mood}
