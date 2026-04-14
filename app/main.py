from pathlib import Path
import shutil

from fastapi import FastAPI, File, Form, UploadFile
from app.rag_pipeline import ask_question

app = FastAPI(title="QA Bot RAG API")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "QA Bot RAG API is running"}


@app.post("/ask/")
async def ask(file: UploadFile = File(...), query: str = Form(...)):
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    answer = ask_question(str(file_path), query)
    return {"answer": answer}
