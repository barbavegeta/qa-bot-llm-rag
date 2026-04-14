# QA Bot using Retrieval-Augmented Generation (RAG)

Deployable question-answering system that allows users to upload a PDF and ask natural language questions using a Retrieval-Augmented Generation pipeline.

## Project Structure

```text
qa-bot-rag/
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   └── config.py
├── ui/
│   └── gradio_app.py
├── data/
├── vectorstore/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## What this does

- loads a PDF
- splits it into chunks
- embeds the chunks
- stores them in Chroma
- retrieves relevant chunks
- generates an answer with an LLM

## Run locally

```bash
pip install -r requirements.txt
python ui/gradio_app.py
```

## Run API

```bash
uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t rag-bot .
docker run -p 8000:8000 rag-bot
```
