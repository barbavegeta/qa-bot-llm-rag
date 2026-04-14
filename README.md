# QA Bot using Retrieval-Augmented Generation (RAG)

Interactive question-answering system that allows users to upload a PDF and ask natural language questions about its content using a Retrieval-Augmented Generation (RAG) pipeline.

## Overview

This project implements a full RAG workflow combining document processing, embeddings, vector search, and large language models to generate context-aware answers.

Unlike standard LLM prompting, this system retrieves relevant document chunks before generating a response, improving accuracy and grounding answers in source material.

---

## Features

- Upload any PDF document
- Automatically process and split text into chunks
- Generate embeddings using IBM watsonx embedding model
- Store vectors in a Chroma vector database
- Retrieve relevant document sections for each query
- Generate answers using an LLM (IBM Granite model)
- Simple user interface using Gradio

---

## Architecture

The pipeline follows a standard RAG workflow:

1. **Document Loading**
   - PDF is loaded and parsed into text

2. **Text Splitting**
   - Documents split into overlapping chunks for better retrieval

3. **Embedding**
   - Each chunk converted into vector representations

4. **Vector Database**
   - Stored in Chroma for similarity search

5. **Retrieval**
   - Relevant chunks retrieved based on user query

6. **LLM Generation**
   - LLM generates answer using retrieved context

---

## Tech Stack

- Python
- LangChain
- IBM watsonx (LLM + embeddings)
- Chroma (vector database)
- Gradio (UI)

---

## Project Structure

qa-bot-llm-rag/
├── qabot.py
├── requirements.txt
├── README.md

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

python qabot.py

3. Open the Gradio interface in your browser
