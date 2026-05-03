"""Configuration helpers for the RAG app.

Set these environment variables before running locally:

export WATSONX_API_KEY="your_ibm_api_key"
export WATSONX_PROJECT_ID="your_watsonx_project_id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"

Optional overrides:

export WATSONX_LLM_MODEL_ID="ibm/granite-3-2-8b-instruct"
export WATSONX_EMBEDDING_MODEL_ID="ibm/slate-30m-english-rtrvr"
export WATSONX_MAX_NEW_TOKENS="256"
export WATSONX_TEMPERATURE="0.5"
"""

import os

WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID") or os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_TOKEN = os.getenv("WATSONX_TOKEN")
