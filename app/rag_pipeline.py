"""RAG pipeline for PDF question answering using IBM watsonx + LangChain."""

import os
import warnings
from pathlib import Path

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")


def _watsonx_credentials() -> dict:
    """Return watsonx credentials from environment variables."""
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID") or os.getenv("PROJECT_ID")
    api_key = os.getenv("WATSONX_API_KEY")
    token = os.getenv("WATSONX_TOKEN")

    if not project_id:
        raise ValueError(
            "Missing Watsonx project ID. Set WATSONX_PROJECT_ID before running the app."
        )

    if not api_key and not token:
        raise ValueError(
            "Missing Watsonx credentials. Set WATSONX_API_KEY or WATSONX_TOKEN before running the app."
        )

    credentials = {
        "url": url,
        "project_id": project_id,
    }

    if api_key:
        credentials["api_key"] = api_key
    else:
        credentials["token"] = token

    return credentials


def get_llm():
    model_id = os.getenv("WATSONX_LLM_MODEL_ID", "ibm/granite-3-2-8b-instruct")
    parameters = {
        GenParams.MAX_NEW_TOKENS: int(os.getenv("WATSONX_MAX_NEW_TOKENS", "256")),
        GenParams.TEMPERATURE: float(os.getenv("WATSONX_TEMPERATURE", "0.5")),
    }

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        params=parameters,
        **_watsonx_credentials(),
    )
    return watsonx_llm


def document_loader(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    loader = PyPDFLoader(str(path))
    return loader.load()


def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(data)


def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": False},
    }

    embedding_model = WatsonxEmbeddings(
        model_id=os.getenv("WATSONX_EMBEDDING_MODEL_ID", "ibm/slate-30m-english-rtrvr"),
        params=embed_params,
        **_watsonx_credentials(),
    )
    return embedding_model


def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory="vectorstore",
    )
    return vectordb


def retriever(file_path: str):
    documents = document_loader(file_path)
    chunks = text_splitter(documents)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


def ask_question(file_path: str, query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    retriever_obj = retriever(file_path)
    llm = get_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )

    response = qa.invoke({"query": query})
    return response["result"]
