from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import warnings

warnings.filterwarnings("ignore")


def get_llm():
    model_id = "ibm/granite-3-2-8b-instruct"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm


def document_loader(file_path: str):
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks


def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": False},
    }

    embedding_model = WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
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
    splits = document_loader(file_path)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever()
    return retriever_obj


def ask_question(file_path: str, query: str) -> str:
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
