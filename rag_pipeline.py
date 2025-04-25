import os
from typing import List

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# Configurações
DATA_FOLDER = "data"
DOCUMENTS_FOLDER = os.path.join(DATA_FOLDER, "documentos")
INDEX_FOLDER = os.path.join(DATA_FOLDER, "indexes")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3-8b-8192"

# Função: carregar documento com DoclingLoader (oficial)
def load_documents_with_docling(file_path: str, export_type=ExportType.DOC_CHUNKS) -> List[LCDocument]:
    """Carrega documentos estruturados de vários formatos usando DoclingLoader."""
    loader = DoclingLoader(
        file_path=file_path,
        export_type=export_type
    )
    documents = loader.load()
    return documents

# Função: opcional — re-chunk se necessário
def split_text_into_chunks(documents: List[LCDocument]) -> List[LCDocument]:
    """Divide documentos em chunks menores para melhor performance do embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
    return text_splitter.split_documents(documents)

# Função: criar ou carregar vetorstore FAISS
def create_or_load_vectorstore(file_path: str, documents: List[LCDocument], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Cria ou carrega um índice vetorial FAISS com os documentos processados."""
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    index_path = os.path.join(INDEX_FOLDER, f"{base_filename}_faiss_index")

    # Garante que o diretório indexes existe
    os.makedirs(INDEX_FOLDER, exist_ok=True)

    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path, ignore_errors=True)
        vectorstore.save_local(index_path)

    return vectorstore

# Função: criar RAG Chain com LangChain
def create_rag_chain(vectorstore: FAISS) -> object:
    """Cria a RAG chain conectando vetorstore e LLM (Groq + Llama3)."""
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    llm = ChatGroq(
        temperature=0.1,
        model_name=LLM_MODEL_NAME,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    template = """
    Você é um assistente jurídico especializado. Analise cuidadosamente o seguinte contexto extraído de documentos jurídicos e responda de forma objetiva, sem adicionar informações externas.

    Se não souber a resposta, diga "Não encontrei informações suficientes no documento."

    <context>
    {context}
    </context>
    """
    prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# Função principal: processar o documento
def process_document(file_path: str) -> object:
    """Pipeline completo: lê o documento, cria vetorstore e prepara a RAG chain."""
    documents = load_documents_with_docling(file_path)
    # Se quiser forçar novo chunk, use a linha abaixo:
    # documents = split_text_into_chunks(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = create_or_load_vectorstore(file_path, documents, embeddings)
    rag_chain = create_rag_chain(vectorstore)
    return rag_chain
