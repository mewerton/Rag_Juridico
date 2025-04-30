import os
import shutil
import pytest
from rag_pipeline import (
    load_documents_with_docling,
    create_or_load_vectorstore,
    process_document,
)
from langchain_huggingface import HuggingFaceEmbeddings

TEST_FILE = "data/documentos/teste_documento.pdf"
INDEX_PATH = "data/indexes/teste_documento_faiss_index"

@pytest.fixture(scope="module", autouse=True)
def cleanup_generated_files():
    # Executa antes do primeiro teste
    yield
    # Executa depois de todos os testes
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)
    print("Índices temporários removidos após os testes.")

def test_load_documents():
    docs = load_documents_with_docling(TEST_FILE)
    assert docs and any(doc.page_content.strip() for doc in docs)

def test_vectorstore_creation():
    docs = load_documents_with_docling(TEST_FILE)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = create_or_load_vectorstore(TEST_FILE, docs, embeddings)
    assert vectorstore

def test_process_document_pipeline():
    chain = process_document(TEST_FILE)
    assert chain
