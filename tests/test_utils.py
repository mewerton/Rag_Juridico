import os
from utils import (
    count_tokens,
    prefix_documents_for_e5,
    hash_filename,
    format_response,
    ensure_directory,
    extract_metadata,
)
from langchain_core.documents import Document


def test_count_tokens():
    assert count_tokens("Isso é um teste.") > 0


def test_prefix_documents_for_e5():
    docs = [Document(page_content="Texto qualquer")]  # simula docling
    prefixed_docs = prefix_documents_for_e5(docs)
    assert prefixed_docs[0].page_content.startswith("passage:")


def test_hash_filename():
    hashed = hash_filename("documento.pdf")
    assert hashed.endswith(".pdf")
    assert len(hashed.split(".")[0]) == 32


def test_format_response():
    texto = "   Resultado da IA\n\ncom espaços.   "
    assert format_response(texto) == "Resultado da IA\ncom espaços."


def test_ensure_directory(tmp_path):
    temp_dir = tmp_path / "nova_pasta"
    ensure_directory(temp_dir)
    assert os.path.exists(temp_dir)


def test_extract_metadata():
    doc = Document(page_content="teste", metadata={"source": "teste.pdf"})
    result = extract_metadata(doc)
    assert "teste.pdf" in result