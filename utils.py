import os
from typing import List
import time
import hashlib
from langchain_core.documents import Document as LCDocument
from transformers import AutoTokenizer


def count_tokens(text: str, model_name: str = "intfloat/multilingual-e5-large") -> int:
    """Conta quantos tokens o texto possui com base no modelo informado."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))

# adiciona prefixo "passage:" ao conteúdo dos documentos
def prefix_documents_for_e5(documents: List[LCDocument]) -> List[LCDocument]:
    """Adiciona prefixo 'passage:' no conteúdo dos documentos (necessário para E5 embeddings)."""
    for doc in documents:
        doc.page_content = f"passage: {doc.page_content.strip()}"
    return documents

def hash_filename(filename: str) -> str:
    """Gera um nome de arquivo hash único baseado no nome original."""
    base, ext = os.path.splitext(filename)
    hashed = hashlib.md5(base.encode()).hexdigest()
    return f"{hashed}{ext}"


def log_time(func):
    """Decorator para medir o tempo de execução de uma função."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"⏱️ {func.__name__} levou {duration:.2f}s para executar.")
        return result
    return wrapper


def extract_metadata(document: LCDocument) -> str:
    """Retorna metadados úteis de um Document para exibição ou debug."""
    meta = document.metadata or {}
    return f"[Origem: {meta.get('source', 'Desconhecida')}]"


def ensure_directory(path: str) -> None:
    """Cria o diretório se ele não existir."""
    os.makedirs(path, exist_ok=True)


def format_response(text: str) -> str:
    """Formata a resposta da IA removendo espaços extras e limpando markdown."""
    return text.strip().replace("\n\n", "\n")
