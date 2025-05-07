from typing import List, Dict, Any

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

from pathlib import Path

from utils import (
    log_time,
    prefix_documents_for_e5,
    hash_filename,
    count_tokens,
    extract_metadata,
    format_response,
)

# Configurações
DATA_FOLDER = Path("data")
DOCUMENTS_FOLDER = DATA_FOLDER / "documentos"
INDEX_FOLDER = DATA_FOLDER / "indexes"
INDEX_FOLDER.mkdir(parents=True, exist_ok=True)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "llama3-8b-8192"
TOKEN_LIMIT = 7000  # limite aproximado de tokens para prompt


@log_time
def load_documents_with_docling(
    file_path: str,
    export_type: ExportType = ExportType.DOC_CHUNKS
) -> List[LCDocument]:
    """
    1° - O DoclingLoader lê o PDF, extrai o texto estruturado em “chunks”
         e retorna uma lista de Documentos.
    """
    loader = DoclingLoader(file_path=file_path, export_type=export_type)
    return loader.load()


def split_text_into_chunks(documents: List[LCDocument]) -> List[LCDocument]:
    """
    2° - O RecursiveCharacterTextSplitter divide os documentos em pedaços
         menores (300 tokens com 100 de sobreposição) para otimizar embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    return splitter.split_documents(documents)


@log_time
def create_or_load_vectorstore(
    file_path: str,
    documents: List[LCDocument],
    embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """
    3° - Cria ou carrega um índice FAISS:
         - Gera um hash a partir do nome do arquivo para nomear a pasta de índice.
         - Se já existir, carrega; caso contrário, cria e salva o novo índice.
    """
    try:
        # Garante que a pasta existe
        INDEX_FOLDER.mkdir(parents=True, exist_ok=True)

        # Nome base e hash do arquivo para isolar índices
        base = Path(file_path).stem
        hashed = hash_filename(base)
        index_path = INDEX_FOLDER / f"{hashed}_faiss_index"

        if index_path.exists():
            # Carrega índice existente
            return FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        # Cria novo índice a partir dos documentos
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(str(index_path))
        return vectorstore

    except FileNotFoundError as e:
        st.error(f"Arquivo não encontrado: {e}")
        return None
    except ValueError as e:
        st.error(f"Formato inválido: {e}")
        return None


def create_rag_chain(vectorstore: FAISS) -> Any:
    """
    4° - Cria o retriever com MMR e configura o LLM Groq + Llama3,
         monta a cadeia de documentos e retrieval e retorna um wrapper.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 100, "lambda_mult": 0.8},
    )

    llm = ChatGroq(
        temperature=0.1,
        model_name=LLM_MODEL_NAME,
        api_key=st.secrets["GROQ_API_KEY"],
    )
    
    # Prompt padrão com contexto e instruções
    template = """
Você é um assistente jurídico especializado. Analise cuidadosamente o seguinte contexto extraído de documentos jurídicos e responda de forma objetiva, sem adicionar informações externas.

Se não souber a resposta, diga "Não encontrei informações suficientes no documento."

<context>
{context}
</context>

Pergunta: {input}

Resposta:
"""
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    class RagChainWrapper:
        """
        5° - Wrapper que:
             a) Estima tokens e alerta se próximo do limite
             b) Invoca o chain original
             c) Exibe metadados e trechos do contexto recuperado
             d) Formata a resposta final
        """
        def __init__(self, chain):
            self._chain = chain

        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            # 1) Estima tokens do prompt
            approx = count_tokens(
                template.format(
                    context="",
                    input=inputs.get("input", "")
                ),
                model_name=EMBEDDING_MODEL_NAME
            )
            if approx > TOKEN_LIMIT:
                st.warning(f"⚠️ Prompt estimado em ~{approx} tokens (limite {TOKEN_LIMIT}).")

            # 2) Invoca o chain original
            output = self._chain.invoke(inputs)

            # 3) Exibe metadados e trechos de contexto recuperado
            if "context" in output:
                st.write("🔍 Documentos recuperados:")
                for doc in output["context"]:
                    #st.write(f"{extract_metadata(doc)} — {doc.page_content[:200]}…")
                    st.write(f"{extract_metadata(doc)} — {doc.page_content}")

            # 4) Pós‑processa a resposta
            if "answer" in output:
                output["answer"] = format_response(output["answer"])

            return output

        __call__ = invoke # permite usar o wrapper como função

    return RagChainWrapper(retrieval_chain)


@log_time
def process_document(file_path: str) -> Any:
    """
    6° - Pipeline completo:
         1) Carrega e prefixa documentos
         2) (Opcional) Chunk splitting
         3) Cria/recupera vectorstore FAISS
         4) Constrói e retorna a cadeia RAG personalizada
    """
    # 1) Load & prefix
    docs = load_documents_with_docling(file_path)
    docs = prefix_documents_for_e5(docs)

    # 2) (opcional) split: use este passo somente se
    #    – você estiver carregando o documento como um único bloco de texto (por ex. via PdfPlumber / PyPDF2),
    #    – ou se quiser forçar chunks de tamanho fixo diferentes dos produzidos pelo Docling;
    #    Caso contrário, o DoclingLoader com export_type=DOC_CHUNKS já retorna trechos bem estruturados
    #    (títulos, parágrafos, tabelas) e não vale a pena requebrá-los com RecursiveCharacterTextSplitter.
    #
    # Para ativar, basta descomentar a linha abaixo:
    # docs = split_text_into_chunks(docs)

    # 3) embeddings e vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vs = create_or_load_vectorstore(file_path, docs, embeddings)

    # 4) Cria e retorna o wrapper da chain
    return create_rag_chain(vs)
