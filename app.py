import os
os.environ["STREAMLIT_WATCHER_PATCHED_MODULES"] = "torch"
import streamlit as st


from rag_pipeline import process_document

# Título do projeto
st.set_page_config(page_title="RAG Jurídico", layout="wide")
st.title("📚 RAG Jurídico")
st.title("Análise de Documentos Jurídicos")

# Inicializar o session_state para guardar o RAG chain
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "document_path" not in st.session_state:
    st.session_state.document_path = None

# Upload do documento
uploaded_file = st.file_uploader("📎 Envie um documento jurídico em PDF", type=["pdf"])

if uploaded_file is not None:
    # Garantir que o diretório existe
    os.makedirs("data/documentos", exist_ok=True)
    
    # Salvar o arquivo na pasta data/documentos/
    file_path = os.path.join("data", "documentos", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Documento '{uploaded_file.name}' salvo com sucesso!")

    # Guardar o caminho no session_state
    st.session_state.document_path = file_path

    # Mostrar botão para processar o documento
    if st.button("🔍 Analisar documento"):
        with st.spinner("Processando o documento, aguarde... ⏳"):
            # Processar o documento usando a pipeline
            st.session_state.rag_chain = process_document(file_path)
        st.success("✅ Documento processado com sucesso! Agora você pode fazer perguntas.")

# Separador
st.divider()

# Campo de pergunta
st.subheader("🤖 Pergunte algo sobre o documento:")

if st.session_state.rag_chain:
    pergunta = st.text_input("Digite sua pergunta aqui...")

    if pergunta:
        with st.spinner("Consultando o documento... 🤖"):
            resposta = st.session_state.rag_chain.invoke({"input": pergunta})
            st.subheader("📄 Resposta da IA:")
            st.write(resposta["answer"])
else:
    st.info("📎 Faça o upload e análise de um documento para poder perguntar.")