import streamlit as st
import os

# Título do projeto
st.set_page_config(page_title="RAG Jurídico", layout="centered")
st.title("📚 RAG Jurídico - Análise Inteligente de Documentos Jurídicos")

# Upload do documento
uploaded_file = st.file_uploader("📎 Envie um documento jurídico em PDF", type=["pdf"])

# Verifica se o arquivo foi enviado
if uploaded_file is not None:
    # Salva o arquivo na pasta data/documentos/
    file_path = os.path.join("data", "documentos", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Documento '{uploaded_file.name}' salvo com sucesso!")

    # Exibir nome e botão para processar
    st.write(f"📄 Documento carregado: `{uploaded_file.name}`")

    # Botão para iniciar o processamento com RAG
    if st.button("🔍 Analisar documento"):
        st.info("🚧 Em breve o processamento com RAG será iniciado aqui.")

        # função do rag_pipeline:
        # from rag_pipeline import processar_documento
        # processar_documento(file_path)

# Campo de pergunta (para quando o RAG estiver implementado)
st.divider()
st.subheader("🤖 Pergunte algo sobre o documento:")
pergunta = st.text_input("Digite sua pergunta aqui...")

if pergunta:
    st.info("⏳ Aguarde... integração com o RAG em breve.")
    # Quando o RAG estiver pronto:
    # resposta = gerar_resposta(pergunta, file_path)
    # st.write(resposta)
