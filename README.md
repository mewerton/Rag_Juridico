# 🧠 rag_juridico

Projeto de RAG (Retrieval-Augmented Generation) voltado para análise de documentos jurídicos com uso de modelos de linguagem (LLMs). Permite carregar arquivos jurídicos (como PDFs) ou coletar dados de APIs públicas, gerar embeddings, indexar os dados com FAISS e realizar perguntas em linguagem natural com respostas baseadas no conteúdo.

---

## 🚀 Objetivo

Demonstrar como a técnica de RAG pode ser aplicada no setor jurídico para facilitar a análise e extração de informações de documentos legais, como contratos, petições, decisões judiciais e pareceres.

---

## 🔧 Tecnologias Utilizadas

- Python 3.10+
- Streamlit
- LangChain
- FAISS
- Sentence-Transformers
- Llama API
- Docling (opcional)

---

## 📁 Estrutura Inicial

```
rag_juridico/
│
├── app.py                 # Interface com Streamlit
├── rag_pipeline.py        # Pipeline de ingestão, embedding, indexação e resposta
├── utils.py               # Funções auxiliares (carregamento, token count, etc.)
├── .gitignore             # Ignora venv, .env, __pycache__, etc
├── requirements.txt       # Bibliotecas necessárias
├── README.md              # Este arquivo
│
├── .streamlit/
│   └── secrets.toml       # Para deploy seguro
│
└── data/
    ├── documentos/        # PDFs ou documentos de entrada
    └── indexes/           # FAISS index gerado para os documento
```

---

## ▶️ Como executar

1. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o app:
```bash
streamlit run app.py
```

---

## 🛠️ Funcionalidades previstas

- [x] Upload de documentos jurídicos
- [x] Extração e chunking do texto
- [x] Geração de embeddings
- [x] Indexação com FAISS
- [x] Consulta via LLM com base nos documentos
- [ ] Coleta de dados de APIs públicas
- [ ] Histórico de consultas
- [ ] Geração de relatórios automáticos

---

## 👨‍💼 Desenvolvido por

**Mewerton de Melo Silva**  
Ciência de Dados | Inteligência Artificial  
Contato: [LinkedIn](https://www.linkedin.com/in/mewerton/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.