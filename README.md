# Rag_project_
This notebook is designed to be a versatile tool for querying information from any PDF document. By simply uploading a PDF, users can ask questions about its content and receive relevant answers based on the document's text.  For demonstration purposes, this project utilizes the LPU Dress Code Policy PDF as an example. 

# PDF RAG with Groq

A simple Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about a PDF document. It uses local embeddings for semantic search and Groq's fast LLM inference to generate answers grounded in the document's content.

## How It Works

1. A PDF is uploaded and split into overlapping text chunks.
2. Each chunk is embedded using a local sentence-transformers model.
3. Embeddings are stored in a FAISS vector index for similarity search.
4. When you ask a question, the most relevant chunks are retrieved and passed to a Groq-hosted LLM (Llama 3.1) as context.
5. The LLM answers using only the retrieved context.

## Tech Stack

- **LLM inference:** [Groq](https://groq.com/) (`llama-3.1-8b-instant`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free)
- **Vector store:** FAISS
- **Orchestration:** LangChain (LCEL / runnables)
- **PDF parsing:** PyPDF

## Requirements

- Python environment with internet access (Google Colab recommended)
- A free [Groq API key](https://console.groq.com/keys)

## Setup

### 1. Install dependencies

```bash
pip install -q langchain langchain-community langchain-groq langchain-text-splitters langchain-huggingface pypdf faiss-cpu sentence-transformers
```

### 2. Set your Groq API key

**In Google Colab**, use the built-in Secrets manager (🔑 icon in the left sidebar) instead of hardcoding your key:

```python
from google.colab import userdata
import os

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

**Locally**, set it as an environment variable instead:

```bash
export GROQ_API_KEY="your-key-here"
```

> ⚠️ Never commit your API key to source control or paste it into shared files/chats.

### 3. Upload / provide a PDF

In Colab:

```python
from google.colab import files

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
```

Locally, just set `pdf_path` to the path of your PDF file.

### 4. Load and split the PDF

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
```

### 5. Build the vector store

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

### 6. Build the RAG chain

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### 7. Ask questions

```python
query = "What is this document about?"
answer = rag_chain.invoke(query)
print("Answer:", answer)
```

To also see which chunks were used to answer:

```python
source_docs = retriever.invoke(query)
for i, doc in enumerate(source_docs, 1):
    print(f"[{i}] Page {doc.metadata.get('page', '?')}: {doc.page_content[:150]}...")
```

## Project Structure

```
.
├── notebook.ipynb   # Main Colab/Jupyter notebook
└── README.md
```

## Notes

- Currently supports **one PDF at a time**. To support multiple PDFs, load and merge documents from a folder before splitting.
- `langchain-community`, `langchain.chains`, and similar module paths change frequently as LangChain evolves — if you hit `ModuleNotFoundError`, check the [LangChain docs](https://python.langchain.com/) for the current import path for your installed version.
- The embedding model runs locally and is free; only the LLM call goes through the Groq API.

## Roadmap

- [ ] Support multiple PDFs
- [ ] Add source citations in the answer output
- [ ] Add a simple chat loop for follow-up questions
- [ ] Add a Gradio/Streamlit UI

## License

MIT
