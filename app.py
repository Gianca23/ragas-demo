"""
app.py - Implementazione del sistema RAG semplice usando LangChain, ChromaDB e Ollama (LangChain 0.3+).
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Modulo ufficiale dedicato aggiornato per Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings

DOCUMENT_PATH = "documents/manuale.txt"
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

def get_rag_chain():
    # 1. Caricamento del documento
    loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
    docs = loader.load()

    # 2. Chunking del testo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )
    splits = text_splitter.split_documents(docs)

    # 3. Embeddings e Vector Database (ChromaDB)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Creazione o caricamento di ChromaDB
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. Definizione del modello LLM (Ollama) e del Prompt Template
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    template = """Sei un assistente per le risposte alle domande basato esclusivamente sul contesto fornito.
Rispondi alla domanda dell'utente utilizzando unicamente il contesto fornito di seguito.
Se la risposta non è presente nel contesto, rispondi con "Informazione non presente nei documenti". Non inventare dettagli.

Contesto:
{context}

Domanda: {question}

Risposta:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Helper per formattare i documenti recuperati
    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    # 5. Costruzione della catena RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def main():
    print("🤖 Inizializzazione della catena RAG...")
    chain, retriever = get_rag_chain()
    print("✅ Sistema pronto! Digita 'exit' per uscire.\n")

    while True:
        query = input("Domanda: ")
        if query.lower() in ["exit", "quit", "esci"]:
            break
        if not query.strip():
            continue

        print("🔍 Ricerca documenti in corso...")
        docs = retriever.invoke(query)
        print("--- Contesto recuperato ---")
        for i, doc in enumerate(docs, 1):
            print(f"[{i}] {doc.page_content.strip()}")
        print("---------------------------")

        print("🤖 Risposta LLM:")
        response = chain.invoke(query)
        print(response)
        print("="*50)

if __name__ == "__main__":
    main()