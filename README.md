# 🤖 Ragas Demo: Sistema RAG Locale con Ollama e Valutazione tramite Ragas

Questo progetto mostra come costruire e valutare una pipeline **RAG
(Retrieval-Augmented Generation)** completamente locale utilizzando
modelli AI eseguiti sul proprio computer.

L'obiettivo non è solamente creare un chatbot AI, ma capire come
misurare la qualità delle risposte generate.

Un sistema RAG può infatti: - recuperare documenti non pertinenti; -
generare informazioni non presenti nei dati; - produrre risposte
incomplete.

Per questo motivo utilizziamo **Ragas**, una libreria che permette di
valutare automaticamente la qualità di un sistema RAG.

------------------------------------------------------------------------

## 🚀 Cosa costruisce questo progetto

Il progetto realizza un piccolo assistente AI capace di:

1.  Leggere documenti locali.
2.  Trasformare il contenuto in embeddings.
3.  Salvare le informazioni in un database vettoriale.
4.  Recuperare il contesto più rilevante.
5.  Generare una risposta tramite un LLM locale.
6.  Valutare il sistema tramite Ragas.

L'intera pipeline funziona localmente e nessun dato viene inviato a
servizi cloud.

------------------------------------------------------------------------

## 🏗️ Architettura

``` text
Documento
    │
    ▼
Document Loader
    │
    ▼
Chunking
    │
    ▼
Embedding Model (nomic-embed-text)
    │
    ▼
ChromaDB
    │
    ▼
Retriever
    │
    ▼
LLM Locale (llama3)
    │
    ▼
Risposta
    │
    ▼
Ragas
    │
    ▼
Metriche
```

------------------------------------------------------------------------

## 🧰 Stack Tecnologico

  Componente        Tecnologia
  ----------------- ---------------------------
  Linguaggio        Python 3.10+
  Framework RAG     LangChain 0.3+
  LLM               Ollama + llama3
  Embeddings        Ollama + nomic-embed-text
  Vector Database   ChromaDB
  Valutazione       Ragas

------------------------------------------------------------------------

## 📋 Prerequisiti

### Python

``` bash
python --version
```

### Ollama

Installare Ollama e scaricare i modelli:

``` bash
ollama pull llama3
ollama pull nomic-embed-text
```

------------------------------------------------------------------------

## 🚀 Installazione

``` bash
git clone https://github.com/tuo-username/ragas-demo.git
cd ragas-demo
```

### Ambiente virtuale

**Windows**

``` powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### Dipendenze

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 💻 Utilizzo

### 1. Indicizzazione documenti

``` bash
python ingest.py
```

Questo comando: - legge i documenti; - crea i chunk; - genera gli
embeddings; - salva tutto in ChromaDB.

### 2. Avvio del chatbot

``` bash
python app.py
```

### 3. Valutazione

``` bash
python evaluate.py
```

------------------------------------------------------------------------

## 📈 Metriche Ragas

-   **Faithfulness** -- verifica che la risposta sia supportata dai
    documenti.
-   **Answer Relevancy** -- misura quanto la risposta soddisfa la
    domanda.
-   **Context Precision** -- valuta se i documenti recuperati sono
    pertinenti.
-   **Context Recall** -- misura se il contesto recuperato è completo.

------------------------------------------------------------------------

## 📁 Struttura del progetto

``` text
ragas-demo/
├── documents/
├── chroma_db/
├── src/
├── app.py
├── ingest.py
├── evaluate.py
├── dataset.json
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 🎯 Cosa imparerai

-   Cos'è un sistema RAG.
-   Come funzionano gli embeddings.
-   Come utilizzare un vector database.
-   Come collegare un LLM ai propri dati.
-   Come valutare un RAG con Ragas.

------------------------------------------------------------------------

## 📚 Articolo

Questo repository accompagna l'articolo:

**"Il mio chatbot AI sembrava intelligente... poi ho scoperto che
sbagliava"**
