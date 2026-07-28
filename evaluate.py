"""
evaluate.py - Valutazione RAG con Ragas ed Ollama senza TimeoutError.
"""

import sys
import types
import json
import warnings
import pandas as pd
from datasets import Dataset

# Silenzia i warning
warnings.filterwarnings("ignore")

# 🛠️ FIX IMPORTS VENDOR: Bypassa l'import legacy di VertexAI
try:
    import langchain_community.chat_models.vertexai
except (ImportError, ModuleNotFoundError):
    dummy_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_mod.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_mod

# Import Ragas
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Integration LangChain 0.3+ per Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Import dal tuo file app.py
from app import get_rag_chain, LLM_MODEL, EMBEDDING_MODEL

def run_evaluation():
    print("⚙️ Inizializzazione della catena RAG...")
    rag_chain, retriever = get_rag_chain()

    # 1. Caricamento del dataset
    with open("dataset.json", "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    print("🚀 Generazione risposte e contesti dal RAG...")
    for item in test_dataset:
        q = item["question"]
        gt = item["ground_truth"]

        # Recupero documenti
        retrieved_docs = retriever.invoke(q)
        context_list = [doc.page_content for doc in retrieved_docs]

        # Risposta RAG
        answer = rag_chain.invoke(q)

        questions.append(q)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(gt)

    # 2. Strutturazione dataset per Ragas
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }

    eval_dataset = Dataset.from_dict(data)

    print("📊 Setup dei modelli giudici Ollama (Timeout esteso a 300s)...")
    
    # Impostiamo il timeout a 300 secondi per evitare l'interruzione dei job
    ollama_llm = OllamaLLM(
        model=LLM_MODEL, 
        temperature=0,
        timeout=300
    )
    ollama_emb = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    evaluator_llm = LangchainLLMWrapper(ollama_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(ollama_emb)

    # 3. Lista delle metriche
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    for metric in metrics:
        metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

    # 4. Configurazione Esecuzione Seriale (evita il sovraccarico di Ollama)
    run_config = RunConfig(
        max_workers=1,      # Esegue 1 valutazione alla volta (no concorrenza)
        timeout=300         # Timeout globale di 5 minuti per task
    )

    print("📈 Esecuzione della valutazione Ragas (Esecuzione sequenziale)...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    print("\n" + "="*50)
    print("📈 RISULTATI VALUTAZIONE RAGAS")
    print("="*50)
    
    df_results = results.to_pandas()
    print(df_results[['user_input', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])
    
    df_results.to_csv("valutazione_risultati.csv", index=False)
    print("\n💾 Esito salvato con successo in 'valutazione_risultati.csv'.")

if __name__ == "__main__":
    run_evaluation()