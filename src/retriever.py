# src/retriever.py

import pickle
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
#from langchain.retrievers import EnsembleRetriever
from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    SEARCH_TYPE,
    SEMANTIC_TOP_K,
    BM25_TOP_K,
    ENSEMBLE_WEIGHTS,
)

BM25_INDEX_PATH = "./bm25_index.pkl"


def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )
    return embedding_model


def load_vectorstore(embedding_model):
    # Guard — don't let ChromaDB silently create empty folder
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            f"Run ingestion first: python -m src.ingestion"
        )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Loaded vectorstore with {vectorstore._collection.count()} chunks")
    return vectorstore

def load_semantic_retriever(vectorstore):
    semantic_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": SEMANTIC_TOP_K}
    )
    print(f"✅ Semantic retriever ready — top {SEMANTIC_TOP_K} chunks")
    return semantic_retriever


def load_bm25_retriever():
    if not Path(BM25_INDEX_PATH).exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. "
            f"Please run ingestion first: python -m src.ingestion"
        )

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)

    bm25_retriever.k = BM25_TOP_K
    print(f"✅ BM25 retriever loaded — top {BM25_TOP_K} chunks")
    return bm25_retriever


def build_ensemble_retriever(semantic_retriever, bm25_retriever):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=ENSEMBLE_WEIGHTS
    )
    print(f"✅ Ensemble retriever ready — weights {ENSEMBLE_WEIGHTS}")
    return ensemble_retriever


def get_retriever():
    embedding_model = load_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    semantic_retriever = load_semantic_retriever(vectorstore)
    bm25_retriever = load_bm25_retriever()
    ensemble_retriever = build_ensemble_retriever(semantic_retriever, bm25_retriever)
    return ensemble_retriever
#note the ensemble has no cap compared to semantic and bm35 capped at 5, it simply combines the chunk from both category but in a 
#situation with less than 10 chunk , a duplicate has occured i.e 7 chunk means 3 duplicates from both

if __name__ == "__main__":
    retriever = get_retriever()

    # Test with casual and precise queries
    #test_queries = [
        #"how long do i have to return somthing",
        #"RMA number processing time",
        #"do u ship to australia",
        #"overnight shipping cost",
    #]

    #print("\n--- Hybrid Retriever Tests ---")
    #for query in test_queries:
        #print(f"\nQuery: {query}")
        #docs = retriever.invoke(query)
        #for i, doc in enumerate(docs):
            #print(f"  Chunk {i+1}: {doc.metadata.get('Header2', 'General')}")


### What changed from L1:

#Three retrievers now work together:

#load_semantic_retriever()  →  ChromaDB (meaning)
##load_bm25_retriever()      →  pickle file (keywords)
#build_ensemble_retriever() →  combines both
#The retriever is exactly like a search bar placed on top of it — it's the interface that:

#Takes a question
#Converts it to a vector
#Goes into the vectorstore and finds the closest matching chunks
#Returns the top K results

