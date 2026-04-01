# src/ingestion.py

import pickle
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from src.config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_DOC_MAP,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HEADERS_TO_SPLIT_ON,
    BM25_TOP_K,
)

# BM25 index path per collection
def get_bm25_path(collection_name: str) -> str:
    return f"./bm25_{collection_name}.pkl"


def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )


def load_documents_for_collection(collection_name: str) -> list:
    """Load only the docs assigned to this collection"""
    docs = []
    filenames = COLLECTION_DOC_MAP[collection_name]

    for filename in filenames:
        filepath = Path(DATA_DIR) / filename
        if not filepath.exists():
            print(f" File not found: {filepath}")
            continue
        loader = TextLoader(str(filepath))
        docs.extend(loader.load())

    print(f"✅ Loaded {len(docs)} documents for {collection_name}")
    return docs


def chunk_documents(docs: list) -> list:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_splits = []
    for doc in docs:
        md_splits = markdown_splitter.split_text(doc.page_content)
        further_splits = text_splitter.split_documents(md_splits)
        all_splits.extend(further_splits)

    print(f"✅ Created {len(all_splits)} chunks")
    return all_splits


def build_vectorstore(all_splits: list, collection_name: str, embedding_model):
    """Build ChromaDB vectorstore for a specific collection"""
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name
    )
    print(f"✅ Stored {vectorstore._collection.count()} chunks in {collection_name}")
    return vectorstore


def build_bm25_index(all_splits: list, collection_name: str):
    """Build and save BM25 index for a specific collection"""
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = BM25_TOP_K

    bm25_path = get_bm25_path(collection_name)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f" BM25 index saved to {bm25_path}")
    return bm25_retriever


def run_ingestion():
    """Ingest all documents into their respective collections"""
    embedding_model = load_embedding_model()

    for collection_name in COLLECTION_DOC_MAP:
        print(f"\n Processing collection: {collection_name}")

        # Check if collection already exists
        existing = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
        if existing._collection.count() > 0:
            print(f" {collection_name} already exists — skipping")
            continue

        # Load, chunk, embed, store
        docs = load_documents_for_collection(collection_name)
        if not docs:
            print(f" No documents found for {collection_name} — skipping")
            continue

        all_splits = chunk_documents(docs)
        build_vectorstore(all_splits, collection_name, embedding_model)
        build_bm25_index(all_splits, collection_name)

    print("\n All collections ingested successfully!")


if __name__ == "__main__":
    run_ingestion()