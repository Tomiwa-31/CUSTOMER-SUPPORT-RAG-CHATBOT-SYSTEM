# src/ingestion.py

import pickle
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
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
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HEADERS_TO_SPLIT_ON,
    BM25_TOP_K,
)

# Path to save BM25 index on disk
BM25_INDEX_PATH = "./bm25_index.pkl"


def load_documents():
    loader = DirectoryLoader(
        path=DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents")
    return docs


def chunk_documents(docs):
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


def build_vectorstore(all_splits):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Stored {vectorstore._collection.count()} chunks in ChromaDB")
    return vectorstore


def build_bm25_index(all_splits):
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = BM25_TOP_K

    # Save to disk so we don't rebuild every restart
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"✅ BM25 index built and saved to {BM25_INDEX_PATH}")
    return bm25_retriever


def run_ingestion():
    docs = load_documents()
    all_splits = chunk_documents(docs)
    vectorstore = build_vectorstore(all_splits)
    bm25_retriever = build_bm25_index(all_splits)#contains all the chunk with a frequency word count of all individual words in it and also how many chunks should be returned
    return vectorstore, bm25_retriever


if __name__ == "__main__":
    run_ingestion()