import pickle
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    CHROMA_DIR,
    PINECONE_INDEX_NAME,
    ACCOUNT_NAMESPACE,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    SEARCH_TYPE,
    SEMANTIC_TOP_K,
    BM25_TOP_K,
    ENSEMBLE_WEIGHTS,
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    ACCOUNT_PROMPT,
)
from src.rewriter import build_query_rewriter
from src.ingestion import download_bm25_from_gcs, get_bm25_path
 
 
def build_account_agent():
 
    # ── Embedding model ────────────────────────────────
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )
 
    # ── Pinecone retriever ─────────────────────────────
    vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    namespace=ACCOUNT_NAMESPACE
)
    semantic_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": SEMANTIC_TOP_K}
    )
 
    # ── BM25 retriever ─────────────────────────────────
    
    bm25_retriever   = download_bm25_from_gcs(ACCOUNT_NAMESPACE)

    # ── Ensemble retriever ─────────────────────────────
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=ENSEMBLE_WEIGHTS
    )
 
    # ── Query rewriter ─────────────────────────────────
    rewriter = build_query_rewriter()
 
    # ── LLM ───────────────────────────────────────────
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE
    )
 
    # ── Prompt ─────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", ACCOUNT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
 
    # ── Format docs ────────────────────────────────────
    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('Header2', 'General')}]\n{doc.page_content}"
            for doc in docs
        )
 
    # ── Rewrite and retrieve ───────────────────────────
    def rewrite_and_retrieve(inputs):
        question = inputs["question"]
        logger = inputs.get("logger")
 
        rewritten = rewriter.invoke({"question": question})
        if logger:
            logger.info("account_query_rewritten",
                original_query=question,
                rewritten_query=rewritten,
                agent="account"
            )
 
        docs = retriever.invoke(rewritten)
        if logger:
            logger.info("account_chunks_retrieved",
                chunk_count=len(docs),
                agent="account"
            )
 
        return format_docs(docs)
 
    # ── RAG chain ──────────────────────────────────────
    agent_chain = (
        {
            "context": RunnableLambda(rewrite_and_retrieve),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnablePassthrough() | RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
 
    print(f"✅ Account agent ready — collection: {ACCOUNT_NAMESPACE}")
    return agent_chain
 
 
if __name__ == "__main__":
    agent = build_account_agent()
 
    test_queries = [
        "i cant log into my account",
        "how do i change my password",
        "how do i delete my account and personal data",
    ]
 
    print("\n--- Account Agent Tests ---")
    for query in test_queries:
        print(f"\nCustomer: {query}")
        response = agent.invoke({
            "question": query,
            "chat_history": [],
        })
        print(f"Agent: {response}")
 