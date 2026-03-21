# src/chain.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from src.retriever import get_retriever
from src.rewriter import build_query_rewriter
from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    MAX_HISTORY_LENGTH,
)
from src.logger import RequestLogger

def load_llm():
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE
    )
    print(f"✅ LLM ready — {LLM_MODEL}")
    return llm


def load_prompt():
    template = """You are a helpful customer support agent for NovaBuy, an ecommerce store.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information \
to answer that, let me escalate this to a human agent."
Do not make up answers.

Context:
{context}

"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    print("✅ Prompt template ready")
    return prompt


def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('Header2', 'General')}]\n{doc.page_content}"
        for doc in docs
    )




def build_rag_chain():#think of it as activating the retriever,rewritter,llm and prompt
    retriever = get_retriever()
    rewriter = build_query_rewriter()
    llm = load_llm()
    prompt = load_prompt()

    def rewrite_and_retrieve(inputs):
        logger = inputs.get("logger")
        question = inputs["question"]

        rewritten = rewriter.invoke({"question": question})
        if logger:
            logger.info("query_rewritten",
                original_query=question,
                rewritten_query=rewritten
            )

        docs = retriever.invoke(rewritten)
        if logger:
            logger.info("chunks_retrieved",
                chunk_count=len(docs),
                sections=[d.metadata.get("Header2", "General") for d in docs]
            )

        return format_docs(docs)

    rag_chain = (
        {
            "context": RunnableLambda(rewrite_and_retrieve),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnablePassthrough() | RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready")
    return rag_chain


def trim_history(chat_history: list, max_length: int) -> list:
    """Keep only the last max_length messages"""
    return chat_history[-max_length:]


if __name__ == "__main__":
    chain = build_rag_chain()
    chat_history = []

    test_queries = [
        "how long do i have 2 return somthing",
        "what about damaged items",
        "who pays the return shipping",
    ]

    print("\n--- L2 RAG Chain Tests ---")
    for query in test_queries:
        print(f"\nCustomer: {query}")

        response = chain.invoke({
            "question": query,
            "chat_history": trim_history(chat_history, MAX_HISTORY_LENGTH)
        })

        print(f"Agent: {response}")

        # Update history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response))


### Clean separation across all three files:

#config.py       →  all settings, no logic
#ingestion.py    →  load, chunk, embed, store  (run ONCE)
#retriever.py    →  load vectorstore, build retriever
#chain.py        →  llm + prompt + rag chain

#each file only knows about config.py and the file directly above it




### Notice the dependency chain is clean and one directional:

#config.py
    #↑
#retriever.py
   # ↑
#chain.py
   # ↑
#main.py   ← next file, ties everything together