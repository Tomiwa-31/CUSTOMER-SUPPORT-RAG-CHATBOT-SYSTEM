#chain.py   = HOW the AI thinks
#main.py    = HOW the user talks to it

# main.py

from src.chain import build_rag_chain, trim_history
from src.config import MAX_HISTORY_LENGTH
from langchain_core.messages import HumanMessage, AIMessage
from src.logger import RequestLogger

# Update run_query()
def run_query(chain, question: str, chat_history: list) -> str:
    logger = RequestLogger()

    logger.info("request_started", question=question)

    print(f"\n{'='*60}")
    print(f"Customer: {question}")
    print(f"{'='*60}")

    response = chain.invoke({
        "question": question,
        "chat_history": trim_history(chat_history, MAX_HISTORY_LENGTH),
        "logger": logger
    })

    logger.info("request_completed",
        answer_length=len(response),
        latency_ms=logger.latency_ms()
    )

    print(f"Agent: {response}")
    print(f"{'='*60}")
    return response


def main():
    print("\n🚀 Starting NovaBuy Customer Support RAG System — L2\n")

    # Build chain once at startup
    chain = build_rag_chain()

    # Session chat history
    chat_history = []

    print("\n✅ System ready! Type your question or 'quit' to exit.\n")

    while True:
        question = input("Customer: ").strip()

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Shutting down NovaBuy support agent. Goodbye!")
            break

        # Get response
        response = run_query(chain, question, chat_history)

        # Update history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    
    main()


