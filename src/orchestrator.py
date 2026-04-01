# src/orchestrator.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from src.router import build_router, classify_question
from src.agents.billing import build_billing_agent
from src.agents.technical import build_technical_agent
from src.agents.account import build_account_agent
from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    GENERAL_PROMPT,
)


class Orchestrator:
    """
    Receives a question → classifies via router → delegates to correct agent.
    Exposes an .invoke() method so it's a drop-in replacement for the RAG chain
    in run_with_memory().
    """

    def __init__(self):
        print("🔄 Building router...")
        self.router = build_router()

        print("🔄 Building billing agent...")
        self.billing_agent = build_billing_agent()

        print("🔄 Building technical agent...")
        self.technical_agent = build_technical_agent()

        print("🔄 Building account agent...")
        self.account_agent = build_account_agent()

        print("🔄 Building general fallback agent...")
        self.general_agent = self._build_general_agent()

        print("✅ Orchestrator ready — all agents loaded")

    def _build_general_agent(self):
        """Fallback agent for questions that don't fit billing/technical/account"""
        llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=TEMPERATURE
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        # General agent has no retriever — LLM answers directly from prompt
        # context is empty since no collection is assigned to general
        return (
            prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, inputs: dict) -> str:
        """
        Drop-in replacement for chain.invoke().
        Accepts: { question, chat_history, logger }
        Returns: str response
        """
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        logger = inputs.get("logger")

        # ── Step 1: classify ──────────────────────────────
        category = classify_question(self.router, question)

        if logger:
            logger.info("router_classified",
                question=question,
                category=category
            )

        # ── Step 2: delegate to correct agent ─────────────
        agent_inputs = {
            "question": question,
            "chat_history": chat_history,
            "logger": logger,
        }

        if category == "billing":
            response = self.billing_agent.invoke(agent_inputs)

        elif category == "technical":
            response = self.technical_agent.invoke(agent_inputs)

        elif category == "account":
            response = self.account_agent.invoke(agent_inputs)

        else:
            # general — no retriever, LLM answers directly
            # pass empty context since GENERAL_PROMPT expects {context}
            general_inputs = {
                "question": question,
                "chat_history": chat_history,
                "context": "",
            }
            response = self.general_agent.invoke(general_inputs)

        if logger:
            logger.info("agent_responded",
                category=category,
                answer_length=len(response)
            )

        return response


def build_orchestrator() -> Orchestrator:
    """Factory function — called once at startup in app.py lifespan"""
    return Orchestrator()