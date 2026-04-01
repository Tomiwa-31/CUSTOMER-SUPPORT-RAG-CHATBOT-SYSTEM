from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    ROUTER_CATEGORIES,
)


def build_router():
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE
    )

    template = """You are a customer support ticket classifier for NovaBuy, an ecommerce store.
Your job is to classify the customer's question into exactly one of these categories:

- billing    → payment, refund, promo codes, loyalty points, order management, cancellations
- technical  → product compatibility, damaged items, shipping, tracking, troubleshooting
- account    → login, password, security, personal data, privacy, account settings
- general    → anything that doesn't clearly fit the above three categories

Rules:
- Respond with ONLY the category name in lowercase
- No explanation, no punctuation, just the single word
- If unsure, respond with: general

Examples:
Customer: "my payment was charged twice"          → billing
Customer: "my package arrived damaged"            → technical
Customer: "i cant log into my account"            → account
Customer: "how does novabuy work"                 → general
Customer: "how do i apply a promo code"           → billing
Customer: "when will my order arrive"             → technical
Customer: "i want to delete my account"           → account

Now classify this:
Customer: {question}
Category:"""

    prompt = ChatPromptTemplate.from_template(template)

    router_chain = prompt | llm | StrOutputParser()

    print("✅ Router ready")
    return router_chain


def classify_question(router, question: str) -> str:
    """Classify question and return category — always returns a valid category"""
    result = router.invoke({"question": question}).strip().lower()

    # Sanitize — ensure result is always a valid category
    if result not in ROUTER_CATEGORIES:
        print(f"⚠️ Router returned unknown category '{result}' — defaulting to general")
        return "general"

    print(f"🎯 Classified as: {result}")
    return result


if __name__ == "__main__":
    router = build_router()

    test_questions = [
        "my payment was charged twice",
        "my package arrived damaged what do i do",
        "i cant log into my account",
        "how long do i have to return something",
        "how do i apply a promo code",
        "do you ship to australia",
        "i want to delete my account",
        "what payment methods do you accept",
    ]

    print("\n--- Router Classification Tests ---")
    for question in test_questions:
        category = classify_question(router, question)
        print(f"Q: {question}")
        print(f"A: {category}\n")