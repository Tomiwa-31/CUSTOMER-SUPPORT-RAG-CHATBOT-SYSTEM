# src/evaluator.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_API_KEY, LLM_MODEL
from src.config import RETRIEVAL_CONFIDENCE_THRESHOLD, LLM_JUDGE_THRESHOLD, USE_LLM_JUDGE
import json


# ── Confidence thresholds ──────────────────────────────
RETRIEVAL_CONFIDENCE_THRESHOLD =RETRIEVAL_CONFIDENCE_THRESHOLD #0.5   # below this → low confidence
LLM_JUDGE_THRESHOLD =LLM_JUDGE_THRESHOLD #3.0              # below this (out of 5) → escalate


def score_retrieval_confidence(docs_and_scores: list) -> float:
    """
    Takes list of (Document, score) tuples from similarity_search_with_score
    Returns average similarity score across top chunks
    Higher = more confident retrieval
    """
    #docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)
    if not docs_and_scores:
        return 0.0

    scores = [score for _, score in docs_and_scores]
    avg_score = sum(scores) / len(scores)

    print(f"📊 Retrieval confidence: {avg_score:.3f} "
          f"(threshold: {RETRIEVAL_CONFIDENCE_THRESHOLD})")

    return avg_score


def is_low_confidence_retrieval(confidence_score: float) -> bool:
    """Returns True if retrieval confidence is too low"""
    return confidence_score < RETRIEVAL_CONFIDENCE_THRESHOLD


class LLMJudge:
    """
    Uses a second Groq LLM call to score the agent's answer
    Scores on three dimensions:
    - Grounding: is the answer supported by the context?
    - Relevance: does it answer the actual question?
    - Completeness: is it a full answer or a partial one?
    """
#we initialize a llm and prompt template in the constructor since we can reuse them across evaluations
    def __init__(self):
        self.llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_template("""
You are an evaluator for a customer support AI system.
Score the following answer on three dimensions.
Respond ONLY with a valid JSON object — no explanation, no markdown.

Question: {question}
Context used: {context}
Answer given: {answer}

Score each dimension from 1 to 5:
- grounding: is the answer fully supported by the context? (1=hallucinated, 5=fully grounded)
- relevance: does the answer address the question? (1=irrelevant, 5=perfectly relevant)  
- completeness: is the answer complete? (1=very incomplete, 5=fully complete)

Respond with exactly this JSON format:
{{"grounding": <score>, "relevance": <score>, "completeness": <score>, "overall": <average>}}
""")

        self.chain = self.prompt | self.llm | StrOutputParser()

    def evaluate(self, question: str, context: str, answer: str) -> dict:
        """
        Returns dict with scores and whether to escalate
        """
        try:
            result = self.chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })

            # Clean and parse JSON
            result = result.strip()
            if result.startswith("```"):#strip out the backticks if llm wrapped the json in a code block
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]

            scores = json.loads(result)#converts json sting output by llm to a python dict

            overall = scores.get("overall", 0)
            should_escalate = overall < LLM_JUDGE_THRESHOLD

            print(f"🧑‍⚖️ LLM Judge scores: {scores}")
            print(f"🧑‍⚖️ Overall: {overall:.1f}/5 "
                  f"({'escalate' if should_escalate else 'pass'})")

            return {
                "scores": scores,
                "overall": overall,
                "should_escalate": should_escalate,
                "reason": "low_llm_judge_score" if should_escalate else None
            }

        except Exception as e:
            print(f" LLM Judge failed: {e} — defaulting to pass")
            return {
                "scores": {},
                "overall": 5.0,
                "should_escalate": False,
                "reason": None
            }#dont route to agent if judge fails — better to risk a bad answer than no answer at all


def check_fallback_response(answer: str) -> bool:
    """
    Detects if the agent gave a fallback 'I don't know' response
    Returns True if escalation is needed
    """
    fallback_phrases = [
        "i don't have enough information",
        "let me escalate",
        "i cannot answer",
        "i don't know",
        "no information available",
        "outside my knowledge",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in fallback_phrases)


def evaluate_response(
    question: str,
    context: str,
    answer: str,
    retrieval_confidence: float = None,
    use_llm_judge: bool = True
) -> dict:
    """
    Master evaluation function — runs all checks and returns final verdict

    Returns:
    {
        "should_escalate": bool,
        "escalation_reason": str or None,
        "confidence_score": float,
        "llm_judge_scores": dict,
    }
    """
    # Check 1 — fallback response detection
    if check_fallback_response(answer):
        print("🚨 Fallback response detected → escalating")
        return {
            "should_escalate": True,
            "escalation_reason": "fallback_response",
            "confidence_score": retrieval_confidence or 0.0,
            "llm_judge_scores": {},
        }

    # Check 2 — retrieval confidence
    if retrieval_confidence is not None:
        if is_low_confidence_retrieval(retrieval_confidence):
            print("🚨 Low retrieval confidence → escalating")
            return {
                "should_escalate": True,
                "escalation_reason": "low_retrieval_confidence",
                "confidence_score": retrieval_confidence,
                "llm_judge_scores": {},
            }

    # Check 3 — LLM judge
    if use_llm_judge:
        judge = LLMJudge()
        judge_result = judge.evaluate(question, context, answer)

        if judge_result["should_escalate"]:
            return {
                "should_escalate": True,
                "escalation_reason": "low_llm_judge_score",
                "confidence_score": retrieval_confidence or 1.0,
                "llm_judge_scores": judge_result["scores"],
            }

    # All checks passed
    return {
        "should_escalate": False,
        "escalation_reason": None,
        "confidence_score": retrieval_confidence or 1.0,
        "llm_judge_scores": judge_result["scores"] if use_llm_judge else {},
    }


if __name__ == "__main__":
    # Test the evaluator
    test_cases = [
        {
            "question": "how long do I have to return an item?",
            "context": "Return window: 30 days from delivery for unopened items.",
            "answer": "You have 30 days from delivery to return an unopened item.",
        },
        {
            "question": "what is the meaning of life?",
            "context": "NovaBuy return policy covers 30 days.",
            "answer": "I don't have enough information to answer that, let me escalate this to a human agent.",
        },
        {
            "question": "how do I track my order?",
            "context": "NovaBuy return policy covers 30 days.",
            "answer": "You can track orders by visiting our website and entering your order number.",
        },
    ]

    print("\n--- Evaluator Tests ---")
    for case in test_cases:
        print(f"\nQ: {case['question']}")
        result = evaluate_response(
            question=case["question"],
            context=case["context"],
            answer=case["answer"],
            retrieval_confidence=0.75,
        )
        print(f"Result: {result}")


### Three layers of evaluation:

#Layer 1 — Fallback detection (instant, no LLM call)
#"I don't have enough information" → escalate immediately

#Layer 2 — Retrieval confidence (from similarity scores)
#avg similarity score < 0.5 → chunks weren't relevant → escalate

#Layer 3 — LLM-as-judge (second Groq call)
#scores grounding + relevance + completeness
#overall < 3.0/5.0 → answer quality too low → escalate