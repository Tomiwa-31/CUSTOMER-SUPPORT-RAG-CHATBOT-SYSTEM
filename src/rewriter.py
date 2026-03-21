# src/rewriter.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE


def build_query_rewriter():
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE
    )

    template = """You are a query rewriter for an ecommerce customer support system.
Your job is to rewrite the customer's question into a clear, formal query that 
matches the language used in policy documents.

Rules:
- Fix spelling mistakes
- Convert casual language to formal policy language
- Keep it concise — one sentence only
- Do not answer the question, just rewrite it

Examples:
Customer: "how long do i have to send stuff back"
Rewritten: "What is the return eligibility window and timeframe for returning items?"

Customer: "my thing arrived broken wht do i do"
Rewritten: "What is the process for returning a damaged or defective item on arrival?"

Customer: "do u ship to uk"
Rewritten: "Does NovaBuy offer international shipping to the United Kingdom?"

Now rewrite this:
Customer: {question}
Rewritten:"""

    prompt = ChatPromptTemplate.from_template(template)

    rewriter_chain = prompt | llm | StrOutputParser()

    print("✅ Query rewriter ready")
    return rewriter_chain


if __name__ == "__main__":
    rewriter = build_query_rewriter()

    # Test it
    #test_queries = [
        #"how long do i have to return somthing",
        #"my package didnt arrive wht do i do",
        #"do u ship to australia",
        #"whts the cost 4 overnight shipping"
    #]

    #print("\n--- Query Rewriter Tests ---")
    #for query in test_queries:
        #rewritten = rewriter.invoke({"question": query})
        #print(f"\nOriginal:  {query}")
        #print(f"Rewritten: {rewritten}")
#```

#---

### Where it sits in the pipeline:
#```
#User: "how long do i have 2 return somthing"
            #↓
    #Query Rewriter
            #↓
#"What is the return eligibility window for returning items?"
            #↓
    #Hybrid Retriever (finds the right chunks now)
            #↓
    #LLM → Answer