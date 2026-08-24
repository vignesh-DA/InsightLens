"""
groq_client.py
Initializes the Groq LLM client via LangChain.
"""

from langchain_groq import ChatGroq
from nlp_cap.config import GROQ_API_KEY


def get_llm() -> ChatGroq:
    """
    Create and return a configured ChatGroq instance.

    Uses llama-3.3-70b-versatile model for high-quality
    aspect extraction, reasoning, and recommendations.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your Groq API key."
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b",
        temperature=0.3,  # Low temp for consistent structured output
        max_tokens=2048,
    )
