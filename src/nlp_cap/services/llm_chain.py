"""
llm_chain.py
LangChain pipeline that uses Groq to perform:
1. Aspect extraction (product features mentioned)
2. Per-aspect sentiment analysis
3. Plain-language reasoning
4. Feature recommendations based on complaints/gaps

Uses PydanticOutputParser for strict JSON output.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from nlp_cap.services.groq_client import get_llm


# --- Pydantic models for structured output ---

class AspectSentiment(BaseModel):
    """Sentiment for a single product aspect/feature."""
    feature: str = Field(description="The product feature or aspect mentioned (e.g., battery, camera, price)")
    sentiment: str = Field(description="Sentiment for this feature: Positive, Negative, or Neutral")


class LLMAnalysisResult(BaseModel):
    """Complete analysis result from the LLM pipeline."""
    aspects: list[AspectSentiment] = Field(
        description="List of product aspects/features mentioned in the review, each with its sentiment"
    )
    reasoning: str = Field(
        description="A plain-language explanation of WHY the review is positive or negative. "
        "This should explain the reviewer's reasoning, not just restate the sentiment."
    )
    recommended_features: list[str] = Field(
        description="2-3 specific, actionable product feature recommendations or improvements "
        "based on the complaints or gaps identified in the review"
    )


# --- Output parser ---
_parser = PydanticOutputParser(pydantic_object=LLMAnalysisResult)

# --- Prompt template ---
_ANALYSIS_PROMPT = PromptTemplate(
    template="""You are an expert product analyst. Analyze the following product review and provide a structured analysis.

PRODUCT REVIEW:
"{review_text}"

Perform ALL of the following steps:

1. **ASPECT EXTRACTION**: Identify every product feature or aspect the reviewer mentions (e.g., battery, camera, price, build quality, shipping, customer service, screen, performance, etc.). Be thorough — extract ALL mentioned aspects, even implicit ones.

2. **PER-ASPECT SENTIMENT**: For each aspect you extracted, determine if the reviewer's sentiment about it is Positive, Negative, or Neutral.

3. **REASONING**: Write a clear, plain-language explanation (2-3 sentences) of WHY this review is positive or negative overall. Don't just say "it's negative" — explain the reviewer's key concerns or praise. Reference specific aspects.

4. **RECOMMENDATIONS**: Based on the complaints, frustrations, or gaps in this review, suggest 2-3 specific, actionable new product features or improvements that the product team could implement. Each recommendation should directly address a problem raised in the review. If the review is entirely positive, suggest features that would make the product even better.

{format_instructions}

Respond ONLY with the JSON object. No additional text before or after.""",
    input_variables=["review_text"],
    partial_variables={"format_instructions": _parser.get_format_instructions()},
)


async def analyze_review(review_text: str) -> LLMAnalysisResult:
    """
    Run the full LLM analysis pipeline on a review.

    Args:
        review_text: The product review to analyze.

    Returns:
        LLMAnalysisResult with aspects, reasoning, and recommendations.
    """
    llm = get_llm()

    # Build and invoke the chain
    chain = _ANALYSIS_PROMPT | llm | _parser
    result = await chain.ainvoke({"review_text": review_text})

    return result
