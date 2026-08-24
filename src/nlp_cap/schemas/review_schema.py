"""
review_schema.py
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    review_text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The product review text to analyze",
        examples=["The battery life is terrible but the camera quality is amazing for the price."],
    )


class AspectSentimentResponse(BaseModel):
    """Sentiment for a single product aspect."""
    feature: str = Field(description="Product feature or aspect")
    sentiment: str = Field(description="Positive, Negative, or Neutral")


class AnalysisResponse(BaseModel):
    """Complete analysis response combining baseline model + LLM results."""
    overall_sentiment: str = Field(
        description="Overall sentiment from the trained baseline model: Positive or Negative"
    )
    baseline_model_confidence: float = Field(
        description="Confidence score of the baseline model prediction (0.0 to 1.0)"
    )
    reasoning: str = Field(
        description="Plain-language explanation of why the review is positive/negative"
    )
    aspects: list[AspectSentimentResponse] = Field(
        description="Per-feature sentiment breakdown"
    )
    recommended_features: list[str] = Field(
        description="AI-generated product feature recommendations"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(description="Error message")
