"""
analyze.py
POST /analyze endpoint — the core API route.
Combines baseline model prediction with LLM analysis.
"""

from fastapi import APIRouter, HTTPException
from nlp_cap.schemas.review_schema import (
    ReviewRequest,
    AnalysisResponse,
    AspectSentimentResponse,
)
from nlp_cap.services.baseline_model import predict_sentiment
from nlp_cap.services.llm_chain import analyze_review

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze a product review",
    description="Analyzes a product review to determine overall sentiment, "
    "aspect-level sentiments, reasoning, and feature recommendations.",
)
async def analyze(request: ReviewRequest):
    """
    Analyze a product review.

    1. Baseline model predicts overall sentiment (fast, deterministic)
    2. LLM pipeline extracts aspects, reasoning, and recommendations
    3. Results are merged into a single response
    """
    try:
        # Step 1: Baseline model — overall sentiment
        overall_sentiment, confidence = predict_sentiment(request.review_text)

        # Step 2: LLM pipeline — aspects, reasoning, recommendations
        try:
            llm_result = await analyze_review(request.review_text)

            # Step 3: Merge results
            response = AnalysisResponse(
                overall_sentiment=overall_sentiment,
                baseline_model_confidence=confidence,
                reasoning=llm_result.reasoning,
                aspects=[
                    AspectSentimentResponse(
                        feature=aspect.feature,
                        sentiment=aspect.sentiment,
                    )
                    for aspect in llm_result.aspects
                ],
                recommended_features=llm_result.recommended_features,
            )
            return response

        except Exception as llm_error:
            # Fallback mechanism: if LLM fails, still return the baseline model's results
            print(f"[!] LLM Analysis failed: {llm_error}. Falling back to baseline model.")
            
            return AnalysisResponse(
                overall_sentiment=overall_sentiment,
                baseline_model_confidence=confidence,
                reasoning=f"The overall sentiment is {overall_sentiment}. (Note: Detailed LLM analysis is currently unavailable, showing fallback baseline model results).",
                aspects=[],
                recommended_features=["(AI recommendations currently unavailable)"]
            )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Configuration error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )
