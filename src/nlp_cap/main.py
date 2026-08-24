"""
main.py
FastAPI application entrypoint for InsightLens.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from nlp_cap.routers.analyze import router as analyze_router
import os

app = FastAPI(
    title="InsightLens API",
    description="Intelligent Product Review Analysis System — "
    "Combines trained ML models with LLM-powered analysis for "
    "aspect-based sentiment, reasoning, and feature recommendations.",
    version="1.0.0",
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the analyze router
app.include_router(analyze_router, tags=["Analysis"])

# Serve frontend static files
_frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend",
)
if os.path.exists(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "InsightLens API"}
