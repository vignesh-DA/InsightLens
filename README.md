# 🔍 InsightLens — Intelligent Product Review Analysis System

A full-stack AI application that analyzes product reviews to provide **overall sentiment**, **aspect-based sentiment breakdown**, **plain-language reasoning**, and **AI-generated feature recommendations**.

## ✨ Key Features

- **Overall Sentiment Classification** — Trained TF-IDF + Logistic Regression baseline model for fast, deterministic predictions
- **Aspect-Based Sentiment Analysis** — Per-feature sentiment (battery ✅, price ❌) via LLM
- **Explainable Reasoning** — Plain-language explanation of *why* a review is positive/negative
- **Feature Recommendations** — AI-suggested product improvements based on reviewer complaints

## 🏗️ Architecture

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Vanilla JavaScript |
| Backend | FastAPI + Uvicorn |
| Baseline ML Model | scikit-learn (TF-IDF + Logistic Regression) |
| LLM Provider | Groq API (llama-3.3-70b-versatile) |
| Orchestration | LangChain + PydanticOutputParser |
| Dataset | Amazon Reviews Polarity (HuggingFace) |

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Free [Groq API key](https://console.groq.com/)

### 2. Install Dependencies

```bash
uv sync
```

### 3. Set Up API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 4. Prepare Dataset

```bash
uv run python src/nlp_cap/data/prepare_data.py
```

This downloads ~5000 Amazon product reviews from HuggingFace for training.

### 5. Train the Baseline Model

```bash
uv run python src/nlp_cap/models/train_baseline.py
```

Trains TF-IDF + Logistic Regression and Naive Bayes, evaluates both, and saves the best model.

### 6. Run the Server

```bash
uv run uvicorn nlp_cap.main:app --reload
```

### 7. Open the App

Navigate to **http://localhost:8000** in your browser.

Alternatively, open `frontend/index.html` directly (update the API URL in `script.js` if needed).

## 📊 API Endpoint

### `POST /analyze`

**Request:**
```json
{
  "review_text": "The battery life is terrible but the camera is amazing."
}
```

**Response:**
```json
{
  "overall_sentiment": "Negative",
  "baseline_model_confidence": 0.82,
  "reasoning": "The reviewer expresses frustration with battery drain while praising camera quality...",
  "aspects": [
    { "feature": "battery", "sentiment": "Negative" },
    { "feature": "camera", "sentiment": "Positive" }
  ],
  "recommended_features": [
    "Battery health monitoring dashboard",
    "Fast-charging optimization mode"
  ]
}
```

## 📁 Project Structure

```
nlp_cap/
├── src/nlp_cap/
│   ├── main.py              # FastAPI app entrypoint
│   ├── config.py             # Environment config
│   ├── routers/
│   │   └── analyze.py        # POST /analyze endpoint
│   ├── schemas/
│   │   └── review_schema.py  # Pydantic models
│   ├── services/
│   │   ├── baseline_model.py # Trained model loader
│   │   ├── llm_chain.py      # LangChain pipeline
│   │   └── groq_client.py    # Groq API wrapper
│   ├── models/
│   │   └── train_baseline.py # Training script
│   └── data/
│       └── prepare_data.py   # Dataset preparation
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .env.example
├── pyproject.toml
└── README.md
```

## 📄 License

MIT