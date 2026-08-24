"""
train_baseline.py
Trains TF-IDF + Logistic Regression and Multinomial Naive Bayes models
on the prepared Amazon Reviews dataset. Performs hyperparameter tuning
using GridSearchCV, evaluates both, and saves the best performer.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Paths
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(MODELS_DIR), "data")
DATASET_PATH = os.path.join(DATA_DIR, "reviews_dataset.csv")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")


def load_data():
    """Load and validate the reviews dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Run 'python src/nlp_cap/data/prepare_data.py' first."
        )

    df = pd.read_csv(DATASET_PATH)
    print(f"[i] Loaded {len(df):,} reviews")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}\n")

    # Drop any rows with missing text
    df = df.dropna(subset=["review_text", "sentiment"])
    return df


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a model and print metrics."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*60}")
    print(f"[+] {name} — Test Set Results")
    print(f"{'='*60}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    print(f"    Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    {cm}")

    return acc, f1


def tune_logistic_regression(X_train, y_train):
    """Hyperparameter tuning for TF-IDF + Logistic Regression using GridSearchCV."""
    print("\n" + "=" * 60)
    print("[*] Tuning Logistic Regression (GridSearchCV, 5-fold CV)...")
    print("=" * 60)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True)),
        ("clf", LogisticRegression(random_state=42)),
    ])

    param_grid = {
        # TF-IDF hyperparameters
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 3],
        "tfidf__max_df": [0.90, 0.95],
        # Logistic Regression hyperparameters
        "clf__C": [0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "liblinear"],
        "clf__max_iter": [1000],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,  # Use all CPU cores
        verbose=1,
        return_train_score=True,
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n[+] LR Grid Search completed in {elapsed:.1f}s")
    print(f"    Combinations tested: {len(grid_search.cv_results_['mean_test_score'])}")
    print(f"    Best CV F1 Score:    {grid_search.best_score_:.4f}")
    print(f"    Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_naive_bayes(X_train, y_train):
    """Hyperparameter tuning for TF-IDF + Multinomial Naive Bayes using GridSearchCV."""
    print("\n" + "=" * 60)
    print("[*] Tuning Multinomial Naive Bayes (GridSearchCV, 5-fold CV)...")
    print("=" * 60)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True)),
        ("clf", MultinomialNB()),
    ])

    param_grid = {
        # TF-IDF hyperparameters
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 3],
        "tfidf__max_df": [0.90, 0.95],
        # Naive Bayes hyperparameters
        "clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n[+] NB Grid Search completed in {elapsed:.1f}s")
    print(f"    Combinations tested: {len(grid_search.cv_results_['mean_test_score'])}")
    print(f"    Best CV F1 Score:    {grid_search.best_score_:.4f}")
    print(f"    Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train():
    """Train, tune, evaluate, and save the best baseline model."""
    # Load data
    df = load_data()
    X = df["review_text"].tolist()
    y = df["sentiment"].tolist()

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[i] Split: {len(X_train):,} train / {len(X_test):,} test")

    # --- Hyperparameter Tuning ---
    lr_pipeline, lr_params, lr_cv_f1 = tune_logistic_regression(X_train, y_train)
    nb_pipeline, nb_params, nb_cv_f1 = tune_naive_bayes(X_train, y_train)

    # --- Final Evaluation on held-out test set ---
    print("\n\n" + "#" * 60)
    print("  FINAL EVALUATION ON HELD-OUT TEST SET")
    print("#" * 60)

    lr_acc, lr_f1 = evaluate_model("Logistic Regression (Tuned)", lr_pipeline, X_test, y_test)
    nb_acc, nb_f1 = evaluate_model("Multinomial Naive Bayes (Tuned)", nb_pipeline, X_test, y_test)

    # --- Select and save best model ---
    if lr_f1 >= nb_f1:
        best_pipeline = lr_pipeline
        best_name = "Logistic Regression"
        best_params = lr_params
    else:
        best_pipeline = nb_pipeline
        best_name = "Multinomial Naive Bayes"
        best_params = nb_params

    print(f"\n{'='*60}")
    print(f"[+] WINNER: {best_name} (Test F1={max(lr_f1, nb_f1):.4f})")
    print(f"{'='*60}")
    print(f"    Best hyperparameters:")
    for param, value in best_params.items():
        print(f"      {param}: {value}")

    # Extract vectorizer and classifier from the winning pipeline
    best_vectorizer = best_pipeline.named_steps["tfidf"]
    best_model = best_pipeline.named_steps["clf"]

    # Save vectorizer and model separately (for loading in baseline_model.py)
    joblib.dump(best_vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[+] Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"[+] Saved model     -> {MODEL_PATH}")
    print("\n[+] Training with hyperparameter tuning complete!")


if __name__ == "__main__":
    train()

