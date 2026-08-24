def main():
    """Entry point for the nlp_cap package."""
    import uvicorn
    uvicorn.run("nlp_cap.main:app", host="127.0.0.1", port=8000, reload=True)
