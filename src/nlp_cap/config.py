"""
config.py
Loads environment variables from .env file.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
_env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env"
)
load_dotenv(_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print("[!] Warning: GROQ_API_KEY not found in .env file.")
    print(f"   Expected .env at: {_env_path}")
    print("   Copy .env.example to .env and add your Groq API key.")
