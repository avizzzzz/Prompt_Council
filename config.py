"""Configuration module — reads from .env with sensible defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── API Keys ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ── Model endpoints ───────────────────────────────────────────────────
GEMINI_ENDPOINT = os.getenv(
    "GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
)

SILICONFLOW_ENDPOINT = os.getenv(
    "SILICONFLOW_ENDPOINT",
    "https://api.siliconflow.cn/v1/chat/completions",
)

DEEPSEEK_ENDPOINT = os.getenv(
    "DEEPSEEK_ENDPOINT",
    "https://api.deepseek.com/v1/chat/completions",
)

# ── Retry / backoff ───────────────────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
BASE_DELAY = float(os.getenv("BASE_DELAY", "2.0"))      # seconds
MAX_DELAY = float(os.getenv("MAX_DELAY", "60.0"))       # seconds
BACKOFF_MULTIPLIER = float(os.getenv("BACKOFF_MULTIPLIER", "2.0"))

# ── Debate loop ───────────────────────────────────────────────────────
MAX_DEBATE_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))
INTER_CALL_DELAY = float(os.getenv("INTER_CALL_DELAY", "1.5"))  # seconds

# ── Output ────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "final_results.md"
