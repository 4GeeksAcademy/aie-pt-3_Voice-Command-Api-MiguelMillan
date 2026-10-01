import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    groq_transcription_model: str
    request_timeout_seconds: float


@lru_cache
def get_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        # llama3-8b-8192 was retired by Groq; llama-3.1-8b-instant is its replacement.
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_transcription_model=os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45")),
    )
