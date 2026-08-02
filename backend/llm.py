from langchain_ollama import ChatOllama
from backend.config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL

def get_chat_model() -> ChatOllama:
    """Returns the Ollama chat LLM instance."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_TEXT_MODEL,
        temperature=0.2
    )

def get_vision_model() -> ChatOllama:
    """Returns the Ollama vision LLM instance for OCR tasks."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_VISION_MODEL,
        temperature=0.0
    )
