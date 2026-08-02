import os
import ssl
from dotenv import load_dotenv

# Load .env variables first
load_dotenv()

# ============================================================================
# SSL BYPASS — MUST run before any library that uses httpx / requests
# ============================================================================
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Monkey-patch httpx BEFORE it gets used by huggingface_hub
import httpx
_original_client_init = httpx.Client.__init__
_original_async_client_init = httpx.AsyncClient.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    _original_client_init(self, *args, **kwargs)

def _patched_async_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    _original_async_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init
httpx.AsyncClient.__init__ = _patched_async_client_init

# ============================================================================
# Configuration Parameters
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3.2").strip()
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision").strip()

# Embeddings Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
