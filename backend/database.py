import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import CHROMA_DB_DIR, EMBEDDING_MODEL

# Shared embedding instance
_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy loads local HuggingFace embeddings model."""
    global _embeddings
    if _embeddings is None:
        print(f"[Database] Initializing embeddings: {EMBEDDING_MODEL}...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
    return _embeddings

def get_vector_store() -> Chroma:
    """Returns persistent Chroma DB connection."""
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings()
    )

def clear_vector_store():
    """Wipes Chroma database collection or directory."""
    try:
        store = get_vector_store()
        db_data = store.get()
        if db_data and db_data.get("ids"):
            store.delete(ids=db_data["ids"])
            print("[Database] All documents deleted from collection successfully.")
            return
    except Exception as e:
        print(f"[Database] Failed to delete documents via collection API: {e}. Attempting directory wipe...")

    # Fallback to directory deletion if API delete failed or was empty
    if os.path.exists(CHROMA_DB_DIR):
        try:
            # We must close any open connections if possible (handled by GC mostly, but try-catch is safe)
            shutil.rmtree(CHROMA_DB_DIR)
            print("[Database] Vector database directory cleared.")
        except Exception as rmtree_err:
            print(f"[Database] Failed to remove database directory: {rmtree_err}")
            raise RuntimeError(f"Database files are locked. Try restarting the server. Error: {rmtree_err}")
