import os
import shutil
import time
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Setup environment, configurations, SSL bypass, and httpx patches
from backend.config import UPLOAD_DIR, FRONTEND_DIR, OLLAMA_TEXT_MODEL
from backend.document_processor import DocumentProcessor
from backend.database import get_vector_store
from backend.llm import get_chat_model
from backend.evaluation import evaluate_rag

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Main document processor instance
doc_processor = DocumentProcessor()

# In-memory evaluations store (persisted for the conversation session)
evaluations_db = {}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message_id: Optional[str] = None
    answer: str
    sources: List[dict]
    evaluation: Optional[dict] = None

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    setup_mode: str = Form("automatic"),
    splitter_type: str = Form("RecursiveCharacterTextSplitter"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
):
    """Upload and process a document file with dynamic splitter parameters."""
    # Validation
    if setup_mode.lower().strip() == "manual":
        if chunk_size < 100 or chunk_size > 10000:
            raise HTTPException(status_code=400, detail="Chunk Size must be between 100 and 10000.")
        if chunk_overlap < 0 or chunk_overlap > 1000:
            raise HTTPException(status_code=400, detail="Chunk Overlap must be between 0 and 1000.")
        if chunk_overlap >= chunk_size:
            raise HTTPException(status_code=400, detail="Chunk Overlap cannot be greater than or equal to Chunk Size.")

    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc_processor.process_file(
            temp_file_path,
            file.filename,
            setup_mode=setup_mode,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return {
            "status": "success",
            "filename": file.filename,
            "message": "File processed and added to knowledge base.",
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        print(f"Error during upload/processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question based on uploaded documents."""
    question = request.message

    try:
        vector_store = get_vector_store()

        # Check for documents
        try:
            db_data = vector_store.get()
            if not db_data or not db_data.get("documents"):
                return ChatResponse(
                    answer="No documents have been uploaded yet. Please upload files first.",
                    sources=[],
                )
        except Exception:
            return ChatResponse(
                answer="No documents have been uploaded yet. Please upload files first.",
                sources=[],
            )

        # 1. Similarity search (Measure Retrieval Latency)
        t_start_retrieval = time.time()
        docs = vector_store.similarity_search(question, k=4)
        retrieval_latency = time.time() - t_start_retrieval
        
        if not docs:
            return ChatResponse(
                answer="I could not find relevant context in the uploaded documents.",
                sources=[],
            )

        # Build context
        context_parts = []
        sources = []
        for i, doc in enumerate(docs):
            src_name = doc.metadata.get("source", "Unknown File")
            page_num = doc.metadata.get("page")
            location = f"{src_name} (Page {page_num})" if page_num else src_name

            context_parts.append(f"--- Doc {i+1} [{location}] ---\n{doc.page_content}")
            sources.append({
                "source": src_name,
                "page": page_num,
                "snippet": doc.page_content[:150] + "...",
            })

        context = "\n\n".join(context_parts)

        # LLM call (Measure LLM Response Time)
        from langchain_core.prompts import ChatPromptTemplate

        llm = get_chat_model()

        system_prompt = (
            "You are a helpful, premium AI chatbot that answers questions based on "
            "the provided document context.\n"
            "Analyze the document snippets carefully. Answer the user's question "
            "accurately using ONLY the provided context.\n"
            "If the answer is not in the context, state clearly that you cannot find "
            "the answer in the uploaded files. Do not fabricate answers.\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        chain = prompt | llm
        
        t_start_llm = time.time()
        response = chain.invoke({"context": context, "question": question})
        llm_latency = time.time() - t_start_llm

        # Compute RAG Evaluation Metrics on the fly
        eval_result = evaluate_rag(
            query=question,
            response=response.content,
            retrieved_docs=docs,
            retrieval_latency=retrieval_latency,
            llm_latency=llm_latency
        )
        
        # Save evaluation result to evaluations database
        message_id = str(uuid.uuid4())
        evaluations_db[message_id] = eval_result

        return ChatResponse(
            message_id=message_id,
            answer=response.content,
            sources=sources,
            evaluation=eval_result
        )

    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

@app.get("/api/evaluation/{message_id}")
async def get_message_evaluation(message_id: str):
    """Retrieve RAG evaluation metrics for a specific message."""
    if message_id in evaluations_db:
        return evaluations_db[message_id]
    raise HTTPException(status_code=404, detail="Evaluation metrics not found for this message.")


@app.get("/api/config")
async def get_config():
    """Retrieve active LLM configurations."""
    return {"provider": "ollama", "model": OLLAMA_TEXT_MODEL}

@app.get("/api/documents")
async def list_documents():
    """Return unique filenames stored in the knowledge base."""
    try:
        vector_store = get_vector_store()
        db_data = vector_store.get()
        if not db_data or not db_data.get("metadatas"):
            return {"documents": []}

        sources = set()
        for meta in db_data["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])

        return {"documents": sorted(sources)}
    except Exception as e:
        print(f"Error listing documents: {e}")
        return {"documents": []}

@app.delete("/api/clear")
async def clear_database():
    """Clear the entire knowledge base."""
    try:
        doc_processor.clear_database()
        return {"status": "success", "message": "Knowledge base cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear: {str(e)}")

# Serve frontend static assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {FRONTEND_DIR}")
