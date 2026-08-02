import time
import re
import numpy as np
from typing import List, Dict, Any
import tiktoken
from langchain_core.documents import Document
from backend.database import get_embeddings

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Counts the number of tokens in a text string using tiktoken, falling back to character estimation on connection/SSL errors."""
    try:
        # tiktoken compiles in Rust and might try downloading encodings, which fails under SSL proxies
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"[Evaluation] tiktoken token counting failed: {e}. Using offline estimation.")
        # Fallback approximation: 1 token is roughly 4 characters in English
        return max(1, len(text) // 4)

def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using simple regex punctuation bounds."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def scale_similarity(raw_sim: float, min_val: float = 0.22, max_val: float = 0.60) -> float:
    """
    Scales raw cosine similarities from all-MiniLM-L6-v2 to a wide, intuitive [0, 1] range.
    MiniLM cosine similarities naturally cluster between 0.20 (unrelated) and 0.65 (high match).
    """
    if raw_sim <= min_val:
        return 0.0
    if raw_sim >= max_val:
        return 1.0
    return float((raw_sim - min_val) / (max_val - min_val))

def is_conversational_filler(sentence: str) -> bool:
    """Checks if a sentence is conversational filler (greetings, test prompts, system fallback refusals)."""
    words = sentence.split()
    if len(words) <= 4:
        return True
    
    # Common conversational fillers/refusals that do not require factual document support
    fillers = [
        "ready when you are",
        "testing the ai",
        "ask your question",
        "cannot find the answer",
        "not mentioned in the context",
        "please go ahead",
        "how can i help you",
        "provided document context",
        "hello",
        "welcome"
    ]
    sentence_lower = sentence.lower()
    for filler in fillers:
        if filler in sentence_lower:
            return True
    return False

def evaluate_rag(
    query: str,
    response: str,
    retrieved_docs: List[Document],
    retrieval_latency: float,
    llm_latency: float
) -> Dict[str, Any]:
    """
    Computes RAG evaluation metrics using the pre-loaded SentenceTransformers model.
    Applies calibrated min-max scaling to align raw MiniLM cosine similarity scores with human-intuitive scales.
    """
    start_eval_time = time.time()
    
    # 1. Initialize Embeddings & Vector Representations
    embeddings_model = get_embeddings()
    
    # Pre-embed query and response
    query_emb = np.array(embeddings_model.embed_query(query))
    response_emb = np.array(embeddings_model.embed_query(response))
    
    # Embed retrieved doc chunks
    doc_embs = []
    for doc in retrieved_docs:
        doc_embs.append(np.array(embeddings_model.embed_query(doc.page_content)))

    # 2. Token Counts & Cost Estimation
    prompt_tokens = count_tokens(query + "\n" + "".join([d.page_content for d in retrieved_docs]))
    completion_tokens = count_tokens(response)
    total_tokens = prompt_tokens + completion_tokens
    
    # Ollama is local, so cost is $0. We also expose a hypothetical OpenAI equivalent cost for educational display.
    estimated_cost = 0.0
    openai_equivalent_cost = (prompt_tokens * 0.0015 + completion_tokens * 0.002) / 1000  # GPT-3.5 turbo equivalent

    # 3. Latency Metrics
    total_response_time = retrieval_latency + llm_latency

    # 4. Semantic Similarity & Answer Relevancy
    raw_relevancy = calculate_cosine_similarity(query_emb, response_emb)
    # Answer Relevancy compares short queries vs long responses: scale with min=0.08, max=0.45
    answer_relevancy = scale_similarity(raw_relevancy, min_val=0.08, max_val=0.45)

    # 5. Context Precision (relevance of retrieved chunks to the query)
    precisions = []
    chunk_similarities = []
    for doc_emb in doc_embs:
        sim = calculate_cosine_similarity(query_emb, doc_emb)
        # Context Precision compares query to short document chunks: scale with min=0.08, max=0.38
        sim_norm = scale_similarity(sim, min_val=0.08, max_val=0.38)
        chunk_similarities.append(sim_norm)
        precisions.append(sim_norm)
    
    context_precision = float(np.mean(precisions)) if precisions else 0.0

    # 6. Faithfulness & Groundedness (Hallucination Detection)
    # Split response into sentences and check semantic overlap with chunks
    response_sentences = split_into_sentences(response)
    sentence_faithfulness_scores = []
    sentence_mappings = []

    for sentence in response_sentences:
        sent_emb = np.array(embeddings_model.embed_query(sentence))
        max_sim = 0.0
        best_chunk_idx = -1
        
        # Calculate raw cosine similarities against chunks
        for idx, doc_emb in enumerate(doc_embs):
            sim = calculate_cosine_similarity(sent_emb, doc_emb)
            if sim > max_sim:
                max_sim = sim
                best_chunk_idx = idx
        
        # Scale the best match similarity to context chunks: scale with min=0.22, max=0.60
        scaled_sentence_score = scale_similarity(max_sim, min_val=0.22, max_val=0.60)
        
        # If it's a conversational sentence/filler, it does not need grounding. We set it to 1.0 to avoid false positives.
        if is_conversational_filler(sentence):
            scaled_sentence_score = 1.0
            
        sentence_faithfulness_scores.append(scaled_sentence_score)
        sentence_mappings.append({
            "sentence": sentence,
            "max_supported_score": scaled_sentence_score,
            "source_chunk_idx": best_chunk_idx
        })

    faithfulness = float(np.mean(sentence_faithfulness_scores)) if sentence_faithfulness_scores else 1.0
    hallucination_score = 1.0 - faithfulness

    # 7. Unsupervised Information Retrieval Heuristics
    # We define a relevance threshold of 0.45 to simulate relevant docs
    rel_threshold = 0.45
    relevant_retrieved = [sim >= rel_threshold for sim in chunk_similarities]
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, is_rel in enumerate(relevant_retrieved):
        if is_rel:
            mrr = 1.0 / (idx + 1)
            break
            
    # Precision @ K & Recall @ K (simulated context recall)
    k = len(retrieved_docs)
    prec_at_k = sum(relevant_retrieved) / k if k > 0 else 0.0
    # Estimate total relevance set size (at least sum of relevant retrieved)
    total_estimated_relevant = max(1, sum(relevant_retrieved))
    recall_at_k = sum(relevant_retrieved) / total_estimated_relevant if total_estimated_relevant > 0 else 0.0
    
    # NDCG
    dcg = 0.0
    idcg = 0.0
    # Ideal DCG (sort relevance score descending)
    sorted_similarities = sorted(chunk_similarities, reverse=True)
    for idx, sim in enumerate(chunk_similarities):
        rel = 1.0 if sim >= rel_threshold else 0.0
        dcg += rel / np.log2(idx + 2)
        
        ideal_rel = 1.0 if sorted_similarities[idx] >= rel_threshold else 0.0
        idcg += ideal_rel / np.log2(idx + 2)
        
    ndcg = (dcg / idcg) if idcg > 0 else (1.0 if dcg == 0 else 0.0)

    # 8. Retrieved Chunks Analysis & LLM Usage Heuristic
    # A chunk is marked as "used" by the LLM if any generated sentence aligns strongly with it (similarity > 0.58)
    chunks_details = []
    chunk_used_counts = [0] * len(retrieved_docs)
    
    for mapping in sentence_mappings:
        if mapping["max_supported_score"] > 0.58 and mapping["source_chunk_idx"] != -1:
            chunk_used_counts[mapping["source_chunk_idx"]] += 1
            
    for idx, doc in enumerate(retrieved_docs):
        used = chunk_used_counts[idx] > 0
        chunks_details.append({
            "chunk_id": f"chunk_{idx + 1}",
            "source_document": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "text": doc.page_content,
            "length": len(doc.page_content),
            "similarity_score": round(chunk_similarities[idx], 4),
            "rank": idx + 1,
            "metadata": doc.metadata,
            "used_by_llm": used,
            "explanation": (
                f"This chunk was ranked #{idx + 1} during semantic retrieval based on a similarity of "
                f"{round(chunk_similarities[idx] * 100, 1)}%. The LLM "
                f"{'directly referenced or synthesized content' if used else 'largely ignored the content'} "
                f"from this chunk while generating the final response."
            )
        })

    # Context Utilization
    chunks_used_count = sum(1 for c in chunks_details if c["used_by_llm"])
    context_utilization = chunks_used_count / len(retrieved_docs) if retrieved_docs else 0.0

    eval_duration = time.time() - start_eval_time

    return {
        "metrics": {
            "answer_relevancy": round(answer_relevancy, 4),
            "context_precision": round(context_precision, 4),
            "context_recall": round(recall_at_k, 4),
            "faithfulness": round(faithfulness, 4),
            "hallucination_score": round(hallucination_score, 4),
            "context_utilization": round(context_utilization, 4),
            "semantic_similarity": round(answer_relevancy, 4),  # Query to Answer match
            "precision_at_k": round(prec_at_k, 4),
            "recall_at_k": round(recall_at_k, 4),
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
            "retrieval_latency": round(retrieval_latency, 4),
            "llm_response_time": round(llm_latency, 4),
            "total_response_time": round(total_response_time, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": round(estimated_cost, 4),
            "openai_equivalent_cost": round(openai_equivalent_cost, 6),
            "evaluation_duration": round(eval_duration, 4)
        },
        "chunks": chunks_details,
        "sentence_mappings": sentence_mappings
    }
