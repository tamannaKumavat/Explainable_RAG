import faiss
import numpy as np
import openai
from typing import List, Tuple, Optional

from constants import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY

# Store for documents and metadata
documents: List[str] = []
metadata: List[dict] = []

# FAISS vector index (dimension must match embedding size)
index: Optional[faiss.IndexFlatL2] = None


def get_embedding(text: str) -> List[float]:
    response = openai.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def index_chunks(chunks: List[str], metadatas: Optional[List[dict]] = None) -> None:
    global index, documents, metadata

    vectors = [get_embedding(chunk) for chunk in chunks]

    documents.extend(chunks)
    if metadatas:
        metadata.extend(metadatas)
    else:
        metadata.extend([{} for _ in chunks])

    vecs_np = np.array(vectors).astype("float32")

    dim = vecs_np.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dim)
    index.add(vecs_np)


def retrieve_similar_chunks(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    if index is None or index.ntotal == 0:
        raise ValueError("FAISS index is empty. Call index_chunks() first.")

    query_vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for i in indices[0]:
        if i < len(documents):
            results.append((documents[i], metadata[i]))
    return results
