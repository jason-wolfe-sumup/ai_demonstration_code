from openai import OpenAI

client = OpenAI(api_key="<API-KEY>")
from pinecone import Pinecone, ServerlessSpec
import cohere
import numpy as np
import nltk
import faiss
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.retrievers.bm25 import BM25Retriever
from functools import lru_cache
from nltk.tokenize import sent_tokenize




# Initialize APIs
co = cohere.Client("<API-KEY>") # Trial Key

# Initialize Pinecone
pinecone = Pinecone(api_key="<API-KEY>", environment="us-east-1")
index_name = "rag-hybrid-index1"


if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine",spec=ServerlessSpec(cloud="aws", region="us-east-1"))  # Using SBERT embeddings
index = pinecone.Index(index_name)

# Load SBERT model for fine-tuned embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Tokenizer for chunking
nltk.download("punkt")
tokenizer = tiktoken.get_encoding("cl100k_base")

# FastAPI app
app = FastAPI()

# Sample long documents
documents = [
    "The Eiffel Tower is located in Paris, France. It was completed in 1889 and is one of the most recognizable structures in the world. It was originally criticized but is now a global icon.",
    "The Great Wall of China is one of the Seven Wonders of the World. It stretches over 13,000 miles and was built to protect against invasions. Construction started as early as the 7th century BC.",
    "Python is a popular programming language for AI and data science. It is widely used for machine learning, web development, and automation.",
    "Machine learning allows computers to learn from data without explicit programming. Deep learning, a subset of ML, powers technologies like image recognition and self-driving cars.",
    "Tesla is an American company known for electric cars and renewable energy solutions. Founded by Elon Musk and others, it has revolutionized the EV industry with cars like the Model S and Model 3.",
]

# Caching SBERT embeddings
@lru_cache(maxsize=1000)
def get_embedding(text):
    return tuple(embedding_model.encode(text, normalize_embeddings=True))

# Improved chunking with overlap
def chunk_text(text, max_tokens=256, overlap=20):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    token_count = 0

    for sentence in sentences:
        tokens = len(tokenizer.encode(sentence))
        if token_count + tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            token_count = sum(len(tokenizer.encode(s)) for s in current_chunk)

        current_chunk.append(sentence)
        token_count += tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Indexing into Pinecone & BM25
doc_texts = []
# bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)

def index_documents():
    for i, doc in enumerate(documents):
        chunks = chunk_text(doc)
        for j, chunk in enumerate(chunks):
            embedding = list(get_embedding(chunk))
            metadata = {"text": chunk}
            index.upsert([(f"doc-{i}-chunk-{j}", embedding, metadata)])
            doc_texts.append(chunk)

    # Add documents to BM25
    global bm25_retriever
    nodes = [SimpleNodeParser().create_node(chunk) for chunk in doc_texts]
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)

index_documents()

# Hybrid search: BM25 + Pinecone
def hybrid_search(query, k=5):
    # BM25 retrieval
    bm25_results = bm25_retriever.retrieve(query)
    bm25_docs = [(res.text, res.score) for res in bm25_results[:k]]

    # Pinecone (semantic) retrieval
    query_embedding = list(get_embedding(query))
    pinecone_results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    pinecone_docs = [(res["metadata"]["text"], res["score"]) for res in pinecone_results["matches"]]

    # Combine results
    combined_results = bm25_docs + pinecone_docs
    combined_results.sort(key=lambda x: x[1], reverse=True)  # Sort by relevance
    return combined_results[:k]

# Rerank results using Cohere
def rerank(query, retrieved_docs):
    rerank_input = [{"text": doc[0]} for doc in retrieved_docs]
    reranked = co.rerank(query=query, documents=rerank_input, top_n=2, model="rerank-english-v2.0")
    return [retrieved_docs[r["index"]] for r in reranked.results]

# Generate response with GPT-4
def generate_response(query):
    retrieved_docs = hybrid_search(query, k=5)
    reranked_docs = rerank(query, retrieved_docs)

    context = " ".join([doc[0] for doc in reranked_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": "You are a helpful AI."},
              {"role": "user", "content": prompt}])
    return response.choices[0].message.content

# API Models
class QueryRequest(BaseModel):
    query: str

# API Endpoints
@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    response = generate_response(request.query)
    return {"query": request.query, "response": response}

@app.get("/")
def health_check():
    return {"status": "RAG API is running!"}

# Run API using `uvicorn main:app --reload`
