import faiss
import numpy as np
from openai import OpenAI
import PyPDF2
import os

client = OpenAI(api_key="<API-KEY>")
from tiktoken import get_encoding

# OpenAI API key (Replace with your own)

# Dummy documents
documents = [
   #  "The Jason Tower is located in Jason\'s office.",
    "The weather in Bellingham Washington is very rainy.",
    "The Great Wall of China is one of the Seven Wonders of the World.",
]


# Encode text into vector embeddings (simple token count encoding for example)
def embed_text(text):
    enc = get_encoding("cl100k_base")
    return np.array(enc.encode(text), dtype=np.float32)

# Create FAISS index
embedding_dim = 512  # Arbitrary high number for demo
index = faiss.IndexFlatL2(embedding_dim)

# Indexing: Convert documents to fixed-size embeddings
document_embeddings = np.zeros((len(documents), embedding_dim), dtype=np.float32)
for i, doc in enumerate(documents):
    emb = embed_text(doc)
    document_embeddings[i, : len(emb)] = emb  # Simple padding
index.add(document_embeddings)

# Retrieval: Given a query, find the closest document
def retrieve(query, k=1):
    query_embedding = np.zeros((1, embedding_dim), dtype=np.float32)
    emb = embed_text(query)
    query_embedding[0, : len(emb)] = emb
    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Generation: Augment query with retrieved context
def generate_response(query):
    context = retrieve(query, k=3)[0]  # Get top-1 relevant document
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": "You are a helpful AI."},
              {"role": "user", "content": prompt}])
    return response.choices[0].message.content

# Example Query
query = "Where is the Jason Tower?"
print(generate_response(query))
