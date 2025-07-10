import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "data" / "faiss"
N_RESULTS = 5

# Load model and index
print("📥 Loading model and FAISS index...")
model = SentenceTransformer("hkunlp/instructor-base")
index = faiss.read_index(str(FAISS_DIR / "instructor_index.index"))

with open(FAISS_DIR / "texts.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Loaded {len(chunks)} chunks.")

# User query
query = input("🔎 Ask a question about anxiety: ").strip()
instruction = "Represent the scientific question for retrieving relevant passages:"
query_text = f"{instruction} {query}"
query_embedding = model.encode([query_text])
query_embedding = np.array(query_embedding).astype("float32")

# Search
D, I = index.search(query_embedding, k=N_RESULTS)

# Display results
print("\n📚 Top matches:\n")

for rank, (i, dist) in enumerate(zip(I[0], D[0]), 1):
    chunk = chunks[i]
    print(f"#{rank} (Distance: {dist:.4f}) - Source: {chunk['source']}")
    print("-" * 80)
    print(chunk["text"][:1000])  # Display first 1000 characters
    print("-" * 80 + "\n")
