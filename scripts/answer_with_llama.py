from llama_cpp import Llama
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "data/faiss/instructor_index.index"
CHUNKS_PATH = BASE_DIR / "data/faiss/texts.pkl"

# === Load FAISS index and text chunks ===
print("📥 Loading FAISS index and text chunks...")
index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# === Load local LLM ===
print("🧠 Loading local LLM...")
llm = Llama(
    model_path="models/mistral-7b-instruct/mistral.Q4_K_M.gguf",
    n_ctx=8192,
    n_threads=8,          # adapte selon ton CPU
    n_gpu_layers=20       # 20 couches sur 8 Go VRAM = safe
)

# === Ask user a question ===
query = input("🔎 Ask a question about anxiety:\n> ").strip()

# === Load SentenceTransformer model ===
print("🧩 Loading embedding model...")
embedding_model = SentenceTransformer("hkunlp/instructor-base")

# === Embed query with instruction simulated ===
instruction = "Represent the scientific question for retrieving relevant passages:"
query_input = f"{instruction} {query}"
query_embedding = embedding_model.encode([query_input])
query_embedding = np.array(query_embedding).astype("float32")

# === Search FAISS ===
k = 10
D, I = index.search(query_embedding, k)

# === Filter chunks by distance threshold ===
distance_threshold = 0.5
print(f"🔍 Found {len(I[0])} results. Filtering by distance < {distance_threshold}...")
filtered_chunks = [
    chunks[i]["text"] if isinstance(chunks[i], dict) else chunks[i]
    for i, d in zip(I[0], D[0])
    if d < distance_threshold
]

# === If no relevant chunks found ===
if not filtered_chunks:
    print("⚠️ No relevant content found in your corpus to answer this question.")
    exit()

# === Truncate context if too long ===
MAX_TOKENS = 1800
# context = "\n\n".join(filtered_chunks)
context = "\n\n".join(str(chunk)[:500] for chunk in filtered_chunks)
if len(context.split()) > MAX_TOKENS:
    context = " ".join(context.split()[:MAX_TOKENS])

# === Build prompt ===
prompt = f"""You are a helpful research assistant specialized in anxiety research.

Context:
{context}

Question:
{query}

Answer:"""

# === Run inference ===
print("\n🤖 Generating answer...\n")
response = llm(prompt, max_tokens=300, temperature=0.7, stop=["Context:", "Question:"])
print("📝 Answer:\n" + response["choices"][0]["text"].strip())
