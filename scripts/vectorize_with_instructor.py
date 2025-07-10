from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "faiss" / "texts.pkl"
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss" / "instructor_index.index"

# === Load model ===
model = SentenceTransformer("hkunlp/instructor-base")

# === Load chunks ===
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"📚 Loaded {len(chunks)} chunks.")
print("⚙️  Encoding all chunks...")

# === Prepare inputs: concat instruction + text ===
instruction = "Represent the document for retrieval:"
texts_with_instruction = [f"{instruction} {chunk['text']}" for chunk in chunks]

# === Generate embeddings ===
embeddings = model.encode(texts_with_instruction, batch_size=16, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# === Create and save FAISS index ===
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, str(FAISS_INDEX_PATH))

print(f"✅ FAISS index saved at {FAISS_INDEX_PATH}")
