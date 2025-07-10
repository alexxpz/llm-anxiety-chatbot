# 🧠 Anxiety Chatbot with Local LLM

This project is a chatbot designed to answer questions related to anxiety using a local Large Language Model (LLM) and semantic search with FAISS.

---

## 📦 Features

- **Local LLM Inference**: Uses a lightweight LLaMA-based model for answering questions.
- **Semantic Search**: FAISS + SentenceTransformers for fast, relevant context retrieval.
- **Chunking & Vectorization**: Input texts are chunked, embedded, and stored in FAISS.
- **Instruction-tuned Embeddings**: Based on `sentence-transformers/all-MiniLM-L6-v2`.
- **No Internet Needed** after setup.

---

## 🧰 Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/llm-anxiety-chatbot.git
cd llm-anxiety-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare your data

- Place your `.txt` or `.pdf` files in the `data/raw/` folder.
- Then run:

```bash
python scripts/prepare_chunks.py
python scripts/vectorize_with_sentence_transformer.py
```

---

## 🤖 Running the Bot

### Option A - Ask a question via terminal

```bash
python scripts/answer_with_llama.py
```

> You'll be prompted to type a question like: `What are the symptoms of anxiety?`

### Option B - Semantic search only

```bash
python scripts/search_with_faiss.py
```

---

## 💡 Notes

- Make sure your model fits your hardware. 7B models usually need at least 8GB VRAM or 16GB RAM with quantization (gguf).
- LLaMA.cpp runs entirely offline after the model is downloaded.

---

## 🔐 License

MIT License.