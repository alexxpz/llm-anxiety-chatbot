import os
import pickle
from pathlib import Path

def chunk_texts_by_paragraph(text_dir: Path, output_path: Path, min_chars: int = 200):
    print("🚀 Starting paragraph chunking...")

    if not text_dir.exists():
        print(f"❌ Text directory not found: {text_dir}")
        return

    all_chunks = []
    txt_files = list(text_dir.glob("*.txt"))

    if not txt_files:
        print(f"⚠️ No .txt files found in: {text_dir}")
        return

    for txt_file in txt_files:
        print(f"📄 Processing file: {txt_file.name}")
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"❌ Failed to read {txt_file.name}: {e}")
            continue

        # Split by double line breaks (paragraphs)
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) >= min_chars]

        for para in paragraphs:
            all_chunks.append({
                "source": txt_file.name,
                "text": para
            })

    print(f"\n✅ Extracted {len(all_chunks)} paragraph chunks from {len(txt_files)} files.")

    # Save the result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"💾 Chunks saved to: {output_path}\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    TEXT_DIR = BASE_DIR / "data" / "texts"
    OUTPUT_PATH = BASE_DIR / "data" / "faiss" / "chunks.pkl"

    chunk_texts_by_paragraph(text_dir=TEXT_DIR, output_path=OUTPUT_PATH)
