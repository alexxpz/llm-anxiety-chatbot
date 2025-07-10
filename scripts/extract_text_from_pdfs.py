import fitz  # PyMuPDF
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "articles"
TEXT_DIR = BASE_DIR / "data" / "texts"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join([page.get_text() for page in doc])

def sanitize_filename(pdf_name):
    return pdf_name.replace(".pdf", ".txt")

def run():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    for pdf_file in pdf_files:
        txt_name = sanitize_filename(pdf_file.name)
        txt_path = TEXT_DIR / txt_name

        if txt_path.exists():
            print(f"✅ Already extracted: {txt_name}")
            continue

        print(f"🧠 Extracting: {pdf_file.name}")
        try:
            text = extract_text_from_pdf(pdf_file)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"❌ Error extracting {pdf_file.name}: {e}")

if __name__ == "__main__":
    run()
