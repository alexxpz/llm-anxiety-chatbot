import requests
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "data" / "articles"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def sanitize_filename(title):
    title = re.sub(r"[\\/*?\"<>|:]", "", title)
    title = re.sub(r"\s+", "_", title)
    return title[:100]

def search_arxiv(query, max_results=10):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    print(f"🔍 Searching arXiv for '{query}'...")
    response = requests.get(ARXIV_API_URL, params=params)
    response.raise_for_status()

    # Parse XML
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    articles = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        pdf_url = entry.find("atom:id", ns).text.strip().replace("abs", "pdf") + ".pdf"
        articles.append({"title": title, "pdf_url": pdf_url})

    return articles

def download_pdf(title, url):
    filename = sanitize_filename(title) + ".pdf"
    filepath = SAVE_DIR / filename

    if filepath.exists():
        print(f"✅ Already exists: {filename}")
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        print(f"⬇️  Downloading: {filename}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")


def run():
    query = input("🔎 Search keywords: ").strip()
    try:
        max_results = int(input("🔢 Number of articles to download: ").strip())
    except ValueError:
        max_results = 5

    articles = search_arxiv(query, max_results)

    if not articles:
        print("⚠️ No results found.")
        return

    for article in articles:
        download_pdf(article["title"], article["pdf_url"])

if __name__ == "__main__":
    run()
