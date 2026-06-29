from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

import json
import os
from tqdm import tqdm

load_dotenv()

def load_jsonl_documents(file_path: str) -> list[Document]:
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # すべてのフィールドをembedding対象に
                text = f"Title: {data['title']}\n\nAuthors: {', '.join(data['authors'])}\n\nVenue: {data['venue']}\n\nYear: {data['year']}\n\nTrack: {data.get('track')}\n\nAward: {data.get('award')}\n\nAbstract: {data['abstract']}"

                # メタデータも保持（Noneは除外）
                metadata = {k: v for k, v in {
                    "paper_id": data["paper_id"],
                    "authors": ", ".join(data["authors"]),
                    "venue": data["venue"],
                    "year": str(data["year"]),
                    "track": data.get("track"),
                    "award": data.get("award"),
                    "source_url": data["source_url"],
                    "pdf_url": data.get("pdf_url"),
                    "arxiv_id": data.get("arxiv_id"),
                    "doi": data.get("doi"),
                    "openreview_id": data.get("openreview_id"),
                    "anthology_id": data.get("anthology_id")
                }.items() if v is not None}

                documents.append(Document(page_content=text, metadata=metadata))
    return documents

print("Loading documents from JSONL...")
documents = load_jsonl_documents("./data/paper_metadata.jsonl")
print(f"Loaded: {len(documents)} documents")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# バッチに分けて進捗表示
batch_size = 50
batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]

db = None
for batch in tqdm(batches, desc="Embedding", unit="batch"):
    if db is None:
        db = Chroma.from_documents(
            batch,
            embeddings,
            persist_directory="./chroma_db"
        )
    else:
        db.add_documents(batch)

print("Done!")
