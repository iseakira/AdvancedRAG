from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

import os
from tqdm import tqdm
load_dotenv()

def file_filter(file_path:str) -> bool:
  return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

print("Loading documents...")
documents = loader.load()
print(f"Loaded: {len(documents)} documents")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# バッチに分けて進捗表示
batch_size = 50
batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]

db = None
for batch in tqdm(batches, desc="Embedding", unit="batch"):
    if db is None:
        db = Chroma.from_documents(batch, embeddings)
    else:
        db.add_documents(batch)

print("Done!")