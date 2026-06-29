from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith設定
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG"

# モデル
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Chromaデータベース
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# 回答生成プロンプト
answer_prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')
