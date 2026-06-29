from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith設定
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG"

# 既存のChromaデータベースを読み込み
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# HyDE (Hypothetical Document Embeddings) RAG
hypothetical_prompt = ChatPromptTemplate.from_template('''
次の質問に対する仮説的なドキュメント（回答例）を一文で書いてください

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

retriever = db.as_retriever(search_kwargs={"k": 5})

# 検索用プロンプト（実際の回答生成）
answer_prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

hyde_rag_chain = {
    "question": RunnablePassthrough(),
    "context": hypothetical_chain | retriever,
} | answer_prompt | model | StrOutputParser()

result = hyde_rag_chain.invoke("LangChainの概要を教えてください")
print(result)
