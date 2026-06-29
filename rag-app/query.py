from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# 既存のChromaデータベースを読み込み
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)




hypothetical_prompt = ChatPromptTemplate.from_template(
  '''
次の質問に回答する一文を書いてください

質問:{question}
'''
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
hypothetical_chain = hypothetical_prompt | model | StrOutputParser


retriever = db.as_retriever()

hyde_rag_chain = {
    "question": RunnablePassthrough(),
    "context": hypothetical_chain | retriever,
} | hypothetical_prompt| model | StrOutputParser()

result = hyde_rag_chain.invoke("論文について教えて")
print(result)
