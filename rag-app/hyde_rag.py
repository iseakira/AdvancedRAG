from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_config import model, retriever, answer_prompt

# HyDE (Hypothetical Document Embeddings) プロンプト
hypothetical_prompt = ChatPromptTemplate.from_template('''
次の質問に対する仮説的なドキュメント（回答例）を一文で書いてください

質問: {question}
''')

# HyDE チェーン
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

# HyDE RAG実装
hyde_rag_chain = {
    "question": RunnablePassthrough(),
    "context": hypothetical_chain | retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))
} | answer_prompt | model | StrOutputParser()

if __name__ == "__main__":
    question = "In Shadow RAG Auditing Data Provenance (S-RAG) framework, what kind of hardware configure was used."
    result = hyde_rag_chain.invoke(question)
    print("=== HyDE RAG ===")
    print(result)
