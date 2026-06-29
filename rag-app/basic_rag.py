from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_config import model, retriever, answer_prompt

# 基本的なRAG実装
basic_rag_chain = {
    "question": RunnablePassthrough(),
    "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))
} | answer_prompt | model | StrOutputParser()

if __name__ == "__main__":
    question = "In Shadow RAG Auditing Data Provenance (S-RAG) framework, what kind of hardware configure was used."
    result = basic_rag_chain.invoke(question)
    print("=== Basic RAG ===")
    print(result)
