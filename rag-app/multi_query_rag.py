from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from rag_config import model, retriever, answer_prompt

# クエリ生成用の出力スキーマ
class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")

# クエリ生成プロンプト
query_generation_prompt = ChatPromptTemplate.from_template('''
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
''')

# クエリ生成チェーン
query_generation_chain = (
    query_generation_prompt | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

# 複数の検索結果を統合
def combine_docs(doc_lists):
    """複数の検索結果を統合"""
    all_docs = []
    for docs in doc_lists:
        all_docs.extend(docs)
    # 重複を削除
    unique_content = []
    seen = set()
    for doc in all_docs:
        if doc.page_content not in seen:
            unique_content.append(doc.page_content)
            seen.add(doc.page_content)
    return "\n\n".join(unique_content)

# 複数クエリRAG実装
multi_query_rag_chain = {
    "question": RunnablePassthrough(),
    "context": query_generation_chain | retriever.map() | combine_docs
} | answer_prompt | model | StrOutputParser()

if __name__ == "__main__":
    question = "In Shadow RAG Auditing Data Provenance (S-RAG) framework, what kind of hardware configure was used."
    result = multi_query_rag_chain.invoke(question)
    print("=== Multi-Query RAG ===")
    print(result)
