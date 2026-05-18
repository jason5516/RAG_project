"""
FastAPI RAG Chat API
使用 MiniMax LLM + Hybrid Search (BM25 + Vector)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi

# ========================
# 全域變數（系統啟動時初始化）
# ========================
embadding = None
db = None
bm25 = None
texts = None
llm = None

def initialize():
    """系統啟動時載入模型與資料庫"""
    global embadding, db, bm25, texts, llm

    print("🚀 初始化系統中...")

    # 1. Embedding 模型
    embadding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"}
    )
    print("✅ Embedding 模型載入完成")

    # 2. ChromaDB 向量資料庫
    db = Chroma(persist_directory="./chroma_db", embedding_function=embadding)
    print("✅ ChromaDB 載入完成")

    # 3. BM25 index
    chunks = db.get()
    texts = chunks["documents"]
    tokenized = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    print("✅ BM25 索引建立完成")

    # 4. MiniMax LLM（OpenAI相容格式）
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        raise RuntimeError("請設定 MINIMAX_API_KEY 環境變數")

    llm = OpenAI(
        model="MiniMax-M2.5",
        openai_api_key=api_key,
        openai_api_base="https://api.minimax.io/v1",
        temperature=0.7,
        max_tokens=512
    )
    print("✅ MiniMax LLM 初始化完成")


def hybrid_search(query: str, k: int = 5, alpha: float = 0.4) -> list:
    """Hybrid Search: BM25 + Vector 融合檢索"""
    # BM25 分數
    bm25_scores = bm25.get_scores(query.split())
    bm25_norm = bm25_scores / (np.max(bm25_scores) + 1e-8)

    # Vector 分數
    results = db.similarity_search_with_score(query, k=20)
    vec_scores = {r.page_content: 1 - score for r, score in results}

    # 融合
    combined = {}
    for i, text in enumerate(texts):
        bm = bm25_norm[i]
        vc = vec_scores.get(text, 0)
        combined[text] = alpha * bm + (1 - alpha) * vc

    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:k]


# ========================
# FastAPI 主程式
# ========================
app = FastAPI(title="RAG Chat API", version="1.0.0")


class ChatRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    alpha: Optional[float] = 0.4


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.on_event("startup")
async def startup_event():
    initialize()


@app.get("/")
async def root():
    return {"message": "RAG Chat API is running", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    RAG Chat 端點：
    - 輸入問題，使用 Hybrid Search 檢索相關上下文
    - 將上下文 + 問題送入 LLM 生成回答
    """
    try:
        # 1. Hybrid Search 檢索
        results = hybrid_search(req.query, k=req.k, alpha=req.alpha)

        if not results:
            return ChatResponse(answer="抱歉，資料庫中找不到相關內容。", sources=[])

        # 2. 組出上下文（Context）
        context = "\n\n".join([text for text, _ in results])
        sources = [text[:200].replace("\n", " ") for text, _ in results]

        # 3. 組 Prompt
        prompt = f"""你是一個专业的问答助手，基於以下參考資料回答用戶問題。

            參考資料：
            {context}

            用戶問題：{req.query}

            請根據參考資料回答，如果資料庫中沒有相關資訊，請如實說明。回答：
            """
        # 4. 呼叫 LLM
        answer = llm.invoke(prompt)
        if isinstance(answer, str):
            answer = answer.strip()
        else:
            answer = str(answer)

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
