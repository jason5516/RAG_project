import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import numpy as np

# 全域變數
embedding = None
db = None
bm25 = None
texts = None
llm = None

def initialize():
    # 使用全域變數
    global embedding, db, bm25, texts, llm

    # 載入embadding與資料庫
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    print("✅ Embedding 模型載入完成")

    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    print("✅ ChromaDB 載入完成")

    # 建立 BM25 index
    chunks = db.get()
    texts = chunks["documents"]
    tokenizer_chunk = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenizer_chunk)

    # 載入llm模型
    api_key = os.environ.get("MINIMAX_API_KEY", "***REMOVED***")
    llm = ChatOpenAI(
        model="MiniMax-M2.7",
        openai_api_key=api_key,
        openai_api_base="https://api.minimax.io/v1",
        temperature=0.7,
        max_tokens=512
    )
    print("✅ MiniMax LLM 初始化完成")

# 建立 hybird_search 方法
def hybird_search(query, k=5, alpha=0.4):
    #  BM25分數
    bm25_score = bm25.get_scores(query.split())
    bm25_scores = bm25_score / (np.max(bm25_score) + 1e-8)

    #  similarity 分數
    results = db.similarity_search_with_score(query, k=20)
    vec_scores = {r.page_content: 1 - score for r, score in results}

    # 合併分數
    combined = {}
    for i, text in enumerate(texts):
        bm = bm25_scores[i]
        vc = vec_scores.get(text, 0)
        combined[text] = alpha * bm + (1-alpha) * vc

    # 排序取 Top K
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時進行初始化
    initialize()
    yield

app = FastAPI(title="RAG chat API", lifespan=lifespan)

class ChatRequst(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
async def health():
    return {"status":"ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequst):
    # Hybird search 結果
    results = hybird_search(req.query, k=5, alpha=0.4)

    # 取出chunk文字
    context_chunks = [text for text, _ in results]

    # 組成prompt
    context = "\n\n".join(context_chunks)
    prompt = f"""根據以下資料回答問題。
    
        資料：{context}
        
        問題：{req.query}
        
        回答：
        
    """

    # 呼叫 LLM
    answer = llm.invoke(prompt).content

    # 取 sources
    sources = [chunk[:500].replace("\n", " ") for chunk in context_chunks]

    return ChatResponse(answer=answer, sources=sources)
