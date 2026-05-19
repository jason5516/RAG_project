import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import numpy as np
import re

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
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    llm = ChatOpenAI(
        model="MiniMax-M2.7",
        openai_api_key=api_key,
        openai_api_base="https://api.minimax.io/v1",
        temperature=0.7
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

class ChatResponse(BaseModel):
    answer: str
    think: str = ""
    sources: list[str] = []

def chat(req):
    # Hybird search 結果
    results = hybird_search(req, k=3, alpha=0.4)

    # 取出chunk文字
    context_chunks = [text for text, _ in results]

    # 組成prompt
    context = "\n\n".join(context_chunks)
    prompt = f"""你是一個專業的問答助手，根據以下參考資料回答問題。請用中文回答。
    
        資料：{context}
        
        問題：{req}
        
        回答：
        
    """

    # 呼叫 LLM
    answer = llm.invoke(prompt).content
    print(f"原始回答：{answer}")
    think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
    think = think_match.group(1) if think_match else ""
    final_answer = re.sub(r'<think>.*?</think>', "", answer, flags=re.DOTALL).strip()

    # 取 sources
    sources = [chunk[:500].replace("\n", " ") for chunk in context_chunks]

    return ChatResponse(think=think, answer=final_answer, sources=sources)

if __name__ == '__main__':
    initialize()
    
    requst = f"申請學貸的條件是什麼?"
    response = chat(requst)
