import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import jieba
import numpy as np


embedding = None
db = None
texts = None
bm25 = None

def initialize_retriever():
    global embedding, db, texts, bm25

    if embedding is not None and db is not None and texts is not None and bm25 is not None:
        return

    # 載入 embadding 與 語義資料庫
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

    chunks = db.get()
    texts = chunks.get("documents", []) or []

    if not texts:
        bm25 = None
        return
    
    # 建立 BM25 index
    tokenized_chunks = [jieba.lcut(text) for text in texts]
    bm25 = BM25Okapi(tokenized_chunks)

# 建立 hybird_search 方法
def hybird_search(query, k=5, alpha=0.4):

    # 初始化資料庫
    initialize_retriever()

    #  BM25分數
    bm25_score = bm25.get_scores(jieba.lcut(query))
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

if __name__ == "__main__":
    # 測試
    print("\n=== Hybird Search 測試 ===")
    results =  hybird_search("學貸的申貸條件是什麼?")
    for i in range(len(results)):
        text, score = results[i]
        print(f"--- 結果 {i+1} (分數: {score:.4f}) ---")
        print(text[:300])
        print()


