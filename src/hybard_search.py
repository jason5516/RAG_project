import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import numpy as np

# 載入 embadding 與 語義資料庫
embadding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
db = Chroma(persist_directory="./chroma_db", embedding_function=embadding)



# 建立 BM25 index
chunks = db.get()
texts = chunks["documents"]
tokenizer_chunk = [text.split() for text in texts]
bm25 = BM25Okapi(tokenizer_chunk)
print("bm25 建立完成")

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

# 測試
print("\n=== Hybird Search 測試 ===")
results =  hybird_search("contributions of this paper")
for i in range(len(results)):
    text, score = results[i]
    print(f"--- 結果 {i+1} (分數: {score:.4f}) ---")
    print(text[:300])
    print()


