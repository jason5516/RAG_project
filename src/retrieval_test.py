import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# 載入 embadding 與 資料庫
embadding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
db = Chroma(persist_directory="./chroma_db", embedding_function=embadding)

# 進行相似度檢索(similarity)
quary = "what is the contribution?"
results = db.similarity_search(quary, k=3)

print(f"找到 {len(results)} 比結果")
for i, r in enumerate(results):
    print(f"=====結果 {i+1}=====")
    print(r.page_content[:200])
    print()

# 進行 MMR(多樣性檢索)
print("+++++MMR多樣性檢索+++++")
results_mmr = db.max_marginal_relevance_search(quary, k=3, fetch_k=10)
for i, r in enumerate(results_mmr):
    print(f"=====結果 {i+1}=====")
    print(r.page_content[:200])
    print()