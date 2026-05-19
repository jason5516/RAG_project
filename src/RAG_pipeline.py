from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


# 將 PDF 檔案使用 loader 讀取進來
loader = PyPDFLoader("data/就學貸款.pdf")
pages = loader.load()

# 將讀取的資料作切塊
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150, separators=["\n\n", "\n", "。", "，", " ", ""])
chunks = splitter.split_documents(pages)
print(f"切成 {len(chunks)} 個區塊")

# 讀取 embadding 模型
embadding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={"device": "cpu"})

# 使用 embadding 模型將詞轉成向量儲存
db = Chroma.from_documents(
    chunks, embadding, persist_directory="./chroma_db"
)

db.persist()

print("完成！向量資料庫已保存到 ./chroma_db")
