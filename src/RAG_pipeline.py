import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import shutil


CHUNK_SIZE=500
CHUNK_OVERLAP=150

def load_pdf_documents(data_dir: Path) -> list[Document]:
    pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"找不到 PDF：{data_dir}")

    documents = []

    # 將 PDF 檔案依序使用 loader 讀取進來
    for pdf_path in pdf_paths:
        pages = PyPDFLoader(str(pdf_path)).load()
        doc_id = pdf_path.stem

        for page_index, page in enumerate(pages):
            page.metadata.update(
                {
                    "source_file": pdf_path.name,
                    "doc_id": doc_id,
                    "page": page_index + 1,
                }
            )
            documents.append(page)

    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    # 將讀取的資料作切塊
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        separators=["\n\n", "\n", "。", "，", " ", ""])
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        doc_id = chunk.metadata["doc_id"]
        chunk.metadata["chunk_id"] = f"{doc_id}-{index:04d}"

    return chunks

def build_vectorstore(chunks: list[Document], persist_dir: Path) -> None:

    # 檢查路徑是否有效，且資料夾名稱是否正確
    if persist_dir.name != "chroma_db":
        raise ValueError(f"拒絕刪除非資料庫目錄：{persist_dir}")

    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    # 讀取 embadding 模型
    embadding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={"device": "cpu"})

    # 使用 embadding 模型將詞轉成向量儲存
    db = Chroma.from_documents(
        chunks, 
        embadding, 
        persist_directory=str(persist_dir),
    )

    db.persist()

def rebuild_knowledge_base() -> dict:
    documents = load_pdf_documents(DEFAULT_DATA_DIR)
    chunks = split_documents(documents)
    build_vectorstore(chunks, DEFAULT_PERSIST_DIR)

    return {
        "document_count": len({doc.metadata["doc_id"] for doc in documents}),
        "page_count": len(documents),
        "chunk_count": len(chunks),
    }


def main() -> None:
    result = rebuild_knowledge_base()

    print(f"文件數：{result['document_count']}")
    print(f"載入頁數：{result['page_count']}")
    print(f"建立 chunk 數：{result['chunk_count']}")
    print(f"完成建庫：{DEFAULT_PERSIST_DIR}")


if __name__ == "__main__":
    main()
