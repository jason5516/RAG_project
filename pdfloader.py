from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/paper.pdf")
pages = loader.load()
