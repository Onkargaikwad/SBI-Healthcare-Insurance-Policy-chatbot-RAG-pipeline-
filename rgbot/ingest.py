import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "data", "faiss_index") # Assuming 'data' folder at root

def ingest_data(pdf_path: str, index_dir: str = INDEX_DIR):
    if os.path.exists(index_dir):
        print(f"Loading existing index from {index_dir}...")
        embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vstore = FAISS.load_local(
            folder_path=index_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        return vstore
    
    print("Building new index - this may take a while...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vstore.save_local(folder_path=index_dir)
    print(f"Index saved to {index_dir}.")
    return vstore

if __name__ == "__main__":
    pdf_path = os.path.join(SCRIPT_DIR, "data", "SBI_General_Health_Insurance.pdf")
    vstore = ingest_data(pdf_path)
    print("Ingestion complete. Vector store ready.")
    
    
