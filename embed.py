from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def create_vector_db():
    # Load product descriptions from file
    with open("/content/drive/MyDrive/rag-assistant/products.txt", "r") as f:
        lines = f.readlines()

    # Convert each line into a Document
    documents = [Document(page_content=line.strip()) for line in lines if line.strip()]

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    db = FAISS.from_documents(documents, embedding_model)

    # Save index locally
    db.save_local("/content/drive/MyDrive/rag-assistant/faiss_index")

# 🧪 Test block
if __name__ == "__main__":
    create_vector_db()
    print("FAISS index created successfully.")