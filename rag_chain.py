from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ Load product descriptions from file
with open("/content/drive/MyDrive/rag-assistant/products.txt", "r") as f:
    lines = f.readlines()

docs = [Document(page_content=line.strip()) for line in lines if line.strip()]

# ✅ Create embedding model and FAISS index
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_fn)

# ✅ Retrieval function
def retrieve_context(query):
    return vectorstore.similarity_search(query, k=3)

# ✅ Test block for standalone execution
if __name__ == "__main__":
    test_queries = [
        "wireless headphones",
        "eco-friendly water bottle",
        "gaming laptop with SSD"
    ]

    for query in test_queries:
        print(f"\n🔎 Query: {query}")
        results = retrieve_context(query)
        for i, doc in enumerate(results, 1):
            print(f"Result {i}: {doc.page_content}")