import os
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings  # updated import

# ✅ Load product descriptions from local file
with open("products.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

docs = [Document(page_content=line.strip()) for line in lines if line.strip()]

# ✅ Create embedding model
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load or rebuild FAISS index
index_path = "faiss_index/index.faiss"
if os.path.exists(index_path):
    vectorstore = FAISS.load_local("faiss_index", embedding_fn, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(docs, embedding_fn)
    vectorstore.save_local("faiss_index")

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
