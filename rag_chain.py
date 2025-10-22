import os
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import

# ✅ Load product descriptions from local file
def load_documents(file_path="products.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [Document(page_content=line.strip()) for line in lines if line.strip()]

# ✅ Create embedding model
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load or rebuild FAISS index
def get_vectorstore(index_dir="faiss_index", products_file="products.txt"):
    index_path = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_path):
        return FAISS.load_local(index_dir, embedding_fn, allow_dangerous_deserialization=True)
    else:
        docs = load_documents(products_file)
        vectorstore = FAISS.from_documents(docs, embedding_fn)
        vectorstore.save_local(index_dir)
        return vectorstore

# ✅ Initialize vectorstore
vectorstore = get_vectorstore()

# ✅ Retrieval function
def retrieve_context(query, k=3):
    return vectorstore.similarity_search(query, k=k)

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
