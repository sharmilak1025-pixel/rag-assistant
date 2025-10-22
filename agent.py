from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embedding model and FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("/content/drive/MyDrive/rag-assistant/faiss_index", embedding_model, allow_dangerous_deserialization=True)

# ✅ Use text2text-generation for flan-t5-base
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_response(query):
    # Retrieve relevant chunks
    retrieved_docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Build prompt
    prompt = f"Based on the following product descriptions, answer the question clearly.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    print("Prompt sent to model...")
    response = generator(prompt, max_new_tokens=100)
    print("Response received.")
    return response[0]['generated_text']

# 🧪 Test block
if __name__ == "__main__":
    query = "Which product helps with noisy environments?"
    response = generate_response(query)
    print("Generated Response:\n", response)