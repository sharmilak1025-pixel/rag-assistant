import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load FAISS index and embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load text generation pipeline
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

def retrieve_context(query):
    results = db.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

def generate_response(query, context):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context: {context}

Question: {query}

Answer:"""
    response = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

def main():
    st.title("🧠 RAG Assistant")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Retrieving context..."):
            context = retrieve_context(query)
        with st.spinner("Generating response..."):
            response = generate_response(query, context)
        st.subheader("📚 Retrieved Context")
        st.write(context)
        st.subheader("💬 Generated Response")
        st.write(response)

# 🧪 Test block
if __name__ == "__main__":
    main()