# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)
CORS(app)

# Initialize components
def initialize_components():
    # Document loading and processing
    loader = DirectoryLoader(
        "/home/koushik-muvva/Desktop/Research(never_ending)/data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Embeddings and vector store
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    
    # LLM and QA chain
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-70b-8192"
    )
    
    # Mental health focused prompt
    prompt_template = """You are a mental health specialist assistant. Follow these rules:
    1. Only answer questions about mental health, emotions, or psychological well-being
    2. If asked about other topics, respond: "I specialize in mental health. How can I help you with emotional concerns?"
    3. Use this context to help:
    {context}
    
    Question: {question}
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

qa_chain = initialize_components()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"response": "Please enter a valid question"})
    
    try:
        result = qa_chain.invoke({"query": user_input})
        return jsonify({"response": result["result"]})
    except Exception as e:
        return jsonify({"response": f"Error processing request: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
