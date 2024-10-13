import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Embedding modelini yükleme
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS vektör veritabanını yükleme veya oluşturma
index_path = "C:/Users/YENER/Desktop/ragmodel/faiss_index"
if os.path.exists(index_path):
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:
    texts = ["Örnek metin 1", "Örnek metin 2", "Örnek metin 3"]
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(index_path)
    print(f"Yeni index oluşturuldu ve {index_path} konumuna kaydedildi.")

# Ollama modelini yükleme
llm = OllamaLLM(model="llama3.1:8b")

# RAG zincirini oluşturma
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_question = request.json['question']
    answer = qa_chain.run(user_question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)