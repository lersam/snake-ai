from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load your local PDF
loader = PyPDFLoader(r"C:\Users\ler_s\Downloads\makinggames.pdf")
data = loader.load()

# 2. Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# 3. Create Vector Store using Local Embeddings
# 'nomic-embed-text' is a great, lightweight embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = FAISS.from_documents(chunks, embeddings)

# 4. Initialize Local Chatbot (Llama 3)
llm = OllamaLLM(model="llama3")

# 5. Setup the Retrieval Chain
rag_bot = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

# 6. Chat with your document
query = "What is the key takeaway from this file?"
response = rag_bot.invoke(query)

print(f"Ollama RAG: {response['result']}")