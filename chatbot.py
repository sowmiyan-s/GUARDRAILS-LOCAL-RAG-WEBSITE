"""
GUARDRAILS LOCAL RAG BOT — CLI Interface
=========================================
A terminal-based RAG chatbot that queries a local PDF document using
a local Ollama LLM and offline HuggingFace embeddings (all-MiniLM-L6-v2).

Usage:
    python chatbot.py --pdf path/to/document.pdf

GitHub: https://github.com/sowmiyan-s
License: MIT
"""

import os
import argparse
import hashlib
from dotenv import load_dotenv

# Fix OMP error for FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def build_rag(pdf_path: str):
    """
    Builds the RAG pipeline processing the PDF and creating vectorstore and chains.
    """
    # 1. Load PDF Document
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split Document into smaller Chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # 3. Create Vector Store with HuggingFace Embeddings (100% Offline via sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    
    persist_dir = os.path.join(".faiss_storage_cli", file_hash + "_Offline")
    
    if os.path.exists(persist_dir):
        print("Loaded offline embeddings from cache...")
        vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating embeddings and vector store using HuggingFace all-MiniLM-L6-v2 (100% Offline)...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(persist_dir)

    # 4. Set up Retriever and Language Model
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model="gemma3:1b") 

    # 5. Create RAG Chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer as concise as possible based on the context.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create document combination chain and return retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def main():
    parser = argparse.ArgumentParser(
        description="GUARDRAILS LOCAL RAG BOT — CLI",
        epilog="GitHub: https://github.com/sowmiyan-s"
    )
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file to query")
    parser.add_argument("--model", type=str, default="gemma3:1b", help="Ollama model name (default: gemma3:1b)")
    args = parser.parse_args()

    # Check if PDF exists
    if not os.path.exists(args.pdf):
        print(f"Error: The file '{args.pdf}' does not exist.")
        return

    # Initialize RAG Pipeline
    try:
        rag_chain = build_rag(args.pdf)
    except Exception as e:
        print(f"Error building RAG Chatbot: {e}")
        return

    print("\n" + "="*60)
    print(" GUARDRAILS LOCAL RAG BOT — CLI")
    print(" Powered by Ollama + HuggingFace Embeddings (Offline)")
    print(" Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")

    # Start chat loop
    while True:
        try:
            question = input("\nYou: ")
            if question.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not question.strip():
                continue

            response = rag_chain.invoke({"input": question})
            print(f"\nChatbot: {response['answer']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
