import os
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

class VectorStoreManager:
    def __init__(self, persist_directory="./vector_store"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        
    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self._save_vector_store()
        return self.vector_store
    
    def load_vector_store(self):
        """Load existing vector store"""
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                return self.vector_store
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return None
        return None
    
    def _save_vector_store(self):
        """Save vector store to disk"""
        if self.vector_store:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store.save_local(self.persist_directory)
    
    def similarity_search(self, query, k=4):
        """Perform similarity search"""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []
    
    def get_retriever(self):
        """Get retriever for RAG"""
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": 4})
        return None
    
    def get_document_count(self):
        """Get number of documents in vector store"""
        if self.vector_store:
            return self.vector_store.index.ntotal
        return 0