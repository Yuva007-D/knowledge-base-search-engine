import streamlit as st
import os
import sys
import tempfile

# Add src to path FIRST - before imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the utils
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="Knowledge Base Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS with better error handling
def load_css():
    try:
        css_file = os.path.join(os.path.dirname(__file__), "static", "style.css")
        if os.path.exists(css_file):
            with open(css_file, 'r', encoding='utf-8') as f:
                css_content = f.read()
                st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        else:
            # Fallback inline CSS if file is missing
            st.markdown("""
            <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 700;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #2e86ab;
                margin-bottom: 1rem;
                font-weight: 600;
            }
            .success-box {
                padding: 1rem;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 0.5rem;
                margin: 1rem 0;
                color: #155724;
            }
            .info-box {
                padding: 1rem;
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 0.5rem;
                margin: 1rem 0;
                color: #0c5460;
            }
            .chat-message {
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                border: 1px solid #e9ecef;
                color: #333333 !important;
            }
            .user-message {
                background-color: #e7f3ff;
                border-left: 4px solid #1f77b4;
                color: #333333 !important;
            }
            .assistant-message {
                background-color: #f8f9fa;
                border-left: 4px solid #28a745;
                color: #333333 !important;
            }
            .stChatMessage {
                color: #333333 !important;
            }
            body {
                color: #333333 !important;
            }
            </style>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

load_css()

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Knowledge Base Search Engine</h1>', unsafe_allow_html=True)
    st.markdown("### Upload documents and ask questions powered by RAG + Gemini AI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from: https://aistudio.google.com/app/apikey",
            placeholder="Enter your Gemini API key here..."
        )
        
        st.markdown("---")
        st.markdown("## 📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF or Text files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple PDF or text files to build your knowledge base"
        )
        
        # Process documents button
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            if uploaded_files and api_key:
                process_documents(uploaded_files, api_key)
            else:
                if not uploaded_files:
                    st.error("📝 Please upload at least one document")
                if not api_key:
                    st.error("🔑 Please enter your Gemini API key")
        
        # System info
        st.markdown("---")
        st.markdown("## 📊 System Info")
        
        if st.session_state.documents_processed:
            vector_manager = VectorStoreManager()
            doc_count = vector_manager.get_document_count()
            st.markdown(f"**Documents in index:** {doc_count}")
            st.markdown(f"**Files processed:** {len(st.session_state.processed_files)}")
            for file in st.session_state.processed_files:
                st.markdown(f"• {file}")
        else:
            st.markdown("No documents processed yet")
        
        st.markdown("---")
        st.markdown("## 💡 How to use:")
        st.markdown("""
        1. 🔑 Enter your Gemini API key
        2. 📁 Upload PDF/text files
        3. 🚀 Click 'Process Documents'
        4. 💬 Ask questions in the chat
        """)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">💬 Chat Interface</div>', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.documents_processed:
                st.error("📝 Please process documents first before asking questions")
            elif not api_key:
                st.error("🔑 Please enter your Gemini API key in the sidebar")
            else:
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Generate and display assistant response
                with st.spinner("🔍 Searching knowledge base..."):
                    response = get_rag_response(prompt, api_key)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
    
    with col2:
        st.markdown('<div class="sub-header">📊 System Status</div>', unsafe_allow_html=True)
        
        # System status
        if st.session_state.documents_processed:
            st.markdown('<div class="success-box">✅ Documents processed and ready for queries</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">📝 Waiting for document upload and processing</div>', unsafe_allow_html=True)
        
        # Quick questions suggestions
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        sample_questions = [
            "What are the main topics covered?",
            "Summarize the key points from the documents",
            "What is the objective of this project?",
            "List the technical requirements mentioned"
        ]
        
        for question in sample_questions:
            if st.button(question, key=question, use_container_width=True):
                if st.session_state.documents_processed and api_key:
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    with st.spinner("🔍 Searching knowledge base..."):
                        response = get_rag_response(question, api_key)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
                else:
                    st.error("Please ensure documents are processed and API key is entered")

def process_documents(uploaded_files, api_key):
    """Process uploaded documents and create vector store"""
    with st.spinner("Processing documents..."):
        try:
            # Initialize components
            doc_processor = DocumentProcessor()
            vector_manager = VectorStoreManager()
            
            # Process files
            all_chunks = doc_processor.process_uploaded_files(uploaded_files)
            
            if not all_chunks:
                st.error("❌ No text could be extracted from the uploaded files")
                return
            
            # Create vector store
            vector_store = vector_manager.create_vector_store(all_chunks)
            st.session_state.vector_store = vector_store
            st.session_state.documents_processed = True
            st.session_state.processed_files = [file.name for file in uploaded_files]
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} files and created {len(all_chunks)} text chunks!")
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")

def get_rag_response(query, api_key):
    """Get RAG response for query"""
    try:
        vector_manager = VectorStoreManager()
        vector_store = vector_manager.load_vector_store()
        
        if not vector_store:
            return "❌ Error: No vector store found. Please process documents first."
        
        # Perform similarity search
        relevant_docs = vector_store.similarity_search(query, k=4)
        
        if not relevant_docs:
            return "❌ No relevant information found in the documents for your query."
        
        # Generate answer using RAG
        rag_engine = RAGEngine(api_key=api_key)
        answer = rag_engine.generate_answer(query, relevant_docs)
        
        return answer
        
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

if __name__ == "__main__":
    main()