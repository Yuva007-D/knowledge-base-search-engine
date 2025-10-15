import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file using pypdf"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def process_uploaded_file(self, file_path):
        """Process uploaded file and return chunks"""
        if file_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        else:
            # For text files
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        if not text.strip():
            return []
        
        # Create documents
        documents = [Document(page_content=text)]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_uploaded_files(self, uploaded_files, temp_dir="./data/temp"):
        """Process multiple uploaded files"""
        os.makedirs(temp_dir, exist_ok=True)
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the file
            chunks = self.process_uploaded_file(temp_path)
            all_chunks.extend(chunks)
            
            # Clean up temporary file
            os.remove(temp_path)
        
        return all_chunks