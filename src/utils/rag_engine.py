import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._initialize_model()
            except Exception as e:
                print(f"Configuration error: {e}")
    
    def _initialize_model(self):
        """Initialize the Gemini model"""
        try:
            # Use models that are actually available from your list
            model_names = [
                "models/gemini-2.0-flash",  # Available in your list
                "models/gemini-2.0-flash-001",  # Available in your list
                "models/gemini-flash-latest",  # Available in your list
                "models/gemini-pro-latest",  # Available in your list
            ]
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model with a simple prompt
                    test_response = self.model.generate_content("Hello")
                    print(f"✅ Successfully initialized model: {model_name}")
                    return
                except Exception as e:
                    print(f"❌ Failed with {model_name}: {e}")
                    continue
            
            print("❌ No compatible model found")
            
        except Exception as e:
            print(f"Error in model initialization: {e}")
    
    def generate_answer(self, query, context_docs):
        """Generate answer using RAG"""
        if not self.api_key:
            return "❌ Error: Please enter your Gemini API key in the sidebar."
        
        if not self.model:
            return "❌ Error: Could not initialize Gemini model. Please check your API key and try again."
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt
        prompt = f"""Using the following context from documents, answer the user's question succinctly and accurately.

Context:
{context}

Question: {query}

Answer the question based only on the provided context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided documents."

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"