import google.generativeai as genai  
import os  
from dotenv import load_dotenv  
  
load_dotenv()  
  
api_key = os.getenv('GEMINI_API_KEY')  
print(f"API Key: {api_key[:10]}...") if api_key else print("No API key found")  
  
if api_key:  
    genai.configure(api_key=api_key)  
  
    try:  
        models = genai.list_models()  
        print("Available models:")  
        for model in models:  
            print(f"- {model.name}")  
    except Exception as e:  
        print(f"Error: {e}")  
else:  
    print("No API key found in .env file") 
