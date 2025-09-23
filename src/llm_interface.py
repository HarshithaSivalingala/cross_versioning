import os
from typing import Optional

def call_llm(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Call Google Gemini API"""
    try:
        import google.generativeai as genai
        
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Use the model from environment or default
        model_name = os.getenv("ML_UPGRADER_MODEL", model)
        
        # Create model instance
        model = genai.GenerativeModel(model_name)
        
        # Generate response
        response = model.generate_content(prompt)
        
        if not response.text:
            raise Exception("Empty response from Gemini API")
            
        return response.text
        
    except ImportError:
        raise Exception("google-generativeai package not installed. Run: pip install google-generativeai")
    except Exception as e:
        raise Exception(f"Gemini API call failed: {str(e)}")s