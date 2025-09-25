import openai
import re

# export OPENAI_API_KEY="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"   

def generate(prompt):
    try:
        client = openai.OpenAI(
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
            )
        
        response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",  # Or any other OpenRouter-supported model
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000
            )
        print(len(response.choices))
        print(response.choices[0].message)
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenRouter API error: {str(e)}")


def call_llm(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Call Google Gemini API"""
    try:
        response = generate(prompt)
        
        if not response:
            raise Exception("Empty response from OpenAI API")
            
        return response
        
    except ImportError:
        raise Exception("openai package not installed. Run: pip install openai")
    except Exception as e:
        raise Exception(f" API call failed: {str(e)}")
    

