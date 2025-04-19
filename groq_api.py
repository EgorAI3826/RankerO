from groq import Groq
import os
import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_info(model_name: str) -> dict:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        parameters = re.search(r'(\d+\.?\d*)B', model_name)
        parameters = parameters.group(0) if parameters else 'unknown'
        return {
            "architecture": model_name,
            "parameters": parameters,
            "context_length": 4096,
            "embedding_length": 5120,
            "quantization": 'unknown'
        }
    except Exception as e:
        logging.error(f"Failed to fetch model info for {model_name}: {str(e)}")
        match = re.search(r'(\d+\.?\d*)B', model_name)
        parameters = match.group(0) if match else 'unknown'
        return {
            "architecture": 'unknown',
            "parameters": parameters,
            "context_length": 4096,
            "embedding_length": 5120,
            "quantization": 'unknown'
        }

def call_groq(prompt: str, model_name: str) -> str:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling Groq API: {e}")
        return ""