import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# LM Studio API configuration
LM_STUDIO_HOST = os.environ.get("LM_STUDIO_HOST", "localhost")
LM_STUDIO_PORT = os.environ.get("LM_STUDIO_PORT", "1234")
LM_STUDIO_API_BASE = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1"

def call_lm_studio(prompt: str, model_name: str) -> str:
    """Call the LM Studio API to generate a response for the given prompt."""
    try:
        url = f"{LM_STUDIO_API_BASE}/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 5000,
            "temperature": 0.1,
            "stop": ["</s>", "<|eot_id|>"]
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "")
    except Exception as e:
        logging.error(f"Error calling LM Studio API: {e}")
        return ""

def get_model_info(model_name: str) -> dict:
    """Retrieve model information from LM Studio."""
    try:
        url = f"{LM_STUDIO_API_BASE}/models"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
        for model in models:
            if model.get("id") == model_name:
                return {
                    "architecture": model.get("architecture", "unknown"),
                    "parameters": model.get("parameter_count", "unknown"),
                    "context_length": model.get("max_context_length", 4096),
                    "embedding_length": model.get("embedding_length", 5120),
                    "quantization": model.get("quantization", "unknown")
                }
        logging.warning(f"Model {model_name} not found in LM Studio.")
        return {
            "architecture": "unknown",
            "parameters": "unknown",
            "context_length": 4096,
            "embedding_length": 5120,
            "quantization": "unknown"
        }
    except Exception as e:
        logging.error(f"Error retrieving model info from LM Studio: {e}")
        return {
            "architecture": "unknown",
            "parameters": "unknown",
            "context_length": 4096,
            "embedding_length": 5120,
            "quantization": "unknown"
        }