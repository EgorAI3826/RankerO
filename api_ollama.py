import ollama
import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_info(model_name: str) -> dict:
    try:
        model_info = ollama.show(model_name)
        details = model_info.get('details', {})
        modelinfo = model_info.get('modelinfo', {})
        architecture = details.get('family', modelinfo.get('general.architecture', 'unknown'))
        parameters = details.get('parameter_size', 'unknown')
        if parameters == 'unknown':
            match = re.search(r'(\d+\.?\d*)B', model_name)
            parameters = match.group(0) if match else 'unknown'
        context_length = modelinfo.get('deepseek.context_length', 4096)
        embedding_length = modelinfo.get('deepseek.embedding_length', 5120)
        quantization = details.get('quantization_level', 'unknown')
        if quantization == 'unknown':
            quant_match = re.search(r':(q\d[^ ]*|f16)', model_name, re.IGNORECASE)
            quantization = quant_match.group(1).upper() if quant_match else 'unknown'
        return {
            "architecture": architecture,
            "parameters": parameters,
            "context_length": context_length,
            "embedding_length": embedding_length,
            "quantization": quantization
        }
    except Exception as e:
        logging.error(f"Failed to fetch model info for {model_name}: {str(e)}")
        quant_match = re.search(r':(q\d[^ ]*|f16)', model_name, re.IGNORECASE)
        quantization = quant_match.group(1).upper() if quant_match else 'unknown'
        match = re.search(r'(\d+\.?\d*)B', model_name)
        parameters = match.group(0) if match else 'unknown'
        return {
            "architecture": 'unknown',
            "parameters": parameters,
            "context_length": 4096,
            "embedding_length": 5120,
            "quantization": quantization
        }

def call_ollama(prompt: str, model_name: str) -> str:
    try:
        response = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0.1})
        return response['response'].strip()
    except Exception as e:
        logging.error(f"Error calling Ollama API: {e}")
        return ""