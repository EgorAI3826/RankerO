import json
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import os
from typing import Dict, List, Tuple
from fractions import Fraction
import csv
import pandas as pd
import queue
from threading import Thread, Event
from api_ollama import call_ollama
from openai_api import call_openai
from groq_api import call_groq
from openrouter_api import call_openrouter
from lm_studio_api import call_lm_studio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MERA_DATASET_DIR = os.path.join(DATASET_DIR, "MERA")

os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    # BASE Datasets
    "AIME 1983-2024 (Mathematics)": {"path": os.path.join(DATASET_DIR, "aime_1983_2024_processed.json"), "threshold": 0.85, "category": "BASE"},
    "GSM8K Practice (Mathematics)": {"path": os.path.join(DATASET_DIR, "gsm8k_train_processed.json"), "threshold": 0.80, "category": "BASE"},
    "GSM8K (Mathematics)": {"path": os.path.join(DATASET_DIR, "gsm8k_processed.json"), "threshold": 0.80, "category": "BASE"},
    "HumanEval (Programming)": {"path": os.path.join(DATASET_DIR, "humaneval_processed.json"), "threshold": 0.90, "category": "BASE"},
    "Thinking Traps (Logic)": {"path": os.path.join(DATASET_DIR, "thinking_traps_processed.json"), "threshold": 0.85, "category": "BASE"},
    "MMLU Pro Test (General Knowledge)": {"path": os.path.join(DATASET_DIR, "mmlu_pro_test_processed.json"), "threshold": 0.85, "category": "BASE"},
    "MMLU Pro (General Knowledge)": {"path": os.path.join(DATASET_DIR, "mmlu_pro_processed.json"), "threshold": 0.85, "category": "BASE"},
    "Yubo2333 (Miscellaneous)": {"path": os.path.join(DATASET_DIR, "yubo2333_processed.json"), "threshold": 0.85, "category": "BASE"},
    "Google Frames (Language)": {"path": os.path.join(DATASET_DIR, "google_frames_processed.json"), "threshold": 0.85, "category": "BASE"},
    "Math 500 (Mathematics)": {"path": os.path.join(DATASET_DIR, "math_500_processed.json"), "threshold": 0.85, "category": "BASE"},
    "HumanEval XL (Programming)": {"path": os.path.join(DATASET_DIR, "humaneval_xl_processed.json"), "threshold": 0.90, "category": "BASE"},
    "RWSD Combined (Language/Reading)": {"path": os.path.join(DATASET_DIR, "rwsd_combined_processed.json"), "threshold": 0.85, "category": "BASE"},
    "RCB Combined (Language/Reading)": {"path": os.path.join(DATASET_DIR, "rcb_combined_processed.json"), "threshold": 0.85, "category": "BASE"},
    "Parus Questions (Language)": {"path": os.path.join(DATASET_DIR, "parus_questions_processed.json"), "threshold": 0.85, "category": "BASE"},
    "Simple Questions (Basic Mathematics)": {
        "data": [
            {"question": "What is 2 + 2?", "expected_answer": "4", "category": "math"},
            {"question": "What is the square root of 16?", "expected_answer": "4", "category": "math"},
            {"question": "Solve for x: 3x = 9", "expected_answer": "3", "category": "math"},
            {"question": "If $f(x) = \\frac{3x-2}{x+2}$, what is the value of $f(2) + f(-1) + f(0)$? Express your answer as a common fraction.", "expected_answer": "\\frac{14}{3}", "category": "math"}
        ],
        "threshold": 0.90,
        "category": "BASE"
    },
    # MERA Datasets
    "MERA/Chegeka (Miscellaneous)": {"path": os.path.join(MERA_DATASET_DIR, "converted_chegeka.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/LCS (Miscellaneous)": {"path": os.path.join(MERA_DATASET_DIR, "converted_lcs.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/Mamuramu (Miscellaneous)": {"path": os.path.join(MERA_DATASET_DIR, "converted_mamuramu.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/MathLogicQA (Mathematics/Logic)": {"path": os.path.join(MERA_DATASET_DIR, "converted_mathlogicqa.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/MultiQ (Language)": {"path": os.path.join(MERA_DATASET_DIR, "converted_multiq.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/Parus (Language)": {"path": os.path.join(MERA_DATASET_DIR, "converted_parus.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuCodeEval (Programming)": {"path": os.path.join(MERA_DATASET_DIR, "converted_rucodeeval.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuDetox (Moderation)": {"path": os.path.join(MERA_DATASET_DIR, "converted_rudetox.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuEthics (Ethics)": {"path": os.path.join(MERA_DATASET_DIR, "converted_ruethics.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuHateSpeech (Moderation)": {"path": os.path.join(MERA_DATASET_DIR, "converted_ruhatespeech.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuHHH (Helpfulness)": {"path": os.path.join(MERA_DATASET_DIR, "converted_ruhhh.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuHumanEval (Programming)": {"path": os.path.join(MERA_DATASET_DIR, "converted_ruhumaneval.json"), "threshold": 0.90, "category": "MERA"},
    "MERA/RuMMLU (General Knowledge)": {"path": os.path.join(MERA_DATASET_DIR, "converted_rummlu.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuModar (Moderation)": {"path": os.path.join(MERA_DATASET_DIR, "converted_rumodar.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuMultiAR (Reasoning)": {"path": os.path.join(MERA_DATASET_DIR, "converted_rumultiar.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RuTIE (Language/Information Extraction)": {"path": os.path.join(MERA_DATASET_DIR, "converted_rutie.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/RWSD (Language/Reading)": {"path": os.path.join(MERA_DATASET_DIR, "converted_rwsd.json"), "threshold": 0.85, "category": "MERA"},
    "MERA/SimpleAR (Reasoning)": {"path": os.path.join(MERA_DATASET_DIR, "converted_simplear.json"), "threshold": 0.85, "category": "MERA"},
}

class ModelEvaluator:
    def __init__(self, embedding_model="bge-m3:567m-fp16"):
        self.embedding_model = embedding_model
        self.api_providers = {
            "ollama": call_ollama,
            "openai": call_openai,
            "groq": call_groq,
            "openrouter": call_openrouter,
            "lm_studio": call_lm_studio
        }
        self.embedding_cache = {}
        self.results_table = []

    def clean_latex(self, text: str) -> str:
        """Remove LaTeX formatting from text."""
        text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\left\(', '(', text)
        text = re.sub(r'\\right\)', ')', text)
        text = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
        text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
        text = re.sub(r'\\pi', 'pi', text)
        text = re.sub(r'\\fac\{([^}]+)\}', r'factorial(\1)', text)
        text = re.sub(r'\\box\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[\{\}\[\]]', '', text)
        text = text.replace(',', '').strip()
        return text

    def normalize_fraction(self, text: str) -> str:
        """Normalize fractions to their simplest form."""
        try:
            match = re.match(r'(\d+)/(\d+)', text)
            if match:
                num, den = int(match.group(1)), int(match.group(2))
                frac = Fraction(num, den)
                return f"{frac.numerator}/{frac.denominator}"
        except (ValueError, ZeroDivisionError):
            pass
        return text

    def try_numeric_comparison(self, predicted: str, expected: str) -> Tuple[bool, float]:
        """Compare numeric values or fractions."""
        try:
            pred_match = re.match(r'(\d+)/(\d+)', predicted)
            exp_match = re.match(r'(\d+)/(\d+)', expected)
            if pred_match and exp_match:
                pred_num, pred_den = int(pred_match.group(1)), int(pred_match.group(2))
                exp_num, exp_den = int(exp_match.group(1)), int(exp_match.group(2))
                pred_val = pred_num / pred_den
                exp_val = exp_num / exp_den
                return abs(pred_val - exp_val) < 1e-5, 1.0 if abs(pred_val - exp_val) < 1e-5 else 0.0
            pred_val = float(predicted)
            exp_val = float(expected)
            return abs(pred_val - exp_val) < 1e-5, 1.0 if abs(pred_val - exp_val) < 1e-5 else 0.0
        except (ValueError, ZeroDivisionError):
            return False, 0.0

    def get_model_response(self, question: str, model_name: str, provider: str) -> str:
        """Get response from the specified model and provider."""
        prompt = (
            "Solve the following problem and return ONLY the final answer in the format <Final Answer / answer>. "
            "Do not include explanations, intermediate steps, or additional text. "
            "If the problem requires code, include only the complete, functional code within <Final Answer / answer>. "
            "For example, for a numerical answer like 902, return: <Final Answer / 902>. "
            "For a code solution, return: <Final Answer / print('Hello')>. \n\n"
            f"Problem: {question}\n\n"
            "Return only the final answer in the format <Final Answer / answer> below:"
        )
        if provider in self.api_providers:
            return self.api_providers[provider](prompt, model_name)
        else:
            logging.error(f"Provider {provider} not supported")
            return ""

    def extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the model's response."""
        match = re.search(r'<Final Answer / (.*?)>', response)
        return match.group(1).strip() if match else None

    def compare_answers(self, predicted: str, expected: str, threshold: float) -> Tuple[bool, float]:
        """Compare predicted and expected answers using embeddings or exact match."""
        if not predicted or not expected:
            return False, 0.0
        predicted_clean = self.clean_latex(predicted)
        expected_clean = self.clean_latex(expected)
        predicted_clean = self.normalize_fraction(predicted_clean)
        expected_clean = self.normalize_fraction(expected_clean)
        is_numeric_match, numeric_similarity = self.try_numeric_comparison(predicted_clean, expected_clean)
        if is_numeric_match:
            return True, numeric_similarity
        if predicted_clean == expected_clean:
            return True, 1.0
        try:
            if predicted_clean not in self.embedding_cache:
                self.embedding_cache[predicted_clean] = ollama.embeddings(model=self.embedding_model, prompt=predicted_clean)['embedding']
            if expected_clean not in self.embedding_cache:
                self.embedding_cache[expected_clean] = ollama.embeddings(model=self.embedding_model, prompt=expected_clean)['embedding']
            pred_embedding = self.embedding_cache[predicted_clean]
            exp_embedding = self.embedding_cache[expected_clean]
            similarity = cosine_similarity([pred_embedding], [exp_embedding])[0][0]
            return similarity >= threshold, similarity
        except Exception as e:
            logging.error(f"Error computing embeddings: {e}")
            return False, 0.0

    def evaluate_model(self, dataset: List[Dict], model_name: str, provider: str, dataset_full_name: str,
                      threshold: float, update_queue: queue.Queue, update_event: Event) -> Dict:
        """Evaluate a model on a dataset."""
        total = len(dataset)
        correct = 0
        results = []
        category_correct = {"reasoning": 0, "coding": 0, "math": 0, "data_analysis": 0, "language": 0, "if": 0}
        category_total = {"reasoning": 0, "coding": 0, "math": 0, "data_analysis": 0, "language": 0, "if": 0}
        detailed_results = []
        for i, record in enumerate(dataset):
            question = record["question"]
            expected_answer = record["expected_answer"]
            category = record.get("category", "math")
            if category in category_total:
                category_total[category] += 1
            else:
                category_total[category] = 1
                category_correct[category] = 0
            try:
                predicted_response = self.get_model_response(question, model_name, provider)
                predicted_answer = self.extract_final_answer(predicted_response)
                if predicted_answer is None:
                    continue
                is_correct, similarity = self.compare_answers(predicted_answer, expected_answer, threshold)
                if is_correct:
                    correct += 1
                    if category in category_correct:
                        category_correct[category] += 1
                result = {
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "expected_answer": expected_answer,
                    "is_correct": is_correct,
                    "similarity": similarity,
                    "category": category
                }
                results.append(result)
                detailed_results.append({
                    "Question": question[:100] + "..." if len(question) > 100 else question,
                    "Predicted Answer": predicted_answer,
                    "Expected Answer": expected_answer,
                    "Embedding": f"{similarity:.4f}",
                    "Correct": "Yes" if result["is_correct"] else "No"
                })
                if len(detailed_results) > 100:
                    detailed_results.pop(0)
                accuracy = (correct / total) * 100 if total > 0 else 0
                update_queue.put((
                    "progress",
                    f"Dataset: {dataset_full_name}\n"
                    f"Model: {model_name} (Provider: {provider})\n"
                    f"Processed: {i+1}/{total}\n"
                    f"Correct answers: {correct}/{i+1}\n"
                    f"Accuracy: {accuracy:.2f}%",
                    pd.DataFrame(detailed_results),
                    i + 1,
                    None
                ))
                update_event.set()
            except Exception as e:
                logging.error(f"Error on record {i+1}: {e}")
                update_queue.put((
                    "error",
                    f"Error processing record {i+1}: {str(e)}",
                    None,
                    i + 1,
                    None
                ))
                update_event.set()
                continue
        scores = {
            "global": (correct / total) * 100 if total > 0 else 0
        }
        for cat in category_total:
            scores[cat] = (category_correct[cat] / category_total[cat]) * 100 if category_total[cat] > 0 else 0
        update_queue.put((
            "final",
            f"Dataset: {dataset_full_name}\n"
            f"Model: {model_name} (Provider: {provider})\n"
            f"Processed: {total}/{total}\n"
            f"Correct answers: {correct}/{total}\n"
            f"Accuracy: {scores['global']:.2f}%",
            pd.DataFrame(detailed_results),
            total,
            None
        ))
        update_event.set()
        self.save_to_csv(detailed_results, model_name, dataset_full_name)
        return {"correct": correct, "total": total, "accuracy": scores["global"], "results": results, "scores": scores}

    def save_to_csv(self, detailed_results: List[Dict], model_name: str, dataset_full_name: str):
        """Save detailed results to a CSV file."""
        filename = os.path.join(
            DATA_DIR,
            f"{model_name.replace('/', '_').replace(':', '_')}_{dataset_full_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}_detailed.csv"
        )
        with open(filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Predicted Answer", "Expected Answer", "Embedding", "Correct"])
            for result in detailed_results:
                writer.writerow([
                    result["Question"],
                    result["Predicted Answer"],
                    result["Expected Answer"],
                    result["Embedding"],
                    result["Correct"]
                ])

def load_dataset(dataset_info: Dict) -> List[Dict]:
    """Load a dataset from a file or dictionary."""
    try:
        if isinstance(dataset_info, dict) and "data" in dataset_info:
            return dataset_info["data"]
        with open(dataset_info["path"], 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return []

def generate_html_table(global_results: List[Dict]) -> str:
    """Generate an HTML table for evaluation results."""
    sorted_results = sorted(global_results, key=lambda x: x.get("accuracy", 0), reverse=True)
    html = "<table border='1' style='border-collapse: collapse; width: 100%; font-family: Arial;'>\n"
    html += (
        "<tr style='background-color: #333; color: white;'>"
        "<th>MODEL</th><th>ORGANIZATION</th><th>DATASET</th><th>ACCURACY</th>"
        "<th>REASONING</th><th>CODING</th><th>MATHEMATICS</th>"
        "<th>DATA ANALYSIS</th><th>LANGUAGE</th><th>IF</th></tr>\n"
    )
    for i, record in enumerate(sorted_results):
        scores = record.get("scores", {})
        bg_color = "#f2f2f2" if i % 2 == 0 else "#ffffff"
        html += f"<tr style='background-color: {bg_color};'>"
        html += (
            f"<td>{record.get('model', 'N/A')}</td><td>{record.get('provider', 'N/A')}</td>"
            f"<td>{record.get('dataset', 'N/A')}</td><td>{record.get('accuracy', 0):.2f}</td>"
            f"<td>{scores.get('reasoning', 0):.2f}</td><td>{scores.get('coding', 0):.2f}</td>"
            f"<td>{scores.get('math', 0):.2f}</td>"
            f"<td>{scores.get('data_analysis', 0):.2f}</td><td>{scores.get('language', 0):.2f}</td>"
            f"<td>{scores.get('if', 0):.2f}</td></tr>\n"
        )
    html += "</table>"
    return html

def load_existing_results(csv_file: str) -> pd.DataFrame:
    """Load existing benchmark results from a CSV file."""
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_dataset_names = list(DATASETS.keys())
            expected_columns = ["Model", "Organization", "Architecture", "Parameters",
                                "Context Length", "Embedding Length", "Quantization"] + all_dataset_names
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = "-"
            df = df[expected_columns]
            return df
        all_dataset_names = list(DATASETS.keys())
        return pd.DataFrame(columns=["Model", "Organization", "Architecture", "Parameters",
                                     "Context Length", "Embedding Length", "Quantization"] + all_dataset_names)
    except Exception as e:
        logging.error(f"Error loading existing results from {csv_file}: {e}")
        all_dataset_names = list(DATASETS.keys())
        return pd.DataFrame(columns=["Model", "Organization", "Architecture", "Parameters",
                                     "Context Length", "Embedding Length", "Quantization"] + all_dataset_names)

def save_benchmark_results(summary_df: pd.DataFrame, csv_file: str):
    """Save benchmark results to a CSV file, merging with existing data."""
    existing_df = load_existing_results(csv_file)
    all_dataset_names = list(DATASETS.keys())
    expected_columns = ["Model", "Organization", "Architecture", "Parameters",
                        "Context Length", "Embedding Length", "Quantization"] + all_dataset_names
    for col in expected_columns:
        if col not in summary_df.columns:
            summary_df[col] = "-"
    summary_df = summary_df[expected_columns]
    for _, row in summary_df.iterrows():
        model = row["Model"]
        provider = row["Organization"]
        mask = (existing_df["Model"] == model) & (existing_df["Organization"] == provider)
        if mask.any():
            for dataset in all_dataset_names:
                if dataset in row and not pd.isna(row[dataset]) and row[dataset] != "-":
                    existing_df.loc[mask, dataset] = row[dataset]
            for col in ["Architecture", "Parameters", "Context Length", "Embedding Length", "Quantization"]:
                if not pd.isna(row[col]) and row[col] != "-":
                    existing_df.loc[mask, col] = row[col]
        else:
            existing_df = pd.concat([existing_df, row.to_frame().T], ignore_index=True)
    for col in expected_columns:
        if col not in existing_df.columns:
            existing_df[col] = "-"
    existing_df = existing_df[expected_columns]
    existing_df.to_csv(csv_file, index=False)
    return existing_df