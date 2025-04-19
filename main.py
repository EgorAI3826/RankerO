import gradio as gr
import pandas as pd
import queue
from threading import Thread, Event
from RO import DATASETS, ModelEvaluator, load_dataset, generate_html_table, save_benchmark_results, load_existing_results
from api_ollama import get_model_info as get_ollama_model_info
from openai_api import get_model_info as get_openai_model_info
from groq_api import get_model_info as get_groq_model_info
from openrouter_api import get_model_info as get_openrouter_model_info
from datetime import datetime
import os

# Determine project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Predefined models
MODELS = [
    ("infidelis/GigaChat-20B-A3B-instruct-v1.5:q4_K_M", "ollama"),
    ("gpt-4o-mini", "openai"),
    ("llama3-8b-8192", "groq"),
    ("qwen/qwen-plus", "openrouter")
]

# Separate datasets into BASE and MERA categories
BASE_DATASETS = [name for name, info in DATASETS.items() if info.get("category") == "BASE"]
MERA_DATASETS = [name for name, info in DATASETS.items() if info.get("category") == "MERA"]

def get_model_info(model_name: str, provider: str) -> dict:
    """Retrieve model information based on provider."""
    providers = {
        "ollama": get_ollama_model_info,
        "openai": get_openai_model_info,
        "groq": get_groq_model_info,
        "openrouter": get_openrouter_model_info
    }
    return providers.get(provider, lambda x: {
        "architecture": "unknown",
        "parameters": "unknown",
        "context_length": 4096,
        "embedding_length": 5120,
        "quantization": "unknown"
    })(model_name)

def run_evaluation(base_datasets, mera_datasets, model_selections, ollama_model, openai_model, groq_model, openrouter_model, progress=gr.Progress()):
    """Run model evaluation on selected datasets and models."""
    dataset_names = (base_datasets or []) + (mera_datasets or [])
    if not dataset_names:
        yield "Error: Please select at least one dataset.", None, None
        return
    selected_models = []
    for model_selection in model_selections:
        parts = model_selection.split(" (")
        if len(parts) == 2:
            model_name = parts[0]
            provider = parts[1].replace(")", "")
            selected_models.append((model_name, provider))
    for model_input, provider in [
        (ollama_model, "ollama"), (openai_model, "openai"),
        (groq_model, "groq"), (openrouter_model, "openrouter")
    ]:
        if model_input:
            selected_models.append((model_input, provider))
    if not selected_models:
        yield "Error: Please select at least one model.", None, None
        return
    yield f"Starting evaluation for {len(selected_models)} models on {len(dataset_names)} datasets...", None, None
    evaluator = ModelEvaluator(embedding_model="bge-m3:567m-fp16")
    summary_data = {}
    for model_name, provider in selected_models:
        model_info = get_model_info(model_name, provider)
        summary_data[(model_name, provider)] = {
            "Model": model_name,
            "Organization": provider,
            "Architecture": model_info.get("architecture", "-"),
            "Parameters": model_info.get("parameters", "-"),
            "Context Length": model_info.get("context_length", "-"),
            "Embedding Length": model_info.get("embedding_length", "-"),
            "Quantization": model_info.get("quantization", "-"),
            **{dataset_name: "-" for dataset_name in DATASETS.keys()}
        }
    global_results = []
    all_evaluation_results = []
    update_queue = queue.Queue()
    update_event = Event()
    selected_datasets = {name: DATASETS[name] for name in dataset_names if name in DATASETS}
    for dataset_full_name, dataset_info in selected_datasets.items():
        dataset = load_dataset(dataset_info)
        threshold = dataset_info.get("threshold", 0.85)
        if not dataset:
            yield f"Error: Could not load dataset {dataset_full_name}.", None, None
            continue
        for model_name, provider in selected_models:
            status = f"Evaluating {model_name} on {dataset_full_name} with threshold {threshold}..."
            yield status, None, None
            def evaluation_thread():
                try:
                    eval_results = evaluator.evaluate_model(
                        dataset, model_name, provider, dataset_full_name, threshold, update_queue, update_event
                    )
                    result_entry = {
                        "model": model_name,
                        "provider": provider,
                        "dataset": dataset_full_name,
                        "total_records": eval_results["total"],
                        "scores": eval_results["scores"],
                        "accuracy": eval_results["accuracy"]
                    }
                    global_results.append(result_entry)
                    all_evaluation_results.append(result_entry)
                    summary_data[(model_name, provider)][dataset_full_name] = f"{eval_results['accuracy']:.2f}"
                    columns = ["Model", "Organization", "Architecture", "Parameters", "Context Length",
                               "Embedding Length", "Quantization"] + list(DATASETS.keys())
                    summary_df = pd.DataFrame(list(summary_data.values()), columns=columns)
                    final_df = save_benchmark_results(summary_df, os.path.join(DATA_DIR, "data.csv"))
                    update_queue.put(("complete", None, None, len(dataset), final_df))
                    update_event.set()
                except Exception as e:
                    update_queue.put(("error", f"Error during evaluation: {str(e)}", None, 0, None))
                    update_event.set()
            Thread(target=evaluation_thread).start()
            detailed_df = pd.DataFrame(columns=["Question", "Predicted Answer", "Expected Answer", "Embedding", "Correct"])
            status = "Starting evaluation..."
            final_df = load_existing_results(os.path.join(DATA_DIR, "data.csv"))
            while True:
                update_event.wait(timeout=0.1)
                try:
                    update_type, status_text, new_df, progress_index, updated_final_df = update_queue.get_nowait()
                    if update_type == "progress":
                        status = status_text
                        if new_df is not None:
                            detailed_df = new_df
                        progress(progress_index / len(dataset), desc=f"Processing {progress_index}/{len(dataset)}")
                        yield status, detailed_df, final_df
                    elif update_type == "final":
                        status = status_text
                        if new_df is not None:
                            detailed_df = new_df
                        progress(1.0, desc="Evaluation completed")
                        yield status, detailed_df, final_df
                    elif update_type == "error":
                        yield status_text, None, final_df
                        break
                    elif update_type == "complete":
                        if updated_final_df is not None:
                            final_df = updated_final_df
                        break
                except queue.Empty:
                    update_event.clear()
                    yield status, detailed_df, final_df
                    continue
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_table = generate_html_table(all_evaluation_results)
    with open(os.path.join(DATA_DIR, f"evaluation_results_{timestamp}.html"), "w", encoding="utf-8") as f:
        f.write(html_table)
    yield f"Evaluation completed. Results saved to evaluation_results_{timestamp}.html", detailed_df, final_df

def clear_selections():
    """Reset all input selections."""
    return [], [], [], "", "", "", ""

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    initial_summary_df = load_existing_results(os.path.join(DATA_DIR, "data.csv"))
    with gr.Blocks(title="RankerO - Model Evaluation Dashboard", css=".gradio-container {font-family: Arial;}") as demo:
        gr.Markdown(
            """
            # RankerO: Model Evaluation Dashboard
            Evaluate and compare AI models across various datasets. Select datasets and models, then start the evaluation to see detailed and summary results.
            """
        )
        with gr.Tabs():
            with gr.Tab("Dataset Selection"):
                with gr.Accordion("BASE Datasets", open=True):
                    gr.Markdown("General-purpose datasets for broad evaluation (e.g., math, programming, language).")
                    base_dataset_dropdown = gr.CheckboxGroup(
                        label="Select BASE Datasets",
                        choices=BASE_DATASETS,
                        value=BASE_DATASETS,
                        interactive=True
                    )
                with gr.Accordion("MERA Datasets", open=False):
                    gr.Markdown("Russian-specific datasets for language, ethics, and programming tasks.")
                    mera_dataset_dropdown = gr.CheckboxGroup(
                        label="Select MERA Datasets",
                        choices=MERA_DATASETS,
                        value=[],
                        interactive=True
                    )
            with gr.Tab("Model Selection"):
                gr.Markdown("Choose predefined or custom models from various providers.")
                model_dropdown = gr.CheckboxGroup(
                    label="Predefined Models",
                    choices=[f"{model[0]} ({model[1]})" for model in MODELS],
                    value=[f"{MODELS[0][0]} ({MODELS[0][1]})"]
                )
                with gr.Row():
                    ollama_model_input = gr.Textbox(
                        label="Custom Ollama Model",
                        placeholder="e.g., infidelis/GigaChat-20B-A3B-instruct-v1.5:q4_K_M",
                        value=""
                    )
                    openai_model_input = gr.Textbox(
                        label="Custom OpenAI Model",
                        placeholder="e.g., gpt-4o-mini",
                        value=""
                    )
                with gr.Row():
                    groq_model_input = gr.Textbox(
                        label="Custom Groq Model",
                        placeholder="e.g., llama3-8b-8192",
                        value=""
                    )
                    openrouter_model_input = gr.Textbox(
                        label="Custom OpenRouter Model",
                        placeholder="e.g., qwen/qwen-plus",
                        value=""
                    )
        with gr.Row():
            start_button = gr.Button("Start Evaluation", variant="primary")
            clear_button = gr.Button("Clear Selection")
        status_text = gr.Textbox(label="Evaluation Status", value="Ready to start evaluation...", interactive=False)
        with gr.Tabs():
            with gr.Tab("Detailed Results"):
                detailed_table = gr.Dataframe(
                    label="Detailed Results (Max 100 Records)",
                    headers=["Question", "Predicted Answer", "Expected Answer", "Embedding", "Correct"],
                    interactive=False
                )
            with gr.Tab("Summary Statistics"):
                summary_table = gr.Dataframe(
                    label="Summary Statistics",
                    headers=["Model", "Organization", "Architecture", "Parameters",
                             "Context Length", "Embedding Length", "Quantization"] + list(DATASETS.keys()),
                    value=initial_summary_df,
                    interactive=False
                )
        download_button = gr.File(label="Download Detailed Results", visible=False)
        start_button.click(
            fn=run_evaluation,
            inputs=[base_dataset_dropdown, mera_dataset_dropdown, model_dropdown, ollama_model_input, openai_model_input, groq_model_input, openrouter_model_input],
            outputs=[status_text, detailed_table, summary_table]
        )
        clear_button.click(
            fn=clear_selections,
            inputs=None,
            outputs=[base_dataset_dropdown, mera_dataset_dropdown, model_dropdown, ollama_model_input, openai_model_input, groq_model_input, openrouter_model_input]
        )
    demo.launch()

if __name__ == "__main__":
    main()