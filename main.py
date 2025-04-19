import gradio as gr
import pandas as pd
import queue
from threading import Thread, Event
from RO import DATASETS, ModelEvaluator, load_dataset, generate_html_table, save_benchmark_results, load_existing_results
from api_ollama import get_model_info as get_ollama_model_info
from openai_api import get_model_info as get_openai_model_info
from groq_api import get_model_info as get_groq_model_info
from openrouter_api import get_model_info as get_openrouter_model_info
from lm_studio_api import get_model_info as get_lm_studio_model_info
from datetime import datetime
import os

# Determine project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Predefined models
MODELS = [
    ("llama3.1:8b", "ollama"),
    ("gpt-4o-mini", "openai"),
    ("llama3-8b-8192", "groq"),
    ("qwen/qwen-plus", "openrouter"),
    ("mistral-7b-instruct", "lm_studio")
]

# Separate datasets into BASE and MERA categories
BASE_DATASETS = [name for name, info in DATASETS.items() if info.get("category") == "BASE"]
MERA_DATASETS = [name for name, info in DATASETS.items() if info.get("category") == "MERA"]

# Available embedding models
EMBEDDING_MODELS = [
    "bge-m3:567m-fp16",
    "all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "other (specify)"
]

def get_model_info(model_name: str, provider: str) -> dict:
    """Retrieve model information based on provider."""
    providers = {
        "ollama": get_ollama_model_info,
        "openai": get_openai_model_info,
        "groq": get_groq_model_info,
        "openrouter": get_openrouter_model_info,
        "lm_studio": get_lm_studio_model_info
    }
    return providers.get(provider, lambda x: {
        "architecture": "unknown",
        "parameters": "unknown",
        "context_length": 4096,
        "embedding_length": 5120,
        "quantization": "unknown"
    })(model_name)

def select_all_base():
    """Select all BASE datasets."""
    return BASE_DATASETS

def deselect_all_base():
    """Deselect all BASE datasets."""
    return []

def select_all_mera():
    """Select all MERA datasets."""
    return MERA_DATASETS

def deselect_all_mera():
    """Deselect all MERA datasets."""
    return []

def select_all_predefined():
    """Select all predefined models."""
    return [f"{model[0]} ({model[1]})" for model in MODELS]

def deselect_all_predefined():
    """Deselect all predefined models."""
    return []

def add_custom_model(provider, model_name, custom_models):
    """Add a custom model to the list."""
    if not model_name or not provider:
        return custom_models, "Please specify both provider and model name."
    model_entry = (model_name.strip(), provider)
    if model_entry not in custom_models:
        custom_models.append(model_entry)
        return custom_models, f"Added {model_name} ({provider})."
    return custom_models, f"{model_name} ({provider}) already added."

def remove_custom_model(models_to_remove, custom_models):
    """Remove selected custom models from the list."""
    updated_models = [m for m in custom_models if f"{m[0]} ({m[1]})" not in models_to_remove]
    return updated_models, "Removed selected models."

def run_evaluation(base_datasets, mera_datasets, model_selections, custom_models, embedding_model, custom_embedding, progress=gr.Progress()):
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
    selected_models.extend(custom_models)
    if not selected_models:
        yield "Error: Please select at least one model.", None, None
        return
    # Handle embedding model selection
    emb_model = embedding_model
    if embedding_model == "other (specify)" and custom_embedding:
        emb_model = custom_embedding.strip()
    if not emb_model:
        emb_model = "bge-m3:567m-fp16"  # Default
    yield f"Starting evaluation for {len(selected_models)} models on {len(dataset_names)} datasets with embedding model {emb_model}...", None, None
    evaluator = ModelEvaluator(embedding_model=emb_model)
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
    return [], [], [], [], "bge-m3:567m-fp16", ""

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    initial_summary_df = load_existing_results(os.path.join(DATA_DIR, "data.csv"))
    with gr.Blocks(title="RankerO - Model Evaluation Dashboard", css=".gradio-container {font-family: Arial;}") as demo:
        gr.Markdown(
            """
            # RankerO: Model Evaluation Dashboard
            Evaluate and compare AI models across various datasets. Select datasets, models, and embedding model, then start the evaluation to see detailed and summary results.
            """
        )
        with gr.Tabs():
            with gr.Tab("Dataset Selection"):
                with gr.Accordion("BASE Datasets", open=True):
                    gr.Markdown("General-purpose datasets for broad evaluation (e.g., math, programming, language).")
                    with gr.Row():
                        base_select_all = gr.Button("Select All")
                        base_deselect_all = gr.Button("Deselect All")
                    base_dataset_dropdown = gr.CheckboxGroup(
                        label="Select BASE Datasets",
                        choices=BASE_DATASETS,
                        value=BASE_DATASETS,
                        interactive=True
                    )
                with gr.Accordion("MERA Datasets", open=False):
                    gr.Markdown("Russian-specific datasets for language, ethics, and programming tasks.")
                    with gr.Row():
                        mera_select_all = gr.Button("Select All")
                        mera_deselect_all = gr.Button("Deselect All")
                    mera_dataset_dropdown = gr.CheckboxGroup(
                        label="Select MERA Datasets",
                        choices=MERA_DATASETS,
                        value=[],
                        interactive=True
                    )
            with gr.Tab("Model Selection"):
                gr.Markdown("Choose predefined or custom models from various providers.")
                with gr.Accordion("Predefined Models", open=True):
                    with gr.Row():
                        model_select_all = gr.Button("Select All")
                        model_deselect_all = gr.Button("Deselect All")
                    model_dropdown = gr.CheckboxGroup(
                        label="Predefined Models",
                        choices=[f"{model[0]} ({model[1]})" for model in MODELS],
                        value=[f"{MODELS[0][0]} ({MODELS[0][1]})"]
                    )
                with gr.Accordion("Custom Models", open=False):
                    gr.Markdown("Add custom models by specifying provider and model name.")
                    provider_dropdown = gr.Dropdown(
                        label="Provider",
                        choices=["ollama", "openai", "groq", "openrouter", "lm_studio"],
                        value="ollama"
                    )
                    custom_model_input = gr.Textbox(
                        label="Custom Model Name",
                        placeholder="e.g., infidelis/GigaChat-20B-A3B-instruct-v1.5:q4_K_M",
                        value=""
                    )
                    add_model_button = gr.Button("Add Custom Model")
                    custom_model_status = gr.Textbox(label="Status", interactive=False)
                    custom_models_state = gr.State(value=[])
                    custom_models_dropdown = gr.CheckboxGroup(
                        label="Custom Models",
                        choices=[],
                        interactive=True
                    )
                    remove_model_button = gr.Button("Remove Selected Custom Models")
            with gr.Tab("Embedding Model"):
                gr.Markdown("Select the embedding model for answer comparison.")
                embedding_model_dropdown = gr.Dropdown(
                    label="Embedding Model",
                    choices=EMBEDDING_MODELS,
                    value="bge-m3:567m-fp16"
                )
                custom_embedding_input = gr.Textbox(
                    label="Custom Embedding Model (if 'other' selected)",
                    placeholder="e.g., my-custom-embedding-model",
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
                             "Context Length", "Embedding Length", "Quantitation"] + list(DATASETS.keys()),
                    value=initial_summary_df,
                    interactive=False
                )
        download_button = gr.File(label="Download Detailed Results", visible=False)
        # Button bindings
        base_select_all.click(fn=select_all_base, outputs=base_dataset_dropdown)
        base_deselect_all.click(fn=deselect_all_base, outputs=base_dataset_dropdown)
        mera_select_all.click(fn=select_all_mera, outputs=mera_dataset_dropdown)
        mera_deselect_all.click(fn=deselect_all_mera, outputs=mera_dataset_dropdown)
        model_select_all.click(fn=select_all_predefined, outputs=model_dropdown)
        model_deselect_all.click(fn=deselect_all_predefined, outputs=model_dropdown)
        add_model_button.click(
            fn=add_custom_model,
            inputs=[provider_dropdown, custom_model_input, custom_models_state],
            outputs=[custom_models_state, custom_model_status]
        )
        custom_models_state.change(
            fn=lambda x: [f"{m[0]} ({m[1]})" for m in x],
            inputs=custom_models_state,
            outputs=custom_models_dropdown
        )
        remove_model_button.click(
            fn=remove_custom_model,
            inputs=[custom_models_dropdown, custom_models_state],
            outputs=[custom_models_state, custom_model_status]
        )
        start_button.click(
            fn=run_evaluation,
            inputs=[
                base_dataset_dropdown, mera_dataset_dropdown, model_dropdown,
                custom_models_state, embedding_model_dropdown, custom_embedding_input
            ],
            outputs=[status_text, detailed_table, summary_table]
        )
        clear_button.click(
            fn=clear_selections,
            inputs=None,
            outputs=[
                base_dataset_dropdown, mera_dataset_dropdown, model_dropdown,
                custom_models_state, embedding_model_dropdown, custom_embedding_input
            ]
        )
    demo.launch()

if __name__ == "__main__":
    main()