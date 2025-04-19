# RankerO: AI Model Evaluation Framework

![RankerO Logo](./logo.png)

## Overview

RankerO is a comprehensive AI model evaluation framework designed to assess and compare the performance of various language models across multiple datasets. This tool provides a standardized way to measure model accuracy, reasoning capabilities, and domain-specific knowledge through automated testing and benchmarking.

## Key Features

- **Multi-Dataset Evaluation**: Test models against 20+ datasets covering mathematics, programming, reasoning, language understanding, and more.
- **Multi-Provider Support**: Evaluate models from Ollama, OpenAI, Groq, and OpenRouter.
- **Detailed Metrics**: Get accuracy scores by category (reasoning, coding, math, etc.).
- **Embedding-Based Comparison**: Uses cosine similarity of embeddings for nuanced answer evaluation.
- **Interactive Dashboard**: Gradio-based UI for easy configuration and result visualization.
- **Benchmark Tracking**: Save and compare results across different model versions.

## Supported Datasets

### BASE Datasets (General-purpose)
- AIME 1983-2024 (Mathematics)
- GSM8K (Mathematics)
- HumanEval (Programming)
- Thinking Traps (Logic)
- MMLU Pro (General Knowledge)
- And 10+ more...

### MERA Datasets (Russian-specific)
- RuCodeEval (Programming)
- RuEthics (Ethics)
- RuHumanEval (Programming)
- RuMMLU (General Knowledge)
- And 10+ more...

## Supported Model Providers

- Ollama (local models)
- OpenAI
- Groq
- OpenRouter

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Ollama**: Required for local model inference and embedding generation.
- **API Keys**: Required for OpenAI, Groq, and OpenRouter (optional if only using Ollama).
- **Git**: For cloning the repository.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EgorAI3826/RankerO.git
   cd RankerO
Create a Virtual Environment (recommended to avoid conflicts with system-wide packages):
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
The repository includes a requirements.txt file listing all required libraries. Install them with:
bash
pip install -r requirements.txt
The dependencies include:
gradio==4.44.0: For the interactive dashboard.
pandas==2.2.3: For data handling.
numpy==1.26.4: For numerical computations.
scikit-learn==1.5.2: For cosine similarity calculations.
ollama==0.3.3: For local model inference and embeddings.
openai==1.51.0: For OpenAI API access.
groq==0.11.0: For Groq API access.
Install Ollama:
Download and install Ollama from ollama.com.
Start the Ollama server:
bash
ollama serve
Pull required models (e.g., llama3.1:8b for evaluation and bge-m3:567m-fp16 for embeddings):
bash
ollama pull llama3.1:8b
ollama pull bge-m3:567m-fp16
Verify models are available:
bash
ollama list
Set Up API Keys:
Create a .env file in the project root to store API keys for non-Ollama providers:
bash
echo "OPENAI_API_KEY=your_key_here" >> .env
echo "GROQ_API_KEY=your_key_here" >> .env
echo "OPENROUTER_API_KEY=your_key_here" >> .env
If only using Ollama, this step is optional.
Download Datasets (optional):
Datasets are downloaded automatically when needed, but you can run down.py to fetch them in advance:
bash
python down.py
Usage
Launch the Evaluation Dashboard:
Ensure the Ollama server is running (ollama serve in a separate terminal).
Start the Gradio interface:
bash
python main.py
Access the web interface at http://localhost:7860.
Configure Evaluation:
Datasets: Select BASE and/or MERA datasets using the checkboxes. Use "Select All" or "Deselect All" for convenience.
Models:
Choose predefined models (e.g., llama3.1:8b (ollama), gpt-4o-mini (openai)).
Add custom models via the "Custom Models" section by specifying the provider and model name (e.g., llama3.1:8b for Ollama).
Example: To use a different Ollama model, add llama3.1:70b or any model listed in ollama list.
Embedding Model: Select an embedding model (default: bge-m3:567m-fp16) or specify a custom one in the "Embedding Model" tab.
Adjust evaluation thresholds if needed (default thresholds are dataset-specific).
Run Evaluation:
Click "Start Evaluation" to test selected models on chosen datasets.
Monitor progress in the "Evaluation Status" textbox.
View results in the "Detailed Results" and "Summary Statistics" tabs.
Interpret Results:
Detailed Results: Question-by-question breakdown with predicted answers, expected answers, embedding similarity, and correctness.
Summary Statistics: Overall accuracy and category-specific scores (reasoning, coding, math, etc.).
Results are saved as HTML (evaluation_results_YYYYMMDD_HHMMSS.html) and CSV (data.csv) in the data directory.
Configuring Models
Predefined Models: Defined in main.py under MODELS. Example:
python
MODELS = [
    ("llama3.1:8b", "ollama"),
    ("gpt-4o-mini", "openai"),
    ("llama3-8b-8192", "groq"),
    ("qwen/qwen-plus", "openrouter")
]
To use a different Ollama model (e.g., llama3.1:70b), either:
Add it to the Gradio UI under "Custom Models".
Edit MODELS in main.py (e.g., replace ("llama3.1:8b", "ollama") with ("llama3.1:70b", "ollama")).
Custom Models: Use the Gradio UI to add models without editing code. Ensure the model is available:
For Ollama: Run ollama pull <model_name> (e.g., ollama pull llama3.1:70b).
For other providers: Verify the model name matches the provider’s API (e.g., gpt-4o for OpenAI).
Troubleshooting
Ollama Server Not Running:
Ensure ollama serve is active before running main.py.
Check available models with ollama list.
Missing Embedding Model:
Pull the default embedding model: ollama pull bge-m3:567m-fp16.
Select an alternative in the Gradio UI if needed.
API Key Errors:
Verify .env file contains valid keys for OpenAI, Groq, or OpenRouter.
Use only Ollama models to bypass API key requirements.
Dependency Issues:
Ensure requirements.txt dependencies are installed in the virtual environment.
Run pip install -r requirements.txt again if errors occur.
Dataset Not Found:
Run python down.py to download datasets or check the dataset directory.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for:
Bug fixes
Additional dataset integrations
New model provider support
Feature enhancements
License
MIT License
Contact
For questions or support, contact EgorAI3826 via GitHub.
RankerO - Empowering AI model evaluation and comparison

### Changes Made
1. **Virtual Environment**:
   - Added a step to create and activate a virtual environment under "Installation".
   - Included platform-specific activation commands (`source venv/bin/activate` for Unix, `venv\Scripts\activate` for Windows).

2. **Requirements.txt**:
   - Explicitly noted that `requirements.txt` is included in the repository.
   - Listed the dependencies with versions (matching the provided `requirements.txt`) for transparency.

3. **Ollama Setup**:
   - Added detailed instructions to install Ollama, start the server, and pull models (`llama3.1:8b`, `bge-m3:567m-fp16`).
   - Included a command to verify available models (`ollama list`).

4. **Model Configuration**:
   - Clarified how to use predefined models and add custom models via the Gradio UI.
   - Provided instructions to edit `MODELS` in `main.py` for advanced users.
   - Included an example of replacing `llama3.1:8b` with `llama3.1:70b`.

5. **Embedding Model**:
   - Emphasized the need to pull the default embedding model (`bge-m3:567m-fp16`).
   - Noted that users can select alternative embedding models in the Gradio UI.

6. **Dataset Download**:
   - Clarified that `down.py` is optional but recommended to pre-download datasets.
   - Added troubleshooting for dataset issues.

7. **Troubleshooting**:
   - Added a dedicated section addressing common issues:
     - Ollama server not running.
     - Missing embedding model.
     - API key errors.
     - Dependency issues.
     - Dataset not found.

8. **Usage Flow**:
   - Structured the "Usage" section to guide users through launching the dashboard, configuring evaluations, running tests, and interpreting results.
   - Highlighted the Gradio UI’s features (e.g., "Select All" buttons, custom model addition).

### Notes
- **Logo**: The `![RankerO Logo](./logo.png)` assumes `logo.png` exists in the repository root. Ensure this file is present, or update the path if it’s in a subdirectory (e.g., `images/logo.png`).
- **Requirements.txt**: The dependencies listed match the provided `requirements.txt`. If additional libraries are needed (e.g., for `down.py` or custom APIs), update the file and README accordingly.
- **Ollama Models**: The README assumes users can pull any Ollama model (e.g., `llama3.1:8b`). If specific models are required, list them explicitly in the "Prerequisites" or "Installation" sections.
- **API Keys**: The `.env` setup is optional for Ollama-only users, reducing setup complexity for local testing.
- **Troubleshooting**: The troubleshooting section covers issues raised in the conversation (e.g., Ollama server, embedding model). Add more specific cases if users report them.

### Next Steps
1. **Update Repository**:
   - Save the updated `README.md` in the repository root.
   - Ensure `requirements.txt` is present with the listed dependencies.
   - Verify that `logo.png` exists or update the README with the correct path.

2. **Test Instructions**:
   - Follow the README steps on a fresh machine or environment to confirm they work.
   - Run `python -m venv venv`, activate it, install dependencies, set up Ollama, and launch `main.py`.

3. **Share with Users**:
   - Update the GitHub repository (`git add .`, `git commit -m "Update README with setup instructions"`, `git push`).
   - Share the updated README with Филипп or other users to confirm it addresses their setup challenges.

If you need help with specific sections (e.g., implementing `down.py`, adding more troubleshooting tips, or enhancing the Gradio UI), or if there are additional comments from Филипп or others, let me know! I can also assist with generating a `logo.png` if needed or reviewing other project files.
