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

Follow these steps to set up RankerO on your system.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EgorAI3826/RankerO.git
   cd RankerO
Create a Virtual Environment:
To avoid conflicts with system-wide packages, create and activate a virtual environment:
bash
python -m venv venv
# On Unix/Linux/MacOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
Install Dependencies:
The repository includes a requirements.txt file with all required libraries. Install them using:
bash
pip install -r requirements.txt
Dependencies:
gradio==4.44.0: Interactive dashboard
pandas==2.2.3: Data handling
numpy==1.26.4: Numerical computations
scikit-learn==1.5.2: Cosine similarity calculations
ollama==0.3.3: Local model inference and embeddings
openai==1.51.0: OpenAI API access
groq==0.11.0: Groq API access
Install Ollama:
Download and install Ollama from ollama.com.
Start the Ollama server in a terminal:
bash
ollama serve
Pull required models for evaluation and embeddings:
bash
ollama pull llama3.1:8b
ollama pull bge-m3:567m-fp16
Verify models are available:
bash
ollama list
Set Up API Keys (Optional):
If using non-Ollama providers (OpenAI, Groq, OpenRouter), add your API keys by editing the respective Python files in your project directory (e.g., openrouter_api.py, groq_api.py, openai_api.py, api_ollama.py). For example:
Locate the files in your project folder (paths may vary, e.g., C:\Users\User\Desktop\Rank\).
Open each file in a text editor and insert your API key as specified in the file (e.g., replace a placeholder like YOUR_API_KEY with your actual key).
Example for openai_api.py:
python
API_KEY = "your_openai_key_here"
Note: Skip this step if only using Ollama models, as Ollama does not require API keys for local inference.
Download Datasets (Optional):
Datasets are downloaded automatically during evaluation. To fetch them in advance:
bash
python down.py
Usage
Run evaluations and view results using the Gradio web interface.
Launch the Evaluation Dashboard:
Ensure the Ollama server is running (ollama serve in a separate terminal).
Start the Gradio interface:
bash
python main.py
Open the web interface at http://localhost:7860.
Configure Evaluation:
Datasets:
Select BASE and/or MERA datasets via checkboxes.
Use "Select All" or "Deselect All" buttons for quick selection.
Models:
Choose predefined models (e.g., llama3.1:8b (ollama), gpt-4o-mini (openai)).
Add custom models in the "Custom Models" section by specifying provider and model name (e.g., llama3.1:70b for Ollama).
Verify custom models are available (e.g., run ollama list for Ollama models).
Embedding Model:
Select an embedding model (default: bge-m3:567m-fp16) or enter a custom one in the "Embedding Model" tab.
Thresholds:
Adjust evaluation thresholds if needed (defaults are dataset-specific).
Run Evaluation:
Click "Start Evaluation" to test selected models on chosen datasets.
Monitor progress in the "Evaluation Status" textbox.
View results in:
Detailed Results: Question-by-question analysis with predicted answers, expected answers, embedding similarity, and correctness.
Summary Statistics: Overall accuracy and category-specific scores (reasoning, coding, math, etc.).
Interpret Results:
Results are saved in the data directory:
HTML report: evaluation_results_YYYYMMDD_HHMMSS.html
CSV summary: data.csv
Use the Gradio interface to explore detailed metrics and compare model performance.

### Changes Made
1. **API Key Setup**:
   - Replaced the `.env` file instructions with guidance to edit the Python files (`openrouter_api.py`, `groq_api.py`, `openai_api.py`, `api_ollama.py`).
   - Clarified that file paths may vary (e.g., `C:\Users\User\Desktop\Rank\` or elsewhere) and provided a generic instruction to locate them in the project directory.
   - Included an example of editing `openai_api.py` to show how to insert an API key.
   - Noted that API keys are optional for Ollama users.

2. **Maintained Aesthetic**:
   - Kept the clean, structured format with numbered steps, bold headings, and bullet points.
   - Used code blocks for commands and Python snippets to ensure clarity.
   - Retained the Markdown note for optional steps to avoid confusion.

3. **Improved Clarity**:
   - Emphasized that users should check the specific Python files for API key placeholders.
   - Added a note about verifying file locations, accommodating variable project directories.
   - Kept instructions concise while ensuring all necessary steps are covered.

### Notes
- **File Paths**: The instruction avoids hardcoding paths like `C:\Users\User\Desktop\Rank\` and uses generic terms like "project directory" to be universally applicable. Users are guided to locate the files themselves.
- **API Key Implementation**: Assumed the Python files have a clear placeholder (e.g., `API_KEY = "your_key_here"`) for users to replace. If the files require a different format (e.g., a configuration dictionary or environment variable fallback), please share their structure for more precise instructions.
- **Other Sections**: Only "Installation" and "Usage" are updated, as requested. If you want the "Configuring Models" or "Troubleshooting" sections to reflect the API key change (e.g., updating troubleshooting to mention editing Python files), I can provide those.
- **Dependencies**: The dependency list remains unchanged, as no new libraries were introduced. Ensure `requirements.txt` is in the repository root.
- **Ollama**: The instructions still emphasize running `ollama serve` and pulling models, as these are critical for most users.

### Next Steps
1. **Integrate into README**:
   - Replace the "Installation" and "Usage" sections in your full `README.md` with the updated text above.
   - Ensure `requirements.txt`, `main.py`, `down.py`, and the API Python files (`openrouter_api.py`, etc.) are in the repository.

2. **Verify Setup**:
   - Test the instructions on a fresh environment:
     - Clone the repo, create a virtual environment, install dependencies.
     - Edit the API Python files with dummy keys (or skip for Ollama).
     - Run `ollama serve`, pull models, and launch `main.py`.
   - Confirm the Gradio interface loads and evaluations run without API key errors.

3. **Update Repository**:
   - Commit the updated `README.md` and any modified files:
     ```bash
     git add README.md requirements.txt
     git commit -m "Update README with API key editing instructions"
     git push
     ```

4. **User Feedback**:
   - Share the updated README with users (e.g., Филипп) to confirm it addresses setup issues.
   - If users report difficulties editing the Python files (e.g., unclear placeholders), provide the file contents or update the instructions.

If you need further refinements (e.g., specific formatting, adding screenshots, or updating other README sections), or if you can share the contents of `openrouter_api.py`, `groq_api.py`, etc., to tailor the API key instructions, let me know! I can also assist with troubleshooting or enhancing the Gradio UI if additional issues arise.

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
