markdown
# RankerO

**Author**: EgorAI3826  
**License**: MIT  
**Version**: 1.0.0  

RankerO is a comprehensive model evaluation dashboard designed to benchmark and compare the performance of various AI models across a diverse set of datasets. It supports multiple API providers (Ollama, OpenAI, Groq, OpenRouter) and evaluates models on tasks spanning mathematics, programming, language, reasoning, ethics, and more. The dashboard provides detailed and summary statistics, with results visualized in tables and saved as CSV and HTML files for further analysis.

## Features

- **Multi-Provider Support**: Evaluate models from Ollama, OpenAI, Groq, and OpenRouter.
- **Diverse Datasets**: Includes BASE datasets (e.g., AIME, GSM8K, HumanEval, MMLU) and MERA datasets (Russian-specific tasks like RuMMLU, RuHumanEval, RuEthics).
- **Flexible Evaluation**: Compare model performance using exact matching, numerical comparison, or cosine similarity of embeddings.
- **Interactive Dashboard**: Built with Gradio, offering a user-friendly interface to select datasets and models, monitor evaluation progress, and view results.
- **Result Persistence**: Saves detailed results as CSV files and generates HTML tables for easy sharing.
- **Customizable Models**: Supports predefined and custom model inputs for each provider.
- **Progress Tracking**: Real-time updates on evaluation status and progress.

## Installation

### Prerequisites

- Python 3.8+
- Git
- Required Python packages (listed in `requirements.txt`)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EgorAI3826/RankerO.git
   cd RankerO
Install Dependencies:
bash
pip install -r requirements.txt
Set Up Environment Variables:
Create a .env file in the project root and add the following API keys:
plaintext
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
Download Datasets:
Run the down.py script to download JSON datasets from the Hugging Face repository:
bash
python down.py
Run the Dashboard:
Launch the Gradio interface:
bash
python main.py
This will start a local web server, and you can access the dashboard in your browser (typically at http://localhost:7860).
Usage
Select Datasets:
Choose from BASE datasets (e.g., AIME, GSM8K, HumanEval) or MERA datasets (e.g., RuMMLU, RuEthics) in the "Dataset Selection" tab.
BASE datasets focus on general tasks, while MERA datasets are tailored for Russian-specific tasks.
Select Models:
Pick predefined models (e.g., infidelis/GigaChat-20B-A3B-instruct-v1.5:q4_K_M, gpt-4o-mini) or enter custom model names for each provider in the "Model Selection" tab.
Run Evaluation:
Click "Start Evaluation" to begin benchmarking.
Monitor progress in the "Evaluation Status" textbox.
View real-time detailed results (up to 100 records) and summary statistics in the respective tabs.
Review Results:
Detailed results are saved as CSV files in the data directory (e.g., <model>_<dataset>_detailed.csv).
Summary results are appended to data/data.csv.
An HTML table of evaluation results is generated in the data directory (e.g., evaluation_results_<timestamp>.html).
Clear Selections:
Use the "Clear Selection" button to reset dataset and model choices.
Project Structure
RankerO/
├── data/                    # Directory for storing evaluation results
├── dataset/                 # Directory for downloaded JSON datasets
├── api_ollama.py            # Ollama API integration
├── groq_api.py              # Groq API integration
├── openai_api.py            # OpenAI API integration
├── openrouter_api.py        # OpenRouter API integration
├── down.py                  # Script to download datasets
├── RO.py                    # Core evaluation logic and dataset definitions
├── main.py                  # Gradio dashboard and evaluation runner
├── requirements.txt         # Python dependencies
└── README.md                # This file
Datasets
BASE Datasets
AIME 1983-2024 (Mathematics): Math problems from the American Invitational Mathematics Examination.
GSM8K (Mathematics): Grade-school math problems.
HumanEval (Programming): Programming tasks for code generation.
MMLU Pro (General Knowledge): Multiple-choice questions across professional fields.
Thinking Traps (Logic): Logical reasoning tasks.
And more (see RO.py for the full list).
MERA Datasets
RuMMLU (General Knowledge): Russian version of MMLU.
RuHumanEval (Programming): Russian programming tasks.
RuEthics (Ethics): Ethical reasoning tasks in Russian.
RuHateSpeech (Moderation): Hate speech detection tasks.
And more (see RO.py for the full list).
Evaluation Metrics
Exact Match: For text answers, checks if the predicted answer matches the expected answer after cleaning (e.g., removing LaTeX).
Numerical Comparison: For numerical or fractional answers, compares values with a small tolerance.
Embedding Similarity: Uses cosine similarity of embeddings (via bge-m3:567m-fp16) with dataset-specific thresholds (e.g., 0.85 or 0.90).
Category Scores: Reports accuracy for categories like reasoning, coding, math, language, data analysis, and instruction following.
Contributing
Contributions are welcome! Please follow these steps:
Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.
Issues
If you encounter any bugs or have feature requests, please open an issue on the GitHub Issues page.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
Built with Gradio for the interactive dashboard.
Datasets sourced from Hugging Face and processed for evaluation.
Thanks to the open-source community for providing tools and libraries used in this project.
Contact
For questions or feedback, reach out to EgorAI3826 via GitHub or email (if provided).
Happy benchmarking with RankerO! 🚀

This README provides a clear overview of the RankerO project, including installation instructions, usage guide, project structure, dataset details, evaluation metrics, and contribution guidelines. It is formatted in Markdown for GitHub compatibility and includes all necessary information for users and contributors.
