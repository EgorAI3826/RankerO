# RankerO: AI Model Evaluation Framework

![RankerO Logo](https://via.placeholder.com/150) *(Placeholder for logo)*

## Overview

RankerO is a comprehensive AI model evaluation framework designed to assess and compare the performance of various language models across multiple datasets. This tool provides a standardized way to measure model accuracy, reasoning capabilities, and domain-specific knowledge through automated testing and benchmarking.

## Key Features

- **Multi-Dataset Evaluation**: Test models against 20+ datasets covering mathematics, programming, reasoning, language understanding, and more
- **Multi-Provider Support**: Evaluate models from Ollama, OpenAI, Groq, and OpenRouter
- **Detailed Metrics**: Get accuracy scores by category (reasoning, coding, math, etc.)
- **Embedding-Based Comparison**: Uses cosine similarity of embeddings for nuanced answer evaluation
- **Interactive Dashboard**: Gradio-based UI for easy configuration and result visualization
- **Benchmark Tracking**: Save and compare results across different model versions

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EgorAI3826/RankerO.git
cd RankerO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (create a `.env` file):
```bash
echo "OPENAI_API_KEY=your_key_here" >> .env
echo "GROQ_API_KEY=your_key_here" >> .env
echo "OPENROUTER_API_KEY=your_key_here" >> .env
```

## Usage

1. Download datasets (optional - will download automatically when needed):
```bash
python down.py
```

2. Launch the evaluation dashboard:
```bash
python main.py
```

3. Access the web interface at `http://localhost:7860`

## Configuration

Customize your evaluation by:
- Selecting specific datasets
- Choosing from predefined models or entering custom model names
- Adjusting evaluation thresholds

## Results Interpretation

Results include:
- Overall accuracy score
- Category-specific performance breakdown
- Detailed question-by-question comparisons
- Embedding similarity scores for answer matching

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any:
- Bug fixes
- Additional dataset integrations
- New model provider support
- Feature enhancements

## License

MIT License

## Contact

For questions or support, contact EgorAI3826 via GitHub.

---

*RankerO - Empowering AI model evaluation and comparison*
