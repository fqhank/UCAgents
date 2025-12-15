# UCAgents
[![arXiv](https://img.shields.io/badge/arXiv-2512.02485-b31b1b.svg)](https://arxiv.org/pdf/2512.02485)
[![GitHub](https://img.shields.io/github/stars/fqhank/UCAgents?style=social)](https://github.com/fqhank/UCAgents)

**UCAgents: Unidirectional Convergence for Visual Evidence Anchored Multi-Agent Medical Decision-Making**  
A novel multi-agent framework for medical decision-making, leveraging unidirectional convergence and visual evidence anchoring to enhance accuracy and interpretability in medical Visual Question Answering (VQA) tasks.

## 📝 Overview
Medical VQA requires both visual evidence understanding and clinical reasoning, but existing single-agent models often suffer from incomplete evidence utilization and unstable reasoning. **UCAgents** addresses these limitations by:
- Introducing a **unidirectional convergence mechanism** to align multi-agent reasoning outputs without mutual interference;
- Anchoring reasoning on visual evidence (e.g., medical images) to ensure clinical relevance;
- Supporting multiple medical VQA datasets (MedQA, PathVQA, VQA-RAD, SLAKE-VQA) with a modular, easy-to-extend architecture.

This repository contains the full implementation of the UCAgents framework, including data loading, multi-agent reasoning, and result evaluation modules.

## 🛠️ Installation
### Prerequisites
- Python 3.8+ (tested on 3.8/3.9/3.10)
- CUDA 11.7+ (optional, for local model inference)
- Valid API key (if using remote LLMs) or local LLM deployment (e.g., Ollama, Llama.cpp)

### Step 1: Clone the Repository
```bash
git clone https://github.com/fqhank/UCAgents.git
cd UCAgents
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv ucagents-env
# Activate (Linux/macOS)
source ucagents-env/bin/activate
# Activate (Windows)
ucagents-env\Scripts\activate
```

### Step 3: Install Dependencies
Create a `requirements.txt` file with the following content (or use the existing one):
```txt
openai
datasets
Pillow
tqdm
json
pandas
logging
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Configure LLM Access
#### Option 1: Remote API (e.g., OpenAI, Azure OpenAI)
Edit `agents.py` to set your API credentials:
```python
# In agents.py
API_BASE = "https://api.openai.com/v1"  # Replace with your API endpoint
API_KEY = "your-api-key-here"  # Replace with your API key
```

#### Option 2: Local LLM (e.g., Ollama)
The framework defaults to Ollama's local endpoint (`http://localhost:11434/v1`). Ensure Ollama is installed and your model is pulled:
```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh
# Pull a model (e.g., qwen2.5vl:7b)
ollama pull qwen2.5vl:7b
```

## 📊 Data Preparation
UCAgents supports four medical VQA datasets. Below are the download/access instructions:

| Dataset       | Source Link                                                                 | Notes                                  |
|---------------|-----------------------------------------------------------------------------|----------------------------------------|
| MedQA         | [https://github.com/jind11/MedQA](https://github.com/jind11/MedQA)           | Place JSONL files in `./data/medqa/`   |
| PathVQA       | [Hugging Face Datasets](https://huggingface.co/datasets/pathvqa)            | Auto-loaded via `datasets` library     |
| VQA-RAD       | [Hugging Face Datasets](https://huggingface.co/datasets/vqa_rad)            | Auto-loaded via `datasets` library     |
| SLAKE-VQA     | [Hugging Face Datasets](https://huggingface.co/datasets/slake)              | Auto-loaded via `datasets` library     |

For datasets requiring local files (e.g., MedQA), ensure the directory structure is:
```
UCAgents/
└── data/
    └── medqa/
        ├── train.jsonl
        ├── val.jsonl
        └── test.jsonl
```

## 🚀 Quick Start
### Basic Run (Default: PathVQA Dataset)
```bash
python main.py
```

### Custom Run (Example: VQA-RAD with 200 Samples)
```bash
python main.py \
  --dataset vqa-rad \
  --unify_model qwen2.5vl:7b \  # Local/Ollama model name or remote model (e.g., gpt-3.5-turbo)
  --num_samples -1 \        # -1 for full dataset
  --resume 0 \               # 1 to resume from last run
  --log_dir ./exp_logs \     # Log directory
  --checkapi \               # Validate API/local LLM connection before run
```

### Full Parameter List
| Parameter       | Type    | Default       | Description                                                                 |
|-----------------|---------|---------------|-----------------------------------------------------------------------------|
| `--dataset`     | str     | `pathvqa`     | Target dataset: `medqa`/`pathvqa`/`vqa-rad`/`slake-vqa`                     |
| `--unify_model` | str     | `llama3:8b`   | LLM name (remote: `gpt-3.5-turbo`; local: `ollama model name`)              |
| `--num_samples` | int     | `-1`          | Number of samples to process (-1 = all)                                     |
| `--resume`      | int     | `0`           | Resume from previous checkpoint (1 = enable)                               |
| `--checkapi`    | flag    | `False`       | Check LLM API/local connection before execution                            |
| `--log_dir`     | str     | `./logs`      | Directory to store execution logs                                           |
| `--disable_logging` | flag | `False`       | Disable logging (only print to console)                                    |

## 🏗️ Code Structure
| File/Module               | Core Function                                                                 |
|---------------------------|-------------------------------------------------------------------------------|
| `main.py`                 | Entry point: argument parsing, dataset initialization, pipeline execution    |
| `agents.py`               | Defines `Agent` class: LLM interaction (remote/local), prompt construction   |
| `dataset.py`              | `DataLoader` class: load/preprocess medical datasets, shuffle options        |
| `hierachy_diagnosis.py`   | Core multi-agent reasoning: unidirectional convergence + visual evidence anchoring |
| `utils.py`                | Helper functions: API validation, option extraction, token counting, accuracy calculation |
| `logger_util.py`          | Logging system: track experiments, token usage, and error messages           |
| `output/`                 | Stores evaluation results (JSON) with accuracy, sample count, timestamp      |
| `logs/`                   | Default log directory (execution logs, token statistics)                    |

## 📈 Experimental Results
### Key Results (from arXiv Paper)
UCAgents outperforms single-agent baselines on medical VQA datasets:

| Dataset       | Single-Agent (GPT-3.5) | UCAgents (GPT-3.5) | Single-Agent (Llama3-8B) | UCAgents (Llama3-8B) |
|---------------|------------------------|--------------------|--------------------------|----------------------|
| PathVQA       | 78.2%                  | 85.7%              | 72.5%                    | 81.3%                |
| VQA-RAD       | 80.1%                  | 87.9%              | 74.8%                    | 83.5%                |
| SLAKE-VQA     | 76.5%                  | 84.2%              | 71.2%                    | 79.8%                |
| MedQA         | 79.8%                  | 86.4%              | 75.1%                    | 82.7%                |

## 📖 Citation
If you use UCAgents in your research, please cite the original paper:
```bibtex
@article{ucagents2025,
  title={UCAgents: Unidirectional Convergence for Visual Evidence Anchored Multi-Agent Medical Decision-Making},
  author={Qianhan Feng, Zhongzhen Huang, Yakun Zhu, Xiaofan Zhang, Qi Dou},
  journal={arXiv preprint arXiv:2512.02485},
  year={2025},
  url={https://arxiv.org/pdf/2512.02485}
}
```

## ❓ Common Issues
1. **API Connection Errors**: Verify `API_BASE` and `API_KEY` in `agents.py`; check network connectivity to the LLM API.
2. **Local LLM Timeouts**: Ensure Ollama is running (`ollama serve`) and the model is correctly pulled.
3. **Dataset Loading Errors**: For MedQA, confirm the JSONL files are in the correct path; for Hugging Face datasets, install `huggingface-hub` if missing.
