# Multi-Document Abstractive Summarization Benchmark

This repository presents a rigorous implementation, training, and evaluation benchmark of ten multi-document abstractive summarization models on the CNN/Daily Mail dataset. The project is designed for reproducibility, extensibility, and fair model comparison using standardized evaluation metrics.

---

## Overview

- **Language**: Python 3.9+
- **Frameworks**: PyTorch, Hugging Face Transformers
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L (F1-scores)
- **Dataset**: CNN/Daily Mail (via Hugging Face or local CSVs)
- **Usage**: Research, comparative analysis, educational demonstration

---

## Folder Structure

```
summarization-benchmark/
├── data/                # Raw and preprocessed datasets
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/              # Saved model checkpoints
├── outputs/             # Model-generated summaries
├── results/             # Evaluation results and plots
│   ├── rouge_scores_comparison.png
│   └── training_time_comparison.png
├── scripts/             # Automation scripts
│   └── run_all_models.py
├── requirements.txt     # Project dependencies
├── notebook.ipynb       # Main end-to-end workflow
└── README.md            # Project documentation
```

---

## Models Implemented

| Model | Type | Source/Implementation |
|-------|------|------------------------|
| PRIMERA | Neural (Proxy) | Simulated version |
| BART | Transformer | facebook/bart-large-cnn |
| PEGASUS | Transformer | google/pegasus-cnn_dailymail |
| T5 (Base, Large) | Transformer | Hugging Face Models |
| Longformer/BigBird/LongT5 | Transformer (Proxy) | Simulated version |
| Absformer | Lead-3 (Proxy) | Custom implementation |
| TextRank | Graph-based | NLTK, scikit-learn |
| Hierarchical Transformer | Simulated Two-stage | Custom design |
| DCA | Ensemble (Simulated) | Proxy setup |
| Topic-Guided | Neural + External Knowledge | Custom implementation |

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- PyTorch
- CUDA 11+ enabled GPU (recommended)

### Installation

```bash
git clone https://github.com/YOUR_GITHUB/summarization-benchmark.git
cd summarization-benchmark
pip install -r requirements.txt
```

### Dataset

- Load via Hugging Face (`cnn_dailymail`) or
- Place `train.csv`, `validation.csv`, and `test.csv` inside `./data/`

---

## How to Use

### Option 1: Jupyter Notebook (Recommended)

Run the full pipeline by executing all cells in `notebook.ipynb`:
- Dataset preparation
- Preprocessing
- Model loading and inference
- Evaluation and visualization

### Option 2: Python Script

Automate the full workflow:
```bash
python scripts/run_all_models.py
```

---

## Evaluation Metrics

All models are evaluated using:

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

Scores are computed as F1-measures and averaged over the test set.

---

## Key Hyperparameters

| Parameter           | Value                  |
|---------------------|------------------------|
| Random Seed         | 42                     |
| Batch Size          | 4                      |
| Max Input Length    | 512–4096 (model-dependent) |
| Max Summary Length  | 128 tokens             |
| Beam Size           | 4                      |
| Learning Rate       | 2e-5                   |

---

## Known Issues & Workarounds

- **Unavailable weights**: Models without public weights are simulated using robust approximations (e.g., Lead-3, TextRank).
- **Dependency conflicts**: Resolved by pinning versions in `requirements.txt`.
- **Unstable libraries**: Custom fallback logic used for graph-based modules.

---

## Requirements

```txt
torch==2.0.1
transformers==4.30.2
datasets==2.13.1
evaluate==0.4.0
rouge-score==0.1.2
sentencepiece
nltk
gensim
scikit-learn
matplotlib
python-Levenshtein
```

---

## Results & Visualizations

Performance charts and metrics are saved in the `results/` folder:
- ROUGE score comparison: `rouge_scores_comparison.png`
- Training time comparison: `training_time_comparison.png`

---

## Citation

```bibtex
@project{summarization-benchmark2025,
  title={Implementation and Evaluation of Multi-Document Abstractive Summarization Models},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_GITHUB/summarization-benchmark}
}
```

---

## Contact

- **Email**: your.email@domain.com
- **GitHub Issues**: [Open an issue](https://github.com/YOUR_GITHUB/summarization-benchmark/issues)
