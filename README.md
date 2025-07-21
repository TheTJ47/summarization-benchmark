# Multi-Document Abstractive Summarization Benchmark

This repository presents a rigorous implementation, training, and evaluation benchmark of ten multi-document abstractive summarization models using **Kaggle Notebooks**. The project is designed for reproducibility, extensibility, and fair model comparison using standardized evaluation metrics.

---

## Overview

- **Platform**: Kaggle Notebooks
- **Language**: Python 3.9+
- **Frameworks**: PyTorch, Hugging Face Transformers
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L (F1-scores)
- **Dataset**: CNN/Daily Mail (via Hugging Face datasets)
- **Usage**: Research, comparative analysis, educational demonstration

---

## Folder Structure (For Reference Only)

```
summarization-benchmark/
├── data/                # Dataset files if used locally (not needed on Kaggle)
├── models/              # Saved model checkpoints (optional)
├── outputs/             # Model-generated summaries
├── results/             # Evaluation results and plots
│   ├── rouge_scores_comparison.png
│   └── training_time_comparison.png
├── scripts/             # Utility scripts (if migrated for offline use)
├── requirements.txt     # Dependency reference
├── notebook.ipynb       # Main end-to-end workflow (Kaggle)
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

## Setup Instructions (Kaggle Environment)

### Getting Started on Kaggle

1. Open[Uploading task-3-10-models (1).ipynb…]()
 the [notebook on Kaggle](https://www.kaggle.com/code/ai003tejas/task-3-10-models) 
2. Ensure that the Kaggle kernel environment includes GPU acceleration (enable GPU under **Notebook Settings**).
3. Install required packages inside the Kaggle notebook (if not pre-installed):

```python
!pip install transformers datasets rouge-score evaluate sentencepiece nltk gensim scikit-learn matplotlib python-Levenshtein
```

4. Execute the notebook cells sequentially to:
   - Load data from Hugging Face datasets
   - Preprocess and tokenize input
   - Run all summarization models
   - Evaluate using ROUGE
   - Visualize the results

---

## How to Use (Kaggle)

- Run the entire pipeline inside `notebook.ipynb` on Kaggle.
- No command-line usage or local setup is necessary.
- Datasets are automatically loaded using Hugging Face API within the notebook.

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
- **Package versions**: Kaggle pre-installs most libraries; others can be installed in the notebook.
- **Unstable modules**: Custom fallback logic used for graph-based components.

---

## Requirements (For Reference Only)

Packages used within the notebook:
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

Output ROUGE scores and runtime plots are generated directly in the notebook interface.

---

## Citation

```bibtex
@project{summarization-benchmark,
  title={Implementation and Evaluation of Multi-Document Abstractive Summarization Models},
  author={Tejas Bagal},
  year={2025},
  url={https://github.com/TheTk47/summarization-benchmark}
}
```

---

## Contact

- **Email**: bagaltejas97@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/TheTJ47/summarization-benchmark/issues)
