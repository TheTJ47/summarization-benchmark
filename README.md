Implementation and Evaluation of Multi-Document Abstractive Summarization Models
This repository presents a comprehensive benchmark of 10 multi-document abstractive summarization models. It includes full implementations, fine-tuning, and evaluation pipelines using the CNN/Daily Mail dataset. Each model is compared head-to-head using ROUGE-1, ROUGE-2, and ROUGE-L metrics, ensuring consistency and reproducibility.

📊 ROUGE Score Comparison Chart
Performance is visualized through ROUGE metric plots and training time comparisons, located in the results/ directory.

📁 Folder Structure
bash
Copy code
summarization-benchmark/
├── data/                # Raw and preprocessed datasets
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/              # Saved model checkpoints
├── outputs/             # Model-generated summaries
├── results/             # Evaluation logs and plots
│   ├── rouge_scores_comparison.png
│   └── training_time_comparison.png
├── scripts/             # Utility and runner scripts
│   └── run_all_models.py
├── requirements.txt     # Required Python dependencies
├── README.md            # Project documentation
└── notebook.ipynb       # Main workflow notebook
🚀 Getting Started
Prerequisites
Python 3.9+

PyTorch

CUDA 11+ (Recommended)

Quick Setup
bash
Copy code
git clone https://github.com/YOUR_GITHUB/summarization-benchmark.git
cd summarization-benchmark
pip install -r requirements.txt
Place dataset files in the /data directory if using local CSVs.

Launch the end-to-end pipeline using:

bash
Copy code
jupyter notebook notebook.ipynb
🧠 Models Implemented
Model	Type / Source
1. PRIMERA	Transformer (Proxy Approximation)
2. BART	Hugging Face – facebook/bart-large-cnn
3. PEGASUS	Hugging Face – google/pegasus-cnn_dailymail
4. T5 (Base, Large)	Hugging Face
5. Longformer / BigBird / LongT5	Long-range Transformers (Proxy)
6. Absformer	Lead-3 Proxy (Unsupervised)
7. TextRank	Graph-Based (Extractive)
8. Hierarchical Transformer	Two-Stage Simulation
9. DCA (Deep Communicating Agents)	Ensemble Simulation
10. Topic-Guided Model	Neural + External Knowledge

🧾 Dataset Preparation
Primary Dataset: CNN/Daily Mail, loaded via Hugging Face or /data CSVs.

Preprocessing includes:

Whitespace normalization

Token-level cleanup

Prefix formatting (e.g., "summarize: " for T5)

You can customize the dataset size and batch processing directly in the notebook.

🔧 Training & Inference
Supervised Models (BART, PEGASUS, T5):

Support both inference and optional fine-tuning (1 epoch default)

Unsupervised & Graph-Based Models:

Implemented using nltk, scikit-learn, and gensim

All models use the same test subset for fair comparison.

📐 Evaluation Metrics
Each model is evaluated using:

ROUGE-1: Unigram overlap

ROUGE-2: Bigram overlap

ROUGE-L: Longest common subsequence (sentence-level)

Results are reported as F1-scores, averaged across all test samples.

📊 Results & Visualizations
Output ROUGE tables and plots are available in:

Copy code
results/
├── rouge_scores_comparison.png
├── training_time_comparison.png
└── evaluation_logs/
⚙️ Hyperparameters
Parameter	Value
Random Seed	42
Batch Size	4 (default)
Max Input Length	512–4096 tokens
Max Summary Length	128 tokens
Beam Size	4
Device	GPU/CPU
Learning Rate	2e-5

🛠️ Issues Encountered & Workarounds
Unavailable Weights: For some models (e.g., Absformer, TG-MultiSum), public checkpoints do not exist. Proxy implementations like Lead-3 or TextRank were used and clearly marked.

Library Conflicts: Resolved via stable versions listed in requirements.txt

Graph-Based Instability: Replaced some third-party libraries with in-house fallback code.

🧪 How to Run Models
Option A: Interactive (Recommended)
Open and run all cells in notebook.ipynb.

Option B: Automated Script
Modify and run:

bash
Copy code
python scripts/run_all_models.py
📦 Requirements
txt
Copy code
torch==2.0.1
transformers==4.30.2
datasets==2.13.1
evaluate==0.4.0
rouge-score==0.1.2
sentencepiece
nltk
python-Levenshtein
gensim
scikit-learn
matplotlib
Install via:

bash
Copy code
pip install -r requirements.txt
📄 Citation
If you use this repository, please cite:

bibtex
Copy code
@project{summarization-benchmark2025,
  title={Implementation and Evaluation of Multi-Document Abstractive Summarization Models},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_GITHUB/summarization-benchmark}
}
📬 Contact
For questions, contributions, or collaborations:

Email: your.email@domain.com

Issues: Submit a GitHub Issue
