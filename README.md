# NeuroSeek-MoE: Healthcare+ML Paper Training Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete data pipeline and training system for training language models on healthcare+ML papers from ArXiv. Features efficient streaming datasets, NeMo Curator integration, and Colab-optimized training.

## 🧠 Overview

Complete pipeline for training language models on healthcare+ML literature:
- **ArXiv Paper Collection**: Collect 30-40k papers with RAM-efficient batch processing
- **PDF Text Extraction**: Extract text from ArXiv PDFs (first 6 pages, 12k chars max)
- **NeMo Curator Integration**: Advanced text curation with quality filtering and domain classification
- **Healthcare-Specific Preprocessing**: Section extraction, medical term preservation
- **Tokenizer Training**: SentencePiece BPE tokenizer optimized for healthcare terminology
- **Streaming Dataset**: Memory-efficient IterableDataset (<500MB RAM)
- **Training Pipeline**: Colab-optimized with mixed precision, gradient accumulation
- **Evaluation & Inference**: Comprehensive metrics and production inference

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/neuroseek-moe.git
cd neuroseek-moe
pip install -r requirements.txt

# Optional: NeMo Curator (Linux only)
# pip install "nemo-curator[text]"  # CPU version
# pip install "nemo-curator[text_cuda12]"  # CUDA 12 version
```

### Option 1: Google Colab (Recommended)

1. Open `notebooks/ArXiv_Pipeline_Colab.ipynb` in Google Colab
2. Run all cells sequentially
3. The notebook handles all setup, configuration, and execution

**Features**: Automatic environment setup, GPU detection, progress monitoring, resume capability

### Option 2: Complete Pipeline (Local/Linux)

Run the entire pipeline with a single command:

```bash
# Run complete pipeline
python run_pipeline.py --config config.yaml

# Resume from a specific step
python run_pipeline.py --config config.yaml --start-from-step 5
```

**Configuration**: Edit `config.yaml` to customize all parameters.

### Option 3: Manual Step-by-Step

```bash
# Step 1: Collect ArXiv papers
python data_pipeline.py collect --max-papers 40000 --output-dir ./data/arxiv

# Step 2: Extract PDF texts
python data_pipeline.py extract \
    --input ./data/arxiv/arxiv_papers.jsonl \
    --output-dir ./data/arxiv/texts \
    --workers 4

# Step 3: Curate with NeMo Curator (Linux only)
python data_pipeline.py curate \
    --text-dir ./data/arxiv/texts \
    --metadata ./data/arxiv/arxiv_papers.jsonl \
    --output ./data/arxiv/curated_dataset.jsonl

# Step 4: Process curated dataset
python data_pipeline.py process \
    --input ./data/arxiv/curated_dataset.jsonl \
    --output ./data/arxiv/processed_dataset.jsonl

# Step 5: Train tokenizer
python data_pipeline.py tokenize \
    --input ./data/arxiv/processed_dataset.jsonl \
    --output-dir ./data/arxiv \
    --vocab-size 50000

# Step 6: Train model
python train_colab.py \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --checkpoint-dir ./checkpoints \
    --batch-size 6 \
    --gradient-accumulation-steps 4 \
    --max-steps 50000
```

## 📊 Data Pipeline

### ArXiv Paper Collection

**Features**:
- RAM-efficient batch collection (default: 25 papers/batch)
- Streaming to disk (no memory accumulation)
- Automatic deduplication by `arxiv_id`
- Rate limiting (3 requests/sec)
- Checkpointing after every batch (fully resumable)
- Memory monitoring with automatic batch size adjustment

**Diverse Query Strategy**: Uses 21 different ML+healthcare/neuroscience query combinations for broad coverage.

```bash
# Basic collection
python data_pipeline.py collect --max-papers 40000

# Custom batch size and RAM target
python data_pipeline.py collect \
    --max-papers 40000 \
    --batch-size 25 \
    --ram-target 50.0
```

**Date Filtering**: Edit `MIN_YEAR` and `MAX_YEAR` in `data_pipeline.py` (around line 240-242).

**Resume**: Automatically resumes from checkpoint. Simply re-run the same command.

### PDF Text Extraction

Extracts first 6 pages, max 12k characters, with parallel processing and resume capability.

### NeMo Curator Curation (Linux only)

Advanced text curation:
- Text cleaning (URLs, emails, citations)
- Quality filtering (word count, alphanumeric ratio, language)
- Domain classification (healthcare relevance scoring)
- Fuzzy deduplication (MinHash similarity 0.95)

### Healthcare-Specific Preprocessing

- Section extraction (Abstract, Introduction, Methods, Results, Discussion)
- Medical term preservation (MMSE, MRI, EEG, etc.)
- Citation normalization
- Medical term detection by domain

### Tokenizer Training

SentencePiece BPE tokenizer:
- Vocabulary: 50,000
- Special tokens: `[DISEASE]`, `[PROTEIN]`, `[DRUG]`, `[GENE]`
- Case preservation (identity normalization)

## 🎯 Training

### Streaming Dataset

Memory-efficient `IterableDataset`:
- Streams from disk (<500MB RAM)
- Worker-aware distribution
- Shuffling with buffer
- Variable-length sequences

### Colab Training

Optimized for Colab GPU (~12GB VRAM):
- Mixed precision training
- Gradient accumulation
- Cosine annealing with warmup
- Checkpointing every 5000 steps

## 📊 Evaluation & Inference

### Evaluation

```bash
python evaluate.py \
    --model-checkpoint ./checkpoints/step_50000.pt \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluations
```

**Metrics**: Perplexity, domain classification accuracy, MRR@20, section classification accuracy

### Inference

```bash
# Generate embedding
python inference.py \
    --checkpoint ./checkpoints/step_50000.pt \
    --tokenizer ./data/arxiv/healthcare_tokenizer.model \
    embed \
    --text "Alzheimer disease and tau protein aggregation"

# Domain classification
python inference.py \
    --checkpoint ./checkpoints/step_50000.pt \
    --tokenizer ./data/arxiv/healthcare_tokenizer.model \
    classify \
    --text "Clinical trial results for Alzheimer's disease treatment"
```

## 📁 Project Structure

```
neuroseek-moe/
├── data_pipeline.py          # Complete data pipeline
├── arxiv_dataset.py          # Streaming IterableDataset
├── training_adapter.py       # Model adapter
├── train_colab.py            # Colab-optimized training
├── train_real.py             # Model implementation
├── evaluate.py               # Evaluation utilities
├── inference.py              # Production inference
├── run_pipeline.py           # Pipeline orchestration
├── config.yaml               # Configuration file
├── requirements.txt          # Dependencies
├── notebooks/
│   └── ArXiv_Pipeline_Colab.ipynb  # Colab notebook
└── data/arxiv/               # Data directory (excluded from Git)
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Number of papers to collect
- NeMo Curator filter thresholds
- Training hyperparameters
- Evaluation settings
- Inference export options

See `config.yaml` for all available options.

## 📚 Key Features

- **Memory Efficiency**: Streaming I/O, batch collection, <500MB RAM during training
- **Checkpointing**: Automatic checkpoints, resume capability, no data loss
- **Performance**: Parallel processing, GPU support, >1000 samples/sec
- **Healthcare-Specific**: Medical term preservation, domain classification, special tokens

## ⚠️ Platform Compatibility

- **NeMo Curator**: Linux only (gracefully skips on macOS/Windows)
- **DeepSpeed**: Linux/Colab only (falls back to PyTorch on macOS)
- **All other components**: Cross-platform

## 📖 Additional Documentation

- `EVALUATION_REQUIREMENTS.md`: Files needed for evaluation and visualization
- `GIT_GUIDELINES.md`: Git repository guidelines and data storage strategies

## 📄 License

MIT License

## 🙏 Acknowledgments

- **ArXiv** for open access to research papers
- **NeMo Curator** for advanced text curation
- **SentencePiece** for efficient tokenization
- **PyTorch** for deep learning framework

---

**NeuroSeek-MoE**: Training language models on healthcare+ML literature 🧠✨
