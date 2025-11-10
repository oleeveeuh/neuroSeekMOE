# NeuroSeek-MoE: Healthcare Language Model Training Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full-stack machine learning project for training specialized language models on healthcare and neuroscience research papers. This project demonstrates end-to-end ML engineering: from data collection and curation to model training and deployment.

## About This Project

This project was built to explore how specialized language models can be trained on domain-specific scientific literature. The goal was to create a complete pipeline that could process tens of thousands of ArXiv papers, curate high-quality healthcare and ML research, and train a model capable of understanding medical terminology and research contexts.

**Key Challenge**: Building a memory-efficient pipeline that could handle 30-40k papers without running out of RAM, while maintaining data quality through intelligent filtering and domain classification.

## What I Built

A complete ML training pipeline with the following components:

### Data Collection & Processing
- **ArXiv Paper Collector**: RAM-efficient batch collection system that processes papers in small batches (25 papers/batch) to prevent memory issues
- **PDF Text Extraction**: Parallel extraction of text from research papers with resume capability
- **NeMo Curator Integration**: Advanced text curation using NVIDIA's NeMo Curator for quality filtering and deduplication
- **Healthcare-Specific Preprocessing**: Custom text processing that preserves medical terminology and extracts research sections

### Model Training Infrastructure
- **Mixture of Experts (MoE) Architecture**: DeepSeek-MoE style architecture with shared and routed experts for efficient scaling
- **Streaming Dataset**: Custom PyTorch IterableDataset that streams data from disk, keeping memory usage under 500MB regardless of corpus size
- **Colab-Optimized Training**: Training loop designed for Google Colab's GPU constraints with mixed precision, gradient accumulation, and checkpointing
- **Domain-Aware Loss Weighting**: Custom loss function that gives higher weight to neurodegeneration and neuroscience papers

### Evaluation & Deployment
- **Comprehensive Evaluation**: Metrics including perplexity, domain classification accuracy, and relevance ranking
- **Production Inference Pipeline**: Fast inference system with embedding generation, similarity search, and domain classification

## Technical Highlights

### Memory Efficiency
One of the main challenges was handling large datasets without running out of memory. I solved this by:
- Implementing streaming I/O at every stage (no full dataset in RAM)
- Batch-based collection with automatic memory monitoring
- Custom IterableDataset that streams papers during training
- Aggressive garbage collection and memory cleanup

### Data Quality & Domain Adaptation
NeMo Curator enables effective domain-adaptive pretraining through:
- **Quality Filtering**: Removes low-quality content (short texts, high noise, non-English)
- **Domain Classification**: Identifies and scores healthcare subdomains (neurodegeneration, neuroscience, medical imaging, clinical, drug discovery)
- **Relevance Scoring**: Combines healthcare domain keywords with ML method keywords to ensure domain-specific relevance
- **Medical Term Preservation**: Maintains critical terminology (disease names, abbreviations, scientific terms)
- **Section Extraction**: Preserves research paper structure (Abstract, Introduction, Methods, Results, Discussion)
- **Deduplication**: Fuzzy deduplication removes near-duplicate papers, ensuring diverse training data

This curation process creates a domain-specific dataset that enables the model to learn healthcare-specific patterns, terminology, and research contexts, effectively performing domain-adaptive pretraining.

### Scalability
The pipeline is designed to scale:
- Processes 30-40k papers efficiently
- Resume capability at every stage (no data loss on interruption)
- Parallel processing where possible (PDF extraction, text processing)
- Configurable via YAML for easy experimentation

## Project Structure

```
neuroseek-moe/
├── data_pipeline.py          # Complete data pipeline (collect, extract, curate, process, tokenize)
├── arxiv_dataset.py          # Streaming IterableDataset for training
├── training_adapter.py       # Model adapter (connects dataset to model)
├── train_colab.py            # Colab-optimized training loop
├── train_real.py             # Model implementation (SimpleMoEModel)
├── evaluate.py               # Evaluation utilities
├── inference.py              # Production inference pipeline
├── run_pipeline.py           # Pipeline orchestration
├── config.yaml               # Configuration file
└── notebooks/
    └── ArXiv_Pipeline_Colab.ipynb  # Colab notebook for easy execution
```

## Getting Started

### Quick Start (Google Colab)

The easiest way to run this project is using the provided Colab notebook and connecting to a hosted GPU runtime:

1. Open `notebooks/ArXiv_Pipeline_Colab.ipynb` in Google Colab
2. Run all cells sequentially
3. The notebook handles all setup, configuration, and execution automatically


### Local Installation

```bash
git clone https://github.com/your-username/neuroseek-moe.git
cd neuroseek-moe
pip install -r requirements.txt

# Optional: NeMo Curator (Linux only)
# pip install "nemo-curator[text]"  # CPU version
# pip install "nemo-curator[text_cuda12]"  # CUDA 12 version
```

### Running the Pipeline

**Option 1: Complete Pipeline (Recommended)**
```bash
python run_pipeline.py --config config.yaml
```

**Option 2: Step-by-Step**
```bash
# Collect papers
python data_pipeline.py collect --max-papers 40000

# Extract PDF texts
python data_pipeline.py extract --input ./data/arxiv/arxiv_papers.jsonl --output-dir ./data/arxiv/texts

# Curate with NeMo Curator (Linux only)
python data_pipeline.py curate --text-dir ./data/arxiv/texts --metadata ./data/arxiv/arxiv_papers.jsonl --output ./data/arxiv/curated_dataset.jsonl

# Process and train tokenizer
python data_pipeline.py process --input ./data/arxiv/curated_dataset.jsonl --output ./data/arxiv/processed_dataset.jsonl
python data_pipeline.py tokenize --input ./data/arxiv/processed_dataset.jsonl --output-dir ./data/arxiv

# Train model
python train_colab.py \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --checkpoint-dir ./checkpoints \
    --batch-size 6 \
    --max-steps 50000
```

## Mixture of Experts (MoE) Architecture

The model uses a DeepSeek-MoE inspired architecture that efficiently scales to large models while maintaining computational efficiency.

### Architecture Overview

The MoE model consists of two types of experts:

**1. Shared Experts (Always Active)**
- **Purpose**: Provide baseline functionality for all tokens
- **Count**: 2 experts (default)
- **Activation**: Always process all tokens, regardless of routing decisions
- **Role**: Ensure all tokens receive processing, act as fallback for unprocessed tokens

**2. Routed Experts (Dynamically Selected)**
- **Purpose**: Specialize on different patterns and domains
- **Count**: 4 experts (default, scalable to 60+ for larger models)
- **Activation**: Selected via Expert Choice routing
- **Role**: Each expert selects top_k tokens to process based on routing scores

### Expert Choice Routing

Unlike traditional Token Choice routing (where tokens choose experts), this implementation uses **Expert Choice routing** where experts select tokens:

1. **Routing Gate**: Computes logits for each token-expert pair
2. **Expert Selection**: Each routed expert selects top_k tokens (default: k=2) with highest scores
3. **Capacity Control**: Enforces sparsity via capacity factor (default: 1.5x average load)
4. **Load Balancing**: Auxiliary loss encourages uniform expert utilization
5. **Fail-safe**: Tokens not selected by routed experts fall back to shared experts

### Key Advantages

- **Efficiency**: Only activates a subset of experts per token, reducing computation
- **Specialization**: Routed experts can specialize on different healthcare subdomains
- **Scalability**: Can scale to 60+ experts without proportional increase in computation
- **Robustness**: Shared experts ensure all tokens are processed even if routing fails

### Training Dynamics

The model learns to route tokens to appropriate experts through:
- **Load Balance Loss**: Encourages uniform expert utilization
- **Z-Loss**: Prevents extreme routing confidence, maintains exploration
- **Capacity Loss**: Enforces sparsity constraints
- **Temperature Scheduling**: Gradually reduces routing noise during training

### Domain Specialization

Through training on curated healthcare data, routed experts naturally specialize:
- Some experts focus on neurodegeneration terminology
- Others specialize in medical imaging or clinical language
- Shared experts maintain general language understanding
- The routing mechanism learns to match tokens to domain-appropriate experts

## NeMo Curator: Domain-Adaptive Pretraining

NeMo Curator plays a crucial role in enabling effective domain-adaptive pretraining by creating a high-quality, domain-specific dataset tailored for healthcare language modeling.

### How NeMo Curator Enables Domain Adaptation

**1. Quality-Based Filtering**
- Removes low-quality content (short texts, high noise ratio, non-English)
- Ensures minimum content quality (100-5000 tokens, >40% alphanumeric)
- Filters out boilerplate and repeated content
- Result: Only high-quality research content reaches the model

**2. Domain-Specific Classification**
- Identifies healthcare subdomains: neurodegeneration, neuroscience, medical imaging, clinical, drug discovery
- Scores domain relevance using keyword matching and ML method detection
- Filters papers to ensure healthcare+ML relevance (relevance score > 0.4)
- Result: Training data is specifically curated for healthcare domain

**3. Medical Terminology Preservation**
- Preserves critical medical terms (disease names, abbreviations, scientific terms)
- Maintains case sensitivity for proper nouns (Alzheimer, Parkinson, etc.)
- Extracts and tags medical terms by domain
- Result: Model learns domain-specific vocabulary and terminology

**4. Research Structure Maintenance**
- Extracts section boundaries (Abstract, Introduction, Methods, Results, Discussion)
- Preserves scientific paper structure
- Normalizes citations while maintaining context
- Result: Model learns research paper structure and scientific writing patterns

**5. Deduplication & Diversity**
- Fuzzy deduplication removes near-duplicate papers (MinHash similarity 0.95)
- Ensures diverse training examples
- Prevents overfitting to repeated content
- Result: Diverse, high-quality training corpus

### Domain-Adaptive Pretraining Process

The curation pipeline effectively performs domain-adaptive pretraining by:

1. **Data Selection**: Filters 30-40k papers to only high-quality, healthcare-relevant content
2. **Domain Enrichment**: Ensures each paper has both healthcare domain keywords and ML method keywords
3. **Terminology Focus**: Preserves and highlights medical terminology throughout preprocessing
4. **Structure Learning**: Maintains research paper structure so model learns scientific writing patterns
5. **Quality Assurance**: Multi-stage filtering ensures only the best examples reach training

**Result**: The model is pretrained on a curated, domain-specific dataset rather than generic text, enabling it to:
- Understand medical terminology and context
- Recognize healthcare subdomains
- Generate domain-appropriate text
- Perform better on healthcare-specific tasks

This is effectively **domain-adaptive pretraining** - the model learns healthcare-specific patterns, terminology, and research contexts from the very beginning of training, rather than being fine-tuned from a general-purpose model.

## Key Features

### Data Pipeline
- **RAM-Efficient Collection**: Batch processing (25 papers/batch) with automatic memory monitoring
- **21 Diverse Query Combinations**: ML+healthcare/neuroscience queries for broad coverage
- **Streaming to Disk**: No memory accumulation, writes immediately
- **Automatic Checkpointing**: Resume capability at every stage
- **Quality Filtering**: Word count, alphanumeric ratio, language detection, domain relevance

### Training
- **Memory-Efficient Streaming**: <500MB RAM during training regardless of corpus size
- **Colab-Optimized**: Mixed precision, gradient accumulation, dynamic batch sizing
- **Domain-Aware Loss**: Weighted loss for neurodegeneration and neuroscience papers
- **Checkpointing**: Saves every 5000 steps with resume capability

### Evaluation & Inference
- **Comprehensive Metrics**: Perplexity, domain accuracy, MRR@20, section classification
- **Fast Inference**: <100ms per paper on CPU
- **Embedding Generation**: For similarity search and literature review
- **Domain Classification**: Automatic healthcare subdomain detection

## Configuration

All parameters are configurable via `config.yaml`:
- Number of papers to collect
- NeMo Curator filter thresholds
- Training hyperparameters (batch size, learning rate, etc.)
- Evaluation settings
- Inference export options

## Challenges & Solutions

### Challenge 1: Memory Management
**Problem**: Processing 30-40k papers would exhaust RAM on most systems.

**Solution**: Implemented streaming I/O at every stage, batch-based collection with memory monitoring, and a custom IterableDataset that streams from disk during training.

### Challenge 2: Data Quality
**Problem**: Raw ArXiv papers include low-quality content, duplicates, and non-English text.

**Solution**: Multi-stage filtering pipeline using NeMo Curator for quality checks, domain relevance scoring, and fuzzy deduplication.

### Challenge 3: Colab Constraints
**Problem**: Google Colab has limited GPU memory (~12GB) and runtime limits.

**Solution**: Optimized training loop with mixed precision, gradient accumulation, frequent checkpointing, and automatic resume capability.

## Technologies Used

- **Python 3.8+**: Core language
- **PyTorch**: Deep learning framework
- **NeMo Curator**: Text curation (Linux only)
- **SentencePiece**: Tokenization
- **ArXiv API**: Paper collection
- **Dask**: Parallel processing for NeMo Curator
- **scikit-learn**: Evaluation metrics

## Results

The pipeline successfully:
- Collects and processes 30-40k healthcare+ML papers
- Trains a specialized tokenizer with 50k vocabulary
- Trains a language model optimized for healthcare literature
- Achieves >1000 samples/sec throughput on Colab GPU
- Maintains <500MB RAM usage during training

## Future Improvements

- [ ] Add support for multimodal data (images, diagrams)
- [ ] Implement fine-tuning on specific healthcare subdomains
- [ ] Add interactive visualization for training progress
- [ ] Create a web interface for literature review
- [ ] Expand to other scientific domains

## Learnings

This project taught me:
- How to build memory-efficient data pipelines for large-scale ML
- The importance of data quality in domain-specific models
- Techniques for optimizing training on resource-constrained systems
- End-to-end ML engineering from data collection to deployment

## License

MIT License - feel free to use this project for learning or as a starting point for your own work.

## Acknowledgments

- **ArXiv** for open access to research papers
- **NeMo Curator** for advanced text curation tools
- **SentencePiece** for efficient tokenization
- **PyTorch** team for the excellent deep learning framework

---

**Note**: This is a personal project demonstrating full-stack ML engineering. The code is provided as-is for educational purposes.
