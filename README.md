# NeuroSeek-MoE: Healthcare+ML Paper Training Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete data pipeline and training system for training language models on healthcare+ML papers from ArXiv. Features efficient streaming datasets, NeMo Curator integration, and Colab-optimized training.

## 🧠 Overview

This project provides a complete pipeline for:
- **ArXiv Paper Collection**: Collect 30-40k healthcare+CS+ML papers with efficient streaming
- **PDF Text Extraction**: Extract and process text from ArXiv PDFs (first 6 pages, 12k chars max)
- **NeMo Curator Integration**: Advanced text curation with quality filtering and domain classification
- **Healthcare-Specific Preprocessing**: Section extraction, medical term preservation, citation normalization
- **Tokenizer Training**: SentencePiece BPE tokenizer optimized for healthcare terminology
- **Streaming Dataset**: Memory-efficient IterableDataset for training (<500MB RAM)
- **Training Pipeline**: Colab-optimized training with mixed precision, gradient accumulation, and checkpointing
- **Evaluation & Inference**: Comprehensive evaluation metrics and production inference pipeline

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/neuroseek-moe.git
cd neuroseek-moe

# Core dependencies
pip install -r requirements.txt

# Optional: NeMo Curator (Linux only)
# pip install "nemo-curator[text]"  # CPU version
# pip install "nemo-curator[text_cuda12]"  # CUDA 12 version
```

### Option 1: Google Colab (Recommended for Non-Linux)

For easy setup and GPU access, use the Colab notebook:

1. Open `notebooks/ArXiv_Pipeline_Colab.ipynb` in Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Install all dependencies
   - Configure the pipeline for Colab
   - Run the complete pipeline
   - Provide progress monitoring and visualization
   - Handle NeMo Curator automatically (Colab uses Linux)

**Features:**
- Automatic environment setup
- GPU detection and optimization
- Progress monitoring with plots
- Resume capability
- Download results to Google Drive

### Option 2: Complete Pipeline (Local/Linux)

Run the entire pipeline from paper collection to inference export with a single command:

```bash
# Run complete pipeline
python run_pipeline.py --config config.yaml

# Resume from a specific step (if pipeline was interrupted)
python run_pipeline.py --config config.yaml --start-from-step 5

# The script automatically:
# - Detects which steps are already complete
# - Resumes from the first incomplete step
# - Cleans up intermediate files after processing
# - Generates a comprehensive report
```

**Configuration**: Edit `config.yaml` to customize:
- Number of papers to collect
- NeMo Curator filter thresholds
- Training hyperparameters
- Evaluation settings
- Inference export options

### Option 3: Manual Step-by-Step Pipeline

```bash
# Step 1: Collect ArXiv papers (30-40k papers)
# Note: Automatically resumes from checkpoint if interrupted
python data_pipeline.py collect --max-papers 40000 --output-dir ./data/arxiv

# To enable date filtering (2015-2024), edit data_pipeline.py:
# Set MIN_YEAR = 2015 and MAX_YEAR = 2024 (around line 207-210)

# Step 2: Extract PDF texts
python data_pipeline.py extract \
    --input ./data/arxiv/arxiv_papers.jsonl \
    --output-dir ./data/arxiv/texts \
    --workers 4

# Step 3: Preprocess and classify domains (optional, if not using NeMo Curator)
python data_pipeline.py preprocess \
    --metadata ./data/arxiv/arxiv_papers.jsonl \
    --text-dir ./data/arxiv/texts \
    --output ./data/arxiv/processed_dataset.jsonl

# Step 4: Curate with NeMo Curator (recommended, Linux only)
python data_pipeline.py curate \
    --text-dir ./data/arxiv/texts \
    --metadata ./data/arxiv/arxiv_papers.jsonl \
    --output ./data/arxiv/curated_dataset.jsonl \
    --min-relevance-score 0.5

# Step 5: Process curated dataset with healthcare-specific preprocessing
python data_pipeline.py process \
    --input ./data/arxiv/curated_dataset.jsonl \
    --output ./data/arxiv/processed_dataset.jsonl \
    --workers 4

# Step 6: Train SentencePiece tokenizer
python data_pipeline.py tokenize \
    --input ./data/arxiv/processed_dataset.jsonl \
    --output-dir ./data/arxiv \
    --vocab-size 50000

# Step 7: Train model (see train_colab.py)
```

## 📊 Data Pipeline

### Stage 1: ArXiv Paper Collection

Collects papers from ArXiv using healthcare+CS+ML queries:
- `cat:cs.LG AND (healthcare OR medical OR clinical)`
- `cat:cs.AI AND (neurodegeneration OR disease)`
- `cat:q-bio.NC AND (machine learning)`

**Features**:
- **RAM-Efficient Batch Collection**: Collects in small batches (10 papers/batch) to prevent OOM
- **Streaming to Disk**: No memory accumulation, writes immediately
- **Deduplication**: By `arxiv_id` (automatic)
- **Rate Limiting**: 3 requests/sec (configurable)
- **Checkpointing**: After every batch (fully resumable)
- **Date Filtering**: Optional date range filtering (disabled by default)
- **Memory Monitoring**: Automatic batch size adjustment based on RAM usage

```bash
# Basic collection (no date filtering, all years)
python data_pipeline.py collect --max-papers 40000

# With custom batch size and RAM target
python data_pipeline.py collect \
    --max-papers 40000 \
    --batch-size 10 \
    --ram-target 50.0
```

**Date Filtering**:

To enable date filtering (e.g., 2015-2024), edit `data_pipeline.py`:

```python
# In data_pipeline.py, around line 207-210:
MIN_YEAR = 2015  # Set to None to disable minimum
MAX_YEAR = 2024  # Set to None to disable maximum
```

Or use the efficient collector directly:

```python
from data_pipeline import RAMEfficientArxivCollector
import os

# Enable date filtering
MIN_YEAR = 2015
MAX_YEAR = 2024

collector = RAMEfficientArxivCollector(
    output_file="./data/arxiv/arxiv_papers.jsonl",
    checkpoint_file="./data/arxiv/collection_checkpoint.json",
    batch_size=10,
    ram_target_percent=50.0
)
```

**Resuming from Checkpoint**:

The collector automatically resumes from the last checkpoint:

1. **Automatic Resume**: Simply run the same command again:
   ```bash
   python data_pipeline.py collect --max-papers 40000
   ```
   The collector will:
   - Load existing papers from `arxiv_papers.jsonl`
   - Load checkpoint state from `collection_checkpoint.json`
   - Continue from where it left off

2. **Checkpoint Files**:
   - `./data/arxiv/arxiv_papers.jsonl`: All collected papers (main output)
   - `./data/arxiv/collection_checkpoint.json`: Checkpoint state (for resume)

3. **Manual Checkpoint Management**:
   - Checkpoints are saved after **every batch** (default: every 10 papers)
   - To force a fresh start, delete `collection_checkpoint.json`
   - To keep existing papers but reset checkpoint, keep `arxiv_papers.jsonl` and delete `collection_checkpoint.json`

**Example: Resume After Interruption**:

```bash
# Collection was interrupted at 5000 papers
# Simply run again - it will resume automatically:
python data_pipeline.py collect --max-papers 40000

# Output:
# 📖 Resuming from checkpoint: 5000 papers
# 📊 Starting from: 5000 papers
# 🎯 Target: 40000 papers
# ... continues from batch 501
```

### Stage 2: PDF Text Extraction

Extracts text from ArXiv PDFs:
- First 6 pages only
- Max 12,000 characters per paper
- Memory-efficient streaming downloads
- Parallel processing (2-4 workers)
- Resume capability with checkpoints

```bash
python data_pipeline.py extract \
    --input ./data/arxiv/arxiv_papers.jsonl \
    --output-dir ./data/arxiv/texts \
    --workers 4 \
    --rate-limit 0.4
```

### Stage 3: NeMo Curator Curation (Optional, Linux only)

Advanced text curation with quality filtering:
- **Text Cleaning**: Remove URLs, emails, citations, normalize whitespace
- **Quality Filtering**: Word count (100-5000), alphanumeric ratio (>40%), language detection
- **Domain Filtering**: Relevance scoring for healthcare domains
- **Deduplication**: Fuzzy deduplication with MinHash (similarity 0.95)
- **GPU Support**: Optional GPU-accelerated deduplication

```bash
python data_pipeline.py curate \
    --text-dir ./data/arxiv/texts \
    --metadata ./data/arxiv/arxiv_papers.jsonl \
    --output ./data/arxiv/curated_dataset.jsonl \
    --use-gpu  # Optional: use GPU for deduplication
```

### Stage 4: Healthcare-Specific Preprocessing

Post-NeMo Curator processing:
- Extract section boundaries (Abstract, Introduction, Methods, Results, Discussion)
- Preserve scientific abbreviations (MMSE, MRI, EEG, fMRI, PET, CSF)
- Keep medical terminology intact
- Remove page numbers, headers/footers
- Normalize citations
- Detect medical terms by domain

```bash
python data_pipeline.py process \
    --input ./data/arxiv/curated_dataset.jsonl \
    --output ./data/arxiv/processed_dataset.jsonl \
    --workers 4
```

### Stage 5: Tokenizer Training

Train SentencePiece BPE tokenizer:
- Vocabulary size: 50,000
- Model type: BPE
- Character coverage: 0.9995
- Special tokens: `[DISEASE]`, `[PROTEIN]`, `[DRUG]`, `[GENE]`
- Normalization: identity (preserve case)
- Validation: Medical term tokenization efficiency

```bash
python data_pipeline.py tokenize \
    --input ./data/arxiv/processed_dataset.jsonl \
    --output-dir ./data/arxiv \
    --vocab-size 50000
```

**Output**:
- `healthcare_tokenizer.model`
- `healthcare_tokenizer.vocab`
- `tokenizer_validation_report.json`

## 🎯 Training

### Streaming Dataset

Memory-efficient `IterableDataset` for training:
- Streams from disk (<500MB RAM regardless of corpus size)
- Worker-aware distribution (no duplicates)
- Shuffling with buffer (shuffle_buffer=100)
- Variable-length sequences (no padding in dataset)
- Skips papers with <64 tokens

```python
from arxiv_dataset import ArXivStreamingDataset, create_dataloader
import sentencepiece as spm

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('./data/arxiv/healthcare_tokenizer.model')

# Create dataset
dataset = ArXivStreamingDataset(
    text_dir='./data/arxiv/texts',
    metadata_jsonl='./data/arxiv/processed_dataset.jsonl',
    tokenizer=tokenizer,
    max_length=512,
    min_length=64,
    shuffle_buffer=100
)

# Create dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=6,
    num_workers=4,
    pin_memory=True
)
```

### Model Adapter

Connects dataset to DeepSeekMoE model:
- Handles device transfer
- Variable-length sequence padding (within batch)
- Forward pass: `logits = model(input_ids)`
- Loss computation: `F.cross_entropy(logits, target_ids)`
- Domain-aware weighting (optional):
  - Neurodegeneration: `loss *= 1.5`
  - Neuroscience: `loss *= 1.2`

```python
from training_adapter import ModelAdapter

adapter = ModelAdapter(
    model=model,
    device='cuda',
    domain_weights={'neurodegeneration': 1.5, 'neuroscience': 1.2}
)

for batch in dataloader:
    result = adapter.process_batch(batch)
    loss = result['loss']
    logits = result['logits']
    metadata = result['batch_metadata']
```

### Colab Training

Optimized for Colab GPU (~12GB VRAM):
- Mixed precision training (`torch.cuda.amp`)
- Gradient accumulation (simulate larger batches)
- Gradient checkpointing (if available)
- Cosine annealing with warmup
- Gradient clipping (max_norm=1.0)
- Dynamic batch sizing
- Checkpointing every 5000 steps

```bash
python train_colab.py \
    --model-path ./checkpoints/model.pt \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --checkpoint-dir ./checkpoints \
    --batch-size 6 \
    --gradient-accumulation-steps 4 \
    --max-steps 50000 \
    --learning-rate 5e-4
```

## 📊 Evaluation

Comprehensive evaluation utilities:
- Perplexity calculation
- Domain classification accuracy
- Neurodegeneration relevance ranking (MRR@20)
- Section classification accuracy

```bash
python evaluate.py \
    --model-checkpoint ./checkpoints/step_50000.pt \
    --test-dataset ./data/arxiv/test_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluation
```

## 🔌 Inference

Production inference pipeline:
- Batch encoding
- Literature review (similarity search)
- Domain classification
- Embedding caching
- Optional: INT8 quantization, ONNX export

```bash
# Generate embedding for a single text
python inference.py \
    --checkpoint ./checkpoints/step_50000.pt \
    --tokenizer ./data/arxiv/healthcare_tokenizer.model \
    embed \
    --text "Alzheimer disease and tau protein aggregation"

# Batch encode multiple texts
python inference.py \
    --checkpoint ./checkpoints/step_50000.pt \
    --tokenizer ./data/arxiv/healthcare_tokenizer.model \
    batch \
    --texts input_texts.txt \
    --output embeddings.npy

# Literature review (similarity search)
python inference.py \
    --checkpoint ./checkpoints/step_50000.pt \
    --tokenizer ./data/arxiv/healthcare_tokenizer.model \
    review \
    --query "neurodegeneration and machine learning" \
    --corpus corpus_embeddings.npz \
    --top-k 10

# Domain classification
python inference.py \
    --checkpoint ./checkpoints/step_50000.pt \
    --tokenizer ./data/arxiv/healthcare_tokenizer.model \
    classify \
    --text "Clinical trial results for Alzheimer's disease treatment"

# Precompute corpus embeddings
python inference.py \
    --checkpoint ./checkpoints/step_50000.pt \
    --tokenizer ./data/arxiv/healthcare_tokenizer.model \
    precompute \
    --corpus corpus_texts.txt \
    --output corpus_embeddings.npz \
    --batch-size 32
```

## 📁 Project Structure

```
neuroseek-moe/
├── data_pipeline.py          # Complete data pipeline (collect, extract, curate, process, tokenize)
├── arxiv_dataset.py          # Streaming IterableDataset for training
├── training_adapter.py       # Model adapter (connects dataset to model)
├── train_colab.py            # Colab-optimized training loop
├── evaluate.py               # Evaluation utilities
├── inference.py              # Production inference pipeline
├── model_architecture.py     # Model architecture definitions
├── train_real.py             # Model implementation (SimpleMoEModel)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── arxiv/                # ArXiv data directory
│       ├── arxiv_papers.jsonl      # Collected metadata
│       ├── texts/                  # Extracted text files
│       ├── curated_dataset.jsonl   # NeMo Curator output
│       ├── processed_dataset.jsonl # Final processed dataset
│       └── healthcare_tokenizer.*   # Trained tokenizer
├── checkpoints/              # Model checkpoints
└── notebooks/                # Jupyter notebooks (optional)
```

## 🔧 Configuration

### Data Pipeline

**ArXiv Collection**:
- `--max-papers`: Maximum papers to collect (default: 40000)
- `--batch-size`: Papers per batch (default: 10, adjust for RAM)
- `--ram-target`: Target RAM percentage to stay below (default: 50.0)
- `--rate-limit`: Requests per second (default: 3.0)
- **Date Filtering**: Edit `MIN_YEAR` and `MAX_YEAR` in `data_pipeline.py` (default: None, all years)
- **Checkpointing**: Automatic after every batch (resume by running the same command)

**PDF Extraction**:
- `--workers`: Number of parallel workers (default: 3, choices: 2-4)
- `--rate-limit`: Delay between requests in seconds (default: 0.4)

**NeMo Curator**:
- `--min-relevance-score`: Minimum domain relevance (default: 0.5)
- `--use-gpu`: Use GPU for deduplication (optional)
- `--skip-dedup`: Skip deduplication (memory-constrained)

**Tokenizer**:
- `--vocab-size`: Vocabulary size (default: 50000)
- `--model-prefix`: Tokenizer file prefix (default: healthcare_tokenizer)

### Training

**Dataset**:
- `max_length`: Maximum sequence length (default: 512)
- `min_length`: Minimum sequence length (default: 64)
- `shuffle_buffer`: Shuffle buffer size (default: 100)

**DataLoader**:
- `batch_size`: Batch size (default: 6 for Colab)
- `num_workers`: Number of workers (default: 4)
- `pin_memory`: Pin memory for GPU (default: True)

**Training**:
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `max_steps`: Maximum training steps (default: 50000)
- `learning_rate`: Learning rate (default: 5e-4)
- `warmup_steps`: Warmup steps (default: 2000)

## 📚 Key Features

### Memory Efficiency
- **Streaming I/O**: All stages stream to disk, no full dataset in RAM
- **IterableDataset**: Streams papers during training (<500MB RAM)
- **Batch Collection**: Small batches (10 papers) prevent OOM errors
- **Memory Monitoring**: Automatic batch size adjustment based on RAM usage
- **Checkpointing**: Resume capability at every stage (automatic resume)

### Checkpointing & Resume
- **Automatic Checkpoints**: Saved after every batch (default: every 10 papers)
- **Resume Support**: Simply re-run the same command to resume from last checkpoint
- **Checkpoint Files**:
  - `arxiv_papers.jsonl`: Main output (all collected papers)
  - `collection_checkpoint.json`: Checkpoint state (for resume)
- **No Data Loss**: Interrupted collections can be resumed without losing progress

### Performance
- **Parallel Processing**: Multi-threaded PDF extraction and preprocessing
- **Dask Integration**: NeMo Curator uses Dask for parallelization
- **GPU Support**: Optional GPU acceleration for deduplication
- **Throughput**: >1000 samples/sec on Colab GPU

### Healthcare-Specific
- **Medical Term Preservation**: Disease names, abbreviations preserved
- **Domain Classification**: Automatic domain tagging (neurodegeneration, neuroscience, etc.)
- **Section Extraction**: Abstract, Introduction, Methods, Results, Discussion
- **Special Tokens**: `[DISEASE]`, `[PROTEIN]`, `[DRUG]`, `[GENE]`

## ⚠️ Platform Compatibility

- **NeMo Curator**: Linux only (will gracefully skip on macOS/Windows)
- **DeepSpeed**: Linux/Colab only (training falls back to PyTorch on macOS)
- **All other components**: Cross-platform (macOS, Linux, Windows)

## 🔄 Advanced Usage

### Date Filtering

To collect papers from a specific date range:

1. **Edit `data_pipeline.py`** (around line 207-210):
   ```python
   # Target date range (set to None to disable date filtering)
   MIN_YEAR = 2015  # Minimum year (None = no minimum)
   MAX_YEAR = 2024  # Maximum year (None = no maximum)
   ```

2. **Run collection**:
   ```bash
   python data_pipeline.py collect --max-papers 40000
   ```

3. **Date filtering behavior**:
   - Papers outside the date range are automatically skipped
   - Progress output shows how many papers were skipped due to date filtering
   - Set `MIN_YEAR = None` and `MAX_YEAR = None` to disable filtering (collect all years)

### Resuming from Checkpoint

The collection pipeline automatically saves checkpoints and can resume:

1. **Automatic Resume**:
   ```bash
   # If collection was interrupted, simply run again:
   python data_pipeline.py collect --max-papers 40000
   # Output: "📖 Resuming from checkpoint: 5000 papers"
   ```

2. **Checkpoint Location**:
   - Main output: `./data/arxiv/arxiv_papers.jsonl`
   - Checkpoint state: `./data/arxiv/collection_checkpoint.json`

3. **Manual Checkpoint Management**:
   ```bash
   # Force fresh start (delete checkpoint, keep existing papers):
   rm ./data/arxiv/collection_checkpoint.json
   
   # Complete reset (delete both checkpoint and output):
   rm ./data/arxiv/arxiv_papers.jsonl
   rm ./data/arxiv/collection_checkpoint.json
   ```

4. **Checkpoint Frequency**:
   - Checkpoints are saved after **every batch** (default: every 10 papers)
   - This ensures minimal data loss if interrupted
   - Adjust `--batch-size` to control checkpoint frequency

5. **Resume Behavior**:
   - Loads existing papers from `arxiv_papers.jsonl`
   - Loads checkpoint state from `collection_checkpoint.json`
   - Skips already-collected papers (deduplication)
   - Continues from the next batch

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **ArXiv** for open access to research papers
- **NeMo Curator** for advanced text curation
- **SentencePiece** for efficient tokenization
- **PyTorch** for deep learning framework

---

**NeuroSeek-MoE**: Training language models on healthcare+ML literature 🧠✨
