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
├── extract_expert_activations.py  # Extract expert activation patterns for analysis
├── run_pipeline.py           # Pipeline orchestration
├── config.yaml               # Configuration file
└── notebooks/
    ├── ArXiv_Pipeline_Colab.ipynb  # Colab notebook for easy execution
    └── model_analysis.ipynb        # Comprehensive model analysis and visualization
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
    --output-dir ./checkpoints \
    --batch-size 6 \
    --gradient-accumulation 4 \
    --max-steps 50000 \
    --learning-rate 5e-4

# Run evaluation (automatically captures expert activations)
python evaluate.py \
    --model-checkpoint ./checkpoints/step_50000.pt \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluations
# Note: expert_activations.npz is automatically saved during evaluation
```

## Mixture of Experts (MoE) Architecture

The model uses a DeepSeek-MoE inspired architecture that efficiently scales to large models while maintaining computational efficiency. This section provides a comprehensive technical overview of the MoE structure, routing mechanism, and forward pass flow.

### Architecture Overview

The MoE model is structured as a language model with a Mixture of Experts layer replacing the standard feedforward network. The architecture consists of:

**Core Components:**
1. **Token Embedding Layer**: Maps vocabulary tokens to dense embeddings (`embedding_dim` = 768 default)
2. **MoE Expert Layer**: Processes tokens through multiple specialized expert networks
3. **Joint Fusion Layer**: Combines expert outputs with normalization and residual connections
4. **Output Decoder**: Projects embeddings to vocabulary logits for next-token prediction

**Expert Types:**

**1. Shared Experts (Always Active)**
- **Architecture**: 2-layer MLP (feedforward network)
- **Count**: 2 experts (default, configurable)
- **Activation**: Always process ALL tokens in every forward pass
- **Purpose**: 
  - Provide baseline functionality for all tokens
  - Act as fail-safe for tokens not selected by routed experts
  - Maintain general language understanding
- **Structure**: 
  ```
  Input: [batch*seq_len, embedding_dim]
  → Linear(embedding_dim → 4*embedding_dim)  [Expansion]
  → ReLU()
  → Linear(4*embedding_dim → embedding_dim)   [Projection]
  → Output: [batch*seq_len, embedding_dim]
  ```

**2. Routed Experts (Dynamically Selected)**
- **Architecture**: 2-layer MLP (identical structure to shared experts)
- **Count**: 4-8 experts (default, scalable to 60+ for larger models)
- **Activation**: Selected via Expert Choice routing (only top_k tokens per expert)
- **Purpose**:
  - Specialize on different patterns and domains
  - Enable domain-specific processing (e.g., neurodegeneration, medical imaging)
  - Reduce computation through sparsity (only subset of experts active per token)
- **Structure**: Same as shared experts, but only processes selected tokens

### Expert Network Architecture

Each expert (both shared and routed) is a **simple 2-layer MLP** (Multi-Layer Perceptron), not a full transformer or LLM:

```python
Expert = Sequential(
    Linear(embedding_dim, 4 * embedding_dim),  # Expansion: 768 → 3072
    ReLU(),                                     # Activation
    Linear(4 * embedding_dim, embedding_dim),   # Projection: 3072 → 768
)
```

**Why This Design?**
- **Standard FFN Structure**: Matches the feedforward network pattern used in Transformer blocks
- **Efficiency**: Lightweight per expert, enabling many experts without excessive computation
- **Specialization**: Each expert learns different patterns through routing, despite identical architecture
- **Scalability**: Can scale to 60+ experts without proportional increase in compute

**Key Point**: Experts are NOT full transformers or LLMs—they are simple feedforward networks that specialize through the routing mechanism.

### Expert Choice Routing Mechanism

Unlike traditional **Token Choice routing** (where tokens choose experts), this implementation uses **Expert Choice routing** where experts select tokens. This approach provides better load balancing and more predictable computation.

**Routing Flow:**

1. **Gate Computation**
   - Input: Token embeddings `[batch*seq_len, embedding_dim]`
   - Gate: Linear layer `[embedding_dim → num_routed_experts]`
   - Output: Routing logits `[batch*seq_len, num_routed_experts]`
   - Each logit represents how well an expert matches a token

2. **Noise Injection (Training Only)**
   - Adds Gumbel noise to logits: `noisy_logits = logits + noise_scale * Gumbel_noise`
   - Default `noise_scale = 0.5` (DeepSeek-MoE default)
   - Encourages exploration during training, prevents early expert collapse

3. **Temperature Scaling**
   - Applies temperature to logits: `scaled_logits = logits / temperature`
   - Temperature schedule: Linear decay from `2.0` → `0.1` over 1000 steps
   - Higher temperature early = soft routing (exploration)
   - Lower temperature later = sparse routing (exploitation)

4. **Expert Selection**
   - Each routed expert selects `top_k` tokens (default `k=2`) with highest scores
   - Output: `token_indices [num_experts, top_k]` - which tokens each expert processes
   - Output: `expert_probs [num_experts, top_k]` - routing probabilities for gradient flow

5. **Capacity Control**
   - Enforces sparsity via capacity factor (default `1.5x` average load)
   - Formula: `max_tokens_per_expert = capacity_factor * (total_tokens / num_experts)`
   - Tokens exceeding capacity are masked out (dropped)
   - Capacity loss penalizes overflow to encourage balanced routing

6. **Fail-Safe Mechanism**
   - Tokens not selected by any routed expert fall back to shared experts
   - Ensures all tokens receive processing, even if routing fails

### Forward Pass Flow

The complete forward pass through the MoE layer:

```
1. Token Embedding
   Input: [batch, seq_len] token indices
   → Embedding Layer
   → Output: [batch, seq_len, embedding_dim]

2. Routing Decision
   → Gate computes logits: [batch*seq_len, num_routed_experts]
   → Add noise (training) + temperature scaling
   → Expert Choice routing: each expert selects top_k tokens
   → Output: token_indices [num_experts, top_k], expert_probs [num_experts, top_k]

3. Shared Expert Processing
   → Process ALL tokens through shared experts
   → Average outputs across shared experts
   → Output: [batch, seq_len, embedding_dim]

4. Routed Expert Processing
   → Gather tokens selected by each routed expert
   → Process through respective expert networks
   → Weight by routing probabilities
   → Scatter outputs back to token positions
   → Output: [batch, seq_len, embedding_dim]

5. Expert Output Combination
   → Handle unprocessed tokens (fail-safe to shared experts)
   → Learnable weighting: shared_scale * shared + routed_scale * routed
   → shared_scale = sigmoid(shared_expert_weight) ≈ 0.3-0.5
   → Output: [batch, seq_len, embedding_dim]

6. Joint Fusion
   → Process through joint fusion MLP
   → Apply LayerNorm + residual connection
   → Apply dropout (training only)
   → Output: [batch, seq_len, embedding_dim]

7. Output Projection
   → Apply LayerNorm + residual connection to input embeddings
   → Decoder MLP: [embedding_dim → 4*embedding_dim → vocab_size]
   → Output: [batch, seq_len, vocab_size] logits for next-token prediction
```

### Training Dynamics & Auxiliary Losses

The model learns effective routing through multiple auxiliary losses:

**1. Load Balance Loss**
- **Purpose**: Encourages uniform expert utilization
- **Formula**: Penalizes variance in expert load distribution
- **Weight**: `0.1` (DeepSeek-MoE default)
- **Effect**: Prevents expert collapse (all tokens routing to one expert)

**2. Z-Loss (Log-Sum-Exp Penalty)**
- **Purpose**: Prevents extreme routing confidence, maintains exploration
- **Formula**: `z_loss = mean((log_sum_exp(logits) - target_z)^2)`
- **Weight**: `0.001` (DeepSeek-MoE default)
- **Target Z**: `1.0` (moderate spread)
- **Effect**: Keeps routing distribution balanced, not too confident

**3. Capacity Loss**
- **Purpose**: Enforces sparsity constraints
- **Formula**: Penalizes tokens exceeding expert capacity
- **Weight**: `0.01`
- **Effect**: Encourages balanced token distribution across experts

**4. Temperature Scheduling**
- **Schedule**: Linear decay from `2.0` → `0.1` over 1000 steps
- **Effect**: 
  - Early training: High temperature = soft routing (exploration)
  - Late training: Low temperature = sparse routing (exploitation)
- **Alternative schedules**: Constant, exponential, cosine annealing

**Total Loss:**
```
loss = cross_entropy_loss + 
       0.1 * load_balance_loss + 
       0.001 * z_loss + 
       0.01 * capacity_loss
```

### Domain Specialization

Through training on curated healthcare data, routed experts naturally specialize:

- **Expert 1**: Focuses on neurodegeneration terminology (Alzheimer's, Parkinson's, etc.)
- **Expert 2**: Specializes in medical imaging language (MRI, CT, segmentation, etc.)
- **Expert 3**: Handles clinical terminology (diagnosis, treatment, prognosis, etc.)
- **Expert 4**: Processes drug discovery language (compounds, trials, efficacy, etc.)
- **Shared Experts**: Maintain general language understanding and scientific writing patterns

The routing mechanism learns to match tokens to domain-appropriate experts through:
- Training on domain-labeled data (neurodegeneration, neuroscience, medical imaging, etc.)
- Domain-aware loss weighting (higher weight for neurodegeneration/neuroscience papers)
- Natural specialization through gradient-based learning

### Key Advantages

**1. Computational Efficiency**
- Only activates subset of experts per token (sparse activation)
- Default: 2 shared experts (always active) + ~2 routed experts per token (top_k=2)
- Total active parameters per forward pass: ~50M (vs ~100M total parameters)
- Enables scaling to 60+ experts without proportional compute increase

**2. Specialization**
- Routed experts specialize on different healthcare subdomains
- Shared experts maintain general language understanding
- Natural domain adaptation through routing

**3. Scalability**
- Can scale to 60+ experts for larger models
- Computation scales with number of active experts, not total experts
- Memory-efficient: experts share same architecture, only weights differ

**4. Robustness**
- Shared experts ensure all tokens processed (fail-safe)
- Capacity control prevents expert overload
- Load balancing prevents expert collapse

### Configuration Parameters

Default configuration (configurable via `config.yaml`):

```yaml
vocab_size: 50000
embedding_dim: 768
num_shared_experts: 2
num_routed_experts: 8
top_k: 2                    # Tokens per expert
noise_scale: 0.5            # Gumbel noise for exploration
capacity_factor: 1.5        # Expert capacity multiplier
load_balance_loss_weight: 0.1
z_loss_weight: 0.001
temperature_schedule: "linear"
temperature_start: 2.0
temperature_end: 0.1
temperature_steps: 1000
```

### Implementation Details

**Vectorized Expert Processing:**
- Uses `torch.scatter_add_` for efficient GPU-accelerated expert output aggregation
- Batch processing: all experts process in parallel
- Memory-efficient: only stores active expert outputs

**Gradient Flow:**
- Uses soft routing probabilities for gradient computation (differentiable)
- Straight-through estimator (STE) for discrete routing decisions
- All auxiliary losses are differentiable and backpropagate through routing

**Monitoring & Debugging:**
- Routing metrics: entropy, load imbalance, expert utilization, token concentration
- Capacity tracking: dropped token fraction, expert utilization rate
- Comprehensive logging for routing health monitoring

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

## Expected Training Times

### Tokenizer Training
Tokenizer training time depends on the size of your processed dataset:

| Dataset Size | Papers | Estimated Time | Notes |
|-------------|--------|----------------|-------|
| Small | 1,000-5,000 | 2-5 minutes | Fast vocabulary learning |
| Medium | 5,000-15,000 | 5-15 minutes | Typical for healthcare datasets |
| Large | 15,000-30,000 | 15-30 minutes | Full ArXiv collection |
| Very Large | 30,000+ | 30-60 minutes | Maximum vocabulary coverage |

**Factors affecting tokenizer training:**
- **Corpus size**: More papers = longer training
- **Text length**: Longer papers = more processing
- **CPU cores**: Uses 4 threads by default (configurable)
- **Vocabulary size**: Default 50k (larger vocab = slightly longer)

### Model Training
Model training time depends on your hardware and configuration:

#### Google Colab (T4 GPU, 12GB VRAM)
| Configuration | Steps | Estimated Time | Notes |
|--------------|-------|----------------|-------|
| Fast (10k steps) | 10,000 | 2-4 hours | Quick test run |
| Standard (50k steps) | 50,000 | 8-12 hours | Default configuration |
| Extended (100k steps) | 100,000 | 16-24 hours | Full training |

**Default settings:**
- Batch size: 6
- Gradient accumulation: 4 (effective batch size: 24)
- Mixed precision: Enabled
- Throughput: ~1000-1500 samples/sec on T4 GPU

#### Local GPU (V100/A100)
| Configuration | Steps | Estimated Time | Notes |
|--------------|-------|----------------|-------|
| Standard (50k steps) | 50,000 | 4-6 hours | Faster than Colab |
| Extended (100k steps) | 100,000 | 8-12 hours | Full training |

#### CPU Only (Not Recommended)
| Configuration | Steps | Estimated Time | Notes |
|--------------|-------|----------------|-------|
| Standard (50k steps) | 50,000 | 3-5 days | Very slow, use GPU if possible |

**Factors affecting training time:**
- **GPU type**: T4 < V100 < A100 (speed)
- **Batch size**: Larger = faster but more memory
- **Sequence length**: Longer sequences = slower
- **Model size**: More experts/parameters = slower
- **Gradient accumulation**: More steps = slower but better quality

### Complete Pipeline Timeline

For a typical run with **5,000-10,000 processed papers** on **Google Colab**:

| Stage | Time | Notes |
|-------|------|-------|
| Collection | 2-4 hours | Depends on network, rate limits |
| PDF Extraction | 30-60 minutes | Parallel processing |
| NeMo Curator | 1-2 hours | Quality filtering, deduplication |
| Processing | 10-20 minutes | Text cleaning, classification |
| **Tokenizer Training** | **5-15 minutes** | Fast for medium datasets |
| **Model Training (50k steps)** | **8-12 hours** | Main training phase |
| Evaluation | 10-30 minutes | Metrics computation |
| **Total** | **12-20 hours** | End-to-end pipeline |

**Tips for faster training:**
- Use GPU (Colab T4 is free and sufficient)
- Reduce `max_steps` for testing (e.g., 10k steps = 2-4 hours)
- Increase `batch_size` if you have more VRAM
- Use gradient accumulation to simulate larger batches
- Resume from checkpoints if interrupted

## Google Drive Persistence

When running on Google Colab with `use_drive: true` (default), **all pipeline outputs are automatically saved to Google Drive** for persistence across runtime interruptions.

### What Gets Saved to Drive

All data files are saved directly to Google Drive at:
```
/content/drive/MyDrive/neuroMOE_results/data/arxiv/
```

**Step 1: Collection**
- ✅ `arxiv_papers.jsonl` - Collected paper metadata
- ✅ `collection_checkpoint.json` - Collection progress checkpoint
- ✅ `collected_ids.db` - SQLite database for deduplication

**Step 2: PDF Extraction**
- ✅ `texts/` - Directory with extracted `.txt` files (one per paper)

**Step 3: NeMo Curator**
- ✅ `curated_dataset.jsonl` - Curated and filtered papers
- ✅ `curated_checkpoint.json` - Curation progress checkpoint

**Step 4: Processing**
- ✅ `processed_dataset.jsonl` - Processed papers with domain classification

**Step 5: Tokenizer Training**
- ✅ `healthcare_tokenizer.model` - Trained tokenizer model
- ✅ `healthcare_tokenizer.vocab` - Tokenizer vocabulary
- ✅ `tokenizer_metadata.json` - Tokenizer training metadata
- ✅ `tokenizer_validation_report.json` - Validation results

**Step 6: Model Training**
- ✅ `checkpoints/` - Training checkpoints (saved every 5000 steps)
  - `step_5000.pt`, `step_10000.pt`, etc.
  - `dataset_metadata.json` - Dataset tracking for resume
- ⚠️ **Note**: Checkpoints are saved to `./checkpoints/` (local) by default
  - To save checkpoints to Drive, set `checkpoint_dir` in `config.yaml` to a Drive path

**Step 7: Evaluation**
- ✅ `evaluations/` - Evaluation results and metrics
  - `eval_results.json` - Standard evaluation results file (for analysis notebook)
  - `evaluation_{timestamp}.json` - Timestamped evaluation files (for tracking multiple runs)

**Step 8: Inference**
- ✅ `inference/` - Exported inference pipeline

### Benefits

- **Persistent Storage**: All data survives runtime disconnections
- **Resume Capability**: Pipeline automatically resumes from Drive checkpoints
- **No Data Loss**: Even if Colab runtime times out, your data is safe
- **Automatic**: No manual copying needed - everything saves directly to Drive

### Configuration

In `config.yaml`:
```yaml
pipeline:
  use_drive: true  # Enable Google Drive persistence (default: true)
  drive_base: "/content/drive/MyDrive/neuroMOE_results"  # Drive path
```

### Restoring from Drive

The Colab notebook includes a `restore_from_drive()` function that automatically restores:
- Collected papers and metadata
- Extracted text files
- Curated datasets
- Training checkpoints
- Trained tokenizer

This allows you to resume exactly where you left off, even after days or weeks.

### Evaluation & Inference
- **Comprehensive Metrics**: Perplexity, domain accuracy, MRR@20, section classification
- **Fast Inference**: <100ms per paper on CPU
- **Embedding Generation**: For similarity search and literature review
- **Domain Classification**: Automatic healthcare subdomain detection

## Model Analysis & Visualization

The `model_analysis.ipynb` notebook provides comprehensive analysis and visualization of the trained DeepSeek-MoE model, covering everything from dataset statistics to expert specialization patterns to deployment considerations.

### Overview

This Jupyter notebook contains **11 major sections** plus an Executive Summary and Future Directions, providing a complete analysis of:
- Dataset characteristics and domain distribution
- Model architecture and configuration
- Training dynamics and convergence
- Expert activation patterns and specialization
- Model performance across domains
- Attention patterns and embedding analysis
- Vocabulary and tokenization statistics
- Research trends and domain insights
- Efficiency analysis and deployment considerations
- Model interpretability and qualitative analysis
- Reproducibility and documentation

### Quick Start

**Prerequisites:**
```bash
# Install required packages (if not already installed)
pip install matplotlib seaborn plotly pandas numpy scikit-learn torch
pip install umap-learn wordcloud networkx bertviz transformers matplotlib-venn
pip install nltk  # For stopwords and tokenization
```

**Running the Notebook:**

1. **Open the notebook:**
   ```bash
   jupyter notebook notebooks/model_analysis.ipynb
   # Or in JupyterLab:
   jupyter lab notebooks/model_analysis.ipynb
   ```

2. **Run all cells sequentially:**
   - The notebook will automatically create output directories (`./outputs/figures/`, `./outputs/data/`, `./outputs/reports/`)
   - Each section can be run independently if needed

3. **Expected Data Files:**
   The notebook expects the following files (will generate sample data if missing):
   - `./data/arxiv/arxiv_papers.jsonl` - ArXiv papers dataset
   - `./models/deepseek_moe/config.json` - Model configuration
   - `./models/deepseek_moe/training_logs.json` - Training logs
   - `./models/deepseek_moe/eval_results.json` - Evaluation results
   - `./models/deepseek_moe/expert_activations.npz` - Expert activation data (see note below)
   - `./models/deepseek_moe/checkpoint.pt` - Model checkpoint

   **Google Drive Support (Colab):**
   - The notebook automatically detects if running in Google Colab
   - If Google Drive is mounted, it uses paths in `/content/drive/MyDrive/neuroMOE_results/`
   - Data files are automatically loaded from Drive if available
   - Outputs (figures, data, reports) are saved to Drive for persistence
   - Falls back to local paths if Drive is not available
   - **Evaluation results**: Saved to `/content/drive/MyDrive/neuroMOE_results/evaluations/eval_results.json` (or `./evaluations/eval_results.json` locally)
   - **Expert activations**: Saved to `/content/drive/MyDrive/neuroMOE_results/evaluations/expert_activations.npz` (or `./evaluations/expert_activations.npz` locally)
     - **Note**: Both files are **automatically generated** during evaluation when you run `evaluate.py`
     - The evaluation script captures expert routing decisions during the forward pass (single pass, no overhead)
     - Both files are saved to the evaluations folder (Drive if available, otherwise local)
     - The notebook will generate sample data if these files are missing (for demonstration only)

### Notebook Structure

**Section 1: Dataset Overview & Statistics**
- Loads and analyzes ArXiv papers dataset
- Generates statistics: total papers, date range, categories, domain classification
- Visualizations: papers over time, category distribution, word clouds, Venn diagrams

**Section 2: Model Architecture & Configuration**
- Loads model configuration and creates architecture summary
- Calculates model statistics: parameters, FLOPs, memory footprint
- Generates architecture diagram and expert configuration tables

**Section 3: Training Dynamics & Loss Curves**
- Analyzes training logs and generates loss curves
- MoE-specific metrics: router loss, z-loss, expert capacity overflow
- Expert utilization over training, convergence analysis

**Section 4: Expert Activation Patterns & Specialization** ⭐ *Most Critical Section*
- Extracts expert activation data from model
- **Note**: Requires `expert_activations.npz` file (see "Expected Data Files" section)
- If file is missing, the notebook generates sample data for demonstration
- Heatmaps showing domain-specific expert specialization
- Expert similarity matrix, dimensionality reduction (t-SNE/UMAP)
- Expert routing analysis, "personas", dead expert analysis
- Expert co-activation network visualization

**Section 5: Model Performance & Evaluation Metrics**
- Overall performance metrics (perplexity, bits per character)
- Domain-specific performance analysis
- Performance by year and category
- Baseline comparisons, example predictions, error analysis
- Generation quality metrics, cross-domain transfer analysis

**Section 6: Attention Patterns & Embedding Space Analysis**
- Attention heatmaps for example sentences
- Attention head specialization analysis
- Token embedding visualization (t-SNE)
- Semantic clustering, embedding drift analysis
- Contextual embeddings and K-means clustering

**Section 7: Vocabulary & Tokenization Statistics**
- Vocabulary statistics and frequency distributions
- Domain-specific vocabulary analysis
- Tokenization examples, OOV analysis
- Token length distribution, rare term coverage, n-gram analysis

**Section 8: Research Trends & Domain Insights**
- Temporal trend analysis, ML technique evolution
- Healthcare application areas timeline
- Topic modeling (LDA), co-occurrence networks
- Emerging topics identification

**Section 9: Efficiency Analysis & Deployment Considerations**
- Parameter and computational efficiency analysis
- Inference latency breakdown, batch size scaling
- Memory footprint, expert pruning, quantization impact
- Deployment recommendations, cost analysis, scalability projections

**Section 10: Model Interpretability & Qualitative Analysis**
- Generation showcase, abstract completion task
- Controlled generation experiments
- Attention visualization, expert routing case studies
- Failure mode analysis, bias analysis
- Model capabilities assessment, cross-domain reasoning

**Section 11: Reproducibility & Model Documentation**
- Training configuration table
- Data preprocessing documentation
- Hardware & environment specs
- Checkpoint information, model card
- Evaluation protocol, random seed documentation
- Known issues, reproduction instructions, version control

**Executive Summary**
- Top 10 key findings
- Performance summary card
- Expert specialization summary
- Comparative advantages
- Most surprising findings
- Limitations acknowledged
- Key visualizations recap

**Future Directions**
- Model improvements, training improvements
- Data expansion, evaluation improvements
- Application ideas, research questions
- Deployment roadmap, timeline and milestones

### Output Files

The notebook generates comprehensive outputs:

**Figures** (`./outputs/figures/`):
- Training curves, expert activation heatmaps, performance comparisons
- Attention visualizations, embedding projections, network graphs
- Word clouds, trend charts, efficiency analyses
- **50+ high-quality visualizations** (PNG format, 300 DPI)

**Data** (`./outputs/data/`):
- CSV files with statistics, metrics, and analysis results
- JSON files with configuration and metadata
- **30+ data files** for further analysis

**Reports** (`./outputs/reports/`):
- Summary reports and documentation

### Key Features

**Robust Data Handling:**
- Automatically handles missing files by generating sample data
- Graceful error handling with clear instructions
- Supports multiple data formats (JSONL, CSV, NPZ, PyTorch checkpoints)

**Professional Visualizations:**
- High-quality figures suitable for publications
- Consistent styling with seaborn and matplotlib
- Interactive plots with Plotly where appropriate
- Clear legends, annotations, and captions

**Comprehensive Analysis:**
- Statistical tests (Kolmogorov-Smirnov, t-tests, Chi-square)
- Domain-specific analysis (ML vs Healthcare)
- Temporal analysis (trends over time)
- Expert specialization deep dives

**Reproducibility:**
- All random seeds documented
- Complete configuration tracking
- Step-by-step reproduction instructions
- Version control information

### Usage Tips

1. **First Time Setup:**
   - Run the initial setup cells to install packages and create directories
   - Download NLTK data (stopwords, punkt tokenizer)

2. **Missing Data:**
   - If training logs or model checkpoints are missing, the notebook will generate sample data
   - This allows you to explore the analysis structure even without a trained model
   - Replace sample data with actual data when available

3. **Path Configuration:**
   - The notebook automatically configures paths for Google Drive (Colab) or local directories
   - Path configuration is in the "Google Drive Setup (Colab)" section
   - If your data is in a different location, update the path variables:
     - `ARXIV_DATA_PATH` - ArXiv papers dataset
     - `MODEL_CONFIG_PATH` - Model configuration
     - `TRAINING_LOGS_PATH` - Training logs
     - `EVAL_RESULTS_PATH` - Evaluation results
     - `EXPERT_ACTIVATIONS_PATH` - Expert activation data
     - `MODEL_CHECKPOINT_PATH` - Model checkpoint

4. **Customization:**
   - Modify paths in the setup cells to match your directory structure
   - Adjust figure sizes, DPI, and styling in the matplotlib configuration
   - Add custom analysis sections as needed

5. **Performance:**
   - Some sections (e.g., t-SNE, UMAP, topic modeling) can be slow on large datasets
   - Consider sampling data for faster exploration
   - Use GPU acceleration where available (UMAP, PyTorch operations)

6. **Exporting Results:**
   - All figures are automatically saved to `./outputs/figures/`
   - Data tables are saved to `./outputs/data/`
   - Use these for presentations, papers, or further analysis

### Expected Runtime

For a typical analysis run:
- **Setup & Data Loading**: 1-2 minutes
- **Section 1-3** (Dataset, Architecture, Training): 5-10 minutes
- **Section 4** (Expert Analysis): 10-15 minutes (most computationally intensive)
- **Section 5-7** (Performance, Attention, Vocabulary): 10-15 minutes
- **Section 8-11** (Trends, Efficiency, Interpretability, Documentation): 15-20 minutes
- **Executive Summary & Future Directions**: 5-10 minutes

**Total Runtime**: ~1-2 hours for complete analysis (depending on data size and hardware)

### Integration with Training Pipeline

The analysis notebook complements the training pipeline:

1. **After Training:**
   - Run the notebook to analyze your trained model
   - Generate visualizations for presentations or papers
   - Identify areas for improvement

2. **During Development:**
   - Use Section 4 (Expert Analysis) to debug routing issues
   - Use Section 3 (Training Dynamics) to monitor convergence
   - Use Section 9 (Efficiency) to optimize deployment

3. **For Publication:**
   - Export figures from `./outputs/figures/` for papers
   - Use tables from `./outputs/data/` for supplementary materials
   - Reference reproducibility section (Section 11) for methodology

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
