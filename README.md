# NeuroSeek-MoE: Multimodal Mixture-of-Experts for Neurodegenerative Disease Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized DeepSeek-MoE variant that integrates text and image reasoning for neurodegenerative disease analysis, supporting Alzheimer's Disease (AD), Parkinson's Disease (PD), Amyotrophic Lateral Sclerosis (ALS), Huntington's Disease (HD), and Multiple Sclerosis (MS).

## 🧠 Purpose

NeuroSeek-MoE is designed to:

- **Accept multimodal queries**: Both text and image inputs for comprehensive disease analysis
- **Generate hybrid outputs**: Structured text explanations + automatically generated diagrams
- **Provide expert routing visualization**: Show which specialized experts contribute to each modality
- **Support 5 major neurodegenerative diseases**: AD, PD, ALS, HD, MS with disease-specific expert routing
- **Benchmark against baselines**: Compare performance against text-only LLMs and image-only diffusion models

## 🏗️ Architecture

### Model Overview

NeuroSeek-MoE uses a simplified, efficient Mixture-of-Experts architecture with a shared expert pool and sparse top-k routing. The model processes text tokens through embedding, routes to experts via a learned gate, and combines expert outputs for final prediction.

### Core Components

#### 1. **Embedding Layer**
- Converts token IDs (vocab size: 10,007) to dense embeddings (default: 128-dim)
- Mean pooling across sequence length to produce fixed-size representations

#### 2. **Shared Expert Pool**
- **Single unified expert pool**: All experts share the same architecture
- **Expert Architecture**: Standard 2-layer MLP with 4× expansion:
  ```
  Linear(embedding_dim → 4×embedding_dim) → ReLU → Linear(4×embedding_dim → embedding_dim)
  ```
- **Default**: 2 experts (configurable via `--num-experts`)
- **No modality-specific experts**: Single expert pool handles all inputs

#### 3. **Gating Network**
- **Simple linear gate**: `nn.Linear(embedding_dim → num_experts)`
- **Temperature scaling**: Applies temperature (default: 1.0) to logits for better routing
- **Top-k sparse routing**: Selects top-2 experts per sample (k=2)
- **Load balancing**: Entropy-based auxiliary loss encourages uniform expert utilization

#### 4. **Expert Combination**
- **Sparse activation**: Only computes outputs for selected top-k experts
- **Weighted sum**: Combines expert outputs using gate probabilities
- **On-the-fly computation**: Experts computed only when selected (efficient)

#### 5. **Joint Fusion & Decoder**
- **Joint fusion**: 2-layer MLP (4× expansion) with LayerNorm and residual connections
- **Decoder**: 2-layer MLP (4× expansion) outputs vocabulary logits
- **Residual connections**: Skip connections for better gradient flow

### Routing Mechanism

**Top-K Sparse Routing:**
1. Input representation → `pooled_text` (mean pooled embeddings)
2. Gate computes logits → `gate_logits = gate(pooled_text)` [batch, num_experts]
3. Apply temperature → `gate_logits / temperature`
4. Top-k selection → Select top-2 experts per sample
5. Compute only selected experts → Efficient sparse activation
6. Weighted combination → Sum of (expert_output × gate_probability)

**Load Balancing:**
- **Auxiliary loss**: Entropy-based loss on gate logits
- **Objective**: Maximize entropy of mean expert probabilities (encourage uniform usage)
- **Weight**: 0.01× aux_loss added to main cross-entropy loss
- **Logging**: Reports active experts per epoch (should approach num_experts)

### Key Features

- **Simplified Architecture**: Single shared expert pool (no separate text/image experts)
- **Sparse Activation**: Only top-k experts computed per sample (efficient)
- **Load Balancing**: Entropy-based auxiliary loss for uniform expert utilization
- **Standard FFN Design**: 4× expansion ratio (transformer-style feedforward)
- **Temperature Scaling**: Tunable temperature for sharper/softer routing
- **Residual Connections**: LayerNorm + residual in joint fusion for stability

### Training Architecture Details

**Model Parameters:**
- `vocab_size`: 10,007 (hash-based tokenization)
- `embedding_dim`: 128 (default)
- `num_experts`: 2 (default, configurable)
- `gate_temperature`: 1.0 (tunable hyperparameter)

**Regularization:**
- **Weight decay**: L2 regularization (1e-5)
- **Dropout**: 0.1 in expert networks and joint fusion
- **Early stopping**: Patience-based (default: 5 epochs)
- **Learning rate scheduling**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Gradient clipping**: Max norm 1.0

**Loss Function:**
- **Main loss**: CrossEntropyLoss (masked for padding tokens)
- **Auxiliary loss**: Entropy-based load balancing (0.01× weight)
- **Total**: `loss = main_loss + 0.01 × aux_loss`

## 📊 Dataset Pipeline

### Data Sources

**Text Data:**
- PubMed abstracts and PMC full-text
- Biomedical literature with disease-specific keywords

**Image Data:**
- NeuroVault neuroimaging data
- NIfTI format brain images
- Research paper figure panels

### Processing Pipeline

1. **Data Fetching**: Automated retrieval from biomedical APIs
2. **Text Processing**: Normalization, filtering, deduplication
3. **Image Processing**: Caption pairing and metadata extraction
4. **Multimodal Combination**: JSONL format with text/image pairs
5. **Caching**: Automatically skips already processed data

### Dataset Statistics

- **Text Records**: ~4,800 curated biomedical texts from PubMed
- **Image Records**: Up to 250 real biomedical images (NIfTI neuroimaging format)
- **Diseases**: Balanced across AD, PD, ALS, HD, MS
- **Note**: Datasets are cached - re-running skips already processed data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/neuroseek-moe.git
cd neuroseek-moe

# Install core dependencies
pip install -r requirements.txt

# Install metrics libraries (for BLEU, BERTScore, CLIPScore, FID)
pip install sacrebleu bert-score torch-fidelity

# Install model libraries (for transformers, CLIP, Stable Diffusion)
pip install git+https://github.com/openai/CLIP.git
pip install diffusers transformers accelerate sentence-transformers

# Optional: Install Jupyter for notebooks
pip install jupyter notebook

# Optional: CUDA support for GPU acceleration
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Complete Pipeline

Follow these steps to build, train, and evaluate the model:

#### 1. Build the Dataset

```bash
# Basic dataset (50 images per disease = 250 total)
python data_pipeline.py biomed-nemo-build \
  --staging ./staging \
  --processed ./processed \
  --max-text 1000 \
  --max-images 200 \
  --max-pages 10 \
  --tar-out ./image_shards \
  --shard-size 100

# This creates three files in ./processed/:
# - text_dataset.jsonl      (PubMed abstracts)
# - image_dataset.jsonl     (NeuroVault images)
# - multimodal_dataset.jsonl (Combined for training)

# Outputs will be cached for faster subsequent runs
```

**For Large-Scale Image Collection:**

```bash
# Calculate pages needed:
# pages = target_images / (50 images/page × 5 diseases)

# For 500 images (100 per disease):
python data_pipeline.py biomed-nemo-build \
  --staging ./staging \
  --processed ./processed \
  --max-images 100 \
  --max-pages 2

# For 1000 images (200 per disease):
python data_pipeline.py biomed-nemo-build \
  --staging ./staging \
  --processed ./processed \
  --max-images 200 \
  --max-pages 4

# This downloads ~1000 images across 5 diseases (4 pages × 50 images × 5 diseases)
```

**Note:** Many NeuroVault URLs fail to download. The counter shows attempted downloads. The actual number of successfully downloaded images will be less than requested.

#### 2. Training

See the comprehensive [Training Section](#-training) below for detailed instructions.

#### 3. Dataset Structure

The pipeline creates properly curated datasets:

```bash
processed/
├── text_dataset.jsonl          # PubMed abstracts
├── image_dataset.jsonl         # NeuroVault images
└── multimodal_dataset.jsonl    # Combined for training

# Each line is a JSON object with:
# - "text": "..." (for text records)
# - "image_path": "..." (for image records)
# - "caption": "..." (for image records)
# - "disease": "AD|PD|ALS|HD|MS"
# - "modality": "text|image"
# - "source": "..."
```

All datasets are automatically cached. Re-running skips already processed data.

#### 4. Quick Start with Colab (Recommended)

For easy testing and experimentation, use the complete Colab notebook:

```bash
# Open the notebook in Google Colab
# notebooks/NeuroSeek_MoE_Complete_Pipeline.ipynb

# The notebook includes:
# - Environment setup (dependencies, GPU check)
# - Data pipeline execution
# - Model training
# - Visualization and analysis
# - Results download
```

**Features:**
- ✅ Complete pipeline in one notebook
- ✅ GPU support (free on Colab)
- ✅ Automatic dependency installation
- ✅ Built-in visualization
- ✅ Google Drive integration for persistence

#### 5. Training Visualization

Use the Jupyter notebook to visualize training progress and metrics:

#### 6. Notebook Analysis

Use Jupyter notebooks for detailed analysis and visualization:

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

**Available Notebooks:**

- `notebooks/training_visualization.ipynb` - Visualize training curves, loss progression, BERTScore tracking, and expert configuration comparisons

#### 6. Training Visualization

**Jupyter Notebook (Post-training analysis):**
```bash
# Visualize training curves and statistics
cd notebooks
jupyter notebook training_visualization.ipynb

# Features:
# - Loss curves over epochs (train vs test)
# - BLEU and BERTScore tracking (train vs test)
# - Training statistics (best epochs, improvements)
# - Overfitting detection (train/test gap analysis)
# - Expert configuration comparison visualization
# - Metric computation explanations
```

#### 7. Training Visualization

Analyze training progress and compare expert configurations:

```bash
# Open the training visualization notebook
cd notebooks
jupyter notebook training_visualization.ipynb

# Features:
# - Loss curves over epochs (train vs test)
# - BERTScore tracking
# - Per-configuration loss curves (for compare_expert_configs.py results)
# - Auxiliary load-balancing loss tracking
# - Active experts per epoch tracking
```

## 📋 Data Curation Features

### Text Data
- **Source**: PubMed abstracts
- **Keywords**: Disease-specific biomedical terms
- **Processing**: NeMo-style curation (if installed)
- **Output**: `processed/text_dataset.jsonl`

### Image Data  
- **Source**: NeuroVault API
- **Format**: NIfTI (.nii.gz) neuroimaging files
- **Search**: Disease-specific terms with robust pagination
- **Deduplication**: Automatic URL deduplication
- **Output**: `processed/image_dataset.jsonl`

### Combined Dataset
- **Purpose**: Training the multimodal MoE model
- **Format**: Combined text + image JSONL
- **Output**: `processed/multimodal_dataset.jsonl`

## 🎓 Training and Testing Explained

### How Model Training Works

**1. Dataset Structure:**
- The pipeline loads data from JSONL files containing multimodal records
- Each record has a `modality` field: `"text"` or `"image"`
- **Text records** contain: `{"text": "...", "disease": "AD", "modality": "text"}`
- **Image records** contain: `{"image_path": "...", "caption": "...", "disease": "AD", "modality": "image"}`
- Currently, **all data is used for training** (no train/test split by default)
- Data is shuffled during training via PyTorch's `DataLoader(shuffle=True)`

**2. Training Process:**
```
1. Load Dataset → Parse JSONL files → Separate text/image records
2. Create Batches → Group records into batches (default: 8 samples)
3. Forward Pass → Model processes input tokens/images → Generates predictions
4. Compute Loss → Compare predictions to targets (next-token prediction)
5. Backward Pass → Calculate gradients via backpropagation
6. Update Weights → Optimizer (Adam) adjusts model parameters
7. Repeat → For each epoch, iterate through all batches
8. Checkpoint → Save model state after each epoch
```

**3. Training Objective:**
- **Task**: Next-token prediction (language modeling)
- **Input**: Text tokens (hashed words → integer IDs 0-10006)
- **Target**: Next token in sequence
- **Loss**: CrossEntropyLoss (ignores padding tokens) + 0.01× auxiliary load-balancing loss
- **Metrics**: BERTScore computed every 100 batches (BLEU removed)
- **Auxiliary Loss**: Entropy-based load balancing encourages uniform expert utilization

**4. Model Architecture:**
- **Embedding Layer**: Converts token IDs to dense vectors (128-dim), mean pooling
- **Shared Expert Pool**: 2 experts (default, configurable) with standard 2-layer MLP (4× expansion)
- **Gating Network**: Simple linear layer routes to top-k experts (k=2)
- **Joint Fusion**: 2-layer MLP with LayerNorm and residual connections
- **Decoder**: 2-layer MLP outputs vocabulary logits

### MoE Routing Mechanism Explained

**Expert Configuration:**
- **Single shared expert pool**: All experts share the same architecture
- Default configuration: 2 experts in the shared pool
- **No modality-specific experts**: All experts can handle any input type
- Experts are selected via learned gating network

**Top-K Sparse Routing Process:**
```
1. Input Processing:
   - Text tokens → Embedding → Mean pooling → Fixed-size representation
   - Input representation: [batch, embedding_dim]

2. Gating Network:
   - Computes routing scores: gate_logits = gate(pooled_text) [batch, num_experts]
   - Applies temperature scaling: gate_logits / temperature
   - Selects top-k experts: top-2 experts per sample (k=2, hardcoded)
   
3. Expert Selection & Computation:
   - Only selected experts are computed (sparse activation)
   - For each sample: computes outputs for top-2 selected experts
   - Weighted combination: Sum of (expert_output × gate_probability)

4. Output Combination:
   - Combined expert outputs → Joint fusion → Decoder → Vocabulary logits
```

**Routing Implementation:**
- **Single linear gate**: `nn.Linear(embedding_dim → num_experts)`
- **Top-k selection**: `top_k_gating()` function selects top-2 experts
- **Sparse computation**: Only computes selected experts (efficient)
- **Load balancing**: Entropy-based auxiliary loss encourages uniform expert usage

**Key Features:**
- **Sparse activation**: Only top-2 experts computed per sample (reduces computation)
- **Efficient**: On-the-fly expert computation (no pre-computation of all experts)
- **Adaptive routing**: Gating network learns which experts to use for each input
- **Load balancing**: Entropy loss ensures all experts are utilized

**Visualization:**
- Training logs show: Active experts per epoch (should approach num_experts)
- Auxiliary loss tracking: Reports load-balancing effectiveness
- Checkpoint files: Store all metrics for visualization in notebooks

**5. Testing/Evaluation:**
- **80/20 train/test split** automatically applied in `train_real.py`
- Training metrics computed on training set
- **Test metrics computed on held-out test set** after each epoch
- Both train and test metrics saved in checkpoints and results
- Test metrics used for model selection in expert comparison
- See [Training Section](#-training) for detailed instructions

### Dataset Sectioning

**Current Implementation:**
- **Automatic 80/20 train/test split** using PyTorch's `random_split`
- Fixed random seed (42) for reproducibility
- Data flow:
```
processed/multimodal_dataset.jsonl
  ↓
Load all records
  ↓
Separate by modality: text_ds[] and image_ds[]
  ↓
Create NeurodegenerativeDataset (pairs text + image)
  ↓
Random Split (80% train, 20% test) [seed=42]
  ↓
Train DataLoader (shuffled) → Training loop
Test DataLoader → Evaluation after each epoch
```

**Evaluation Process:**
- After each training epoch:
  - Model switches to evaluation mode (`model.eval()`)
  - Runs inference on test set (no gradients)
  - Computes test loss, BLEU, and BERTScore
  - Compares train vs test metrics to monitor overfitting
  - Switches back to training mode for next epoch

**Metrics Tracked:**
- **Training metrics**: Loss, BLEU, BERTScore (on training data)
- **Test metrics**: Loss, BLEU, BERTScore (on test data)
- Both saved in checkpoints and final results
- Expert comparison uses **test metrics** for model selection

## 📝 Training

NeuroSeek-MoE provides multiple training modes. **Real PyTorch training** is recommended for actual model learning, while the inference scaffold is useful for routing visualization and architecture demonstration.

---

### 1. Real PyTorch Training (Recommended)

**File**: `train_real.py` - Performs actual neural network training with learnable parameters

#### Quick Start

```bash
# Train with automatic GPU/CPU selection (default learning rate: 0.0001)
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --outputs outputs \
  --checkpoints checkpoints \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --device auto
```

#### Key Features

- ✅ **Real parameter learning**: PyTorch optimizers (Adam), loss functions (CrossEntropyLoss), backward pass
- ✅ **Simplified MoE architecture**: Single shared expert pool with sparse top-k routing
- ✅ **Load balancing**: Entropy-based auxiliary loss for uniform expert utilization
- ✅ **GPU/CPU support**: Automatic device detection or explicit selection (`auto`, `cpu`, `cuda`)
- ✅ **Checkpointing**: Saves model and optimizer state after each epoch
- ✅ **Resume training**: Continue from saved checkpoints with `--resume-from EPOCH`
- ✅ **Metrics tracking**: BERTScore computed every 100 batches (BLEU removed)
- ✅ **Auxiliary loss logging**: Reports load-balancing loss and active experts per epoch
- ✅ **80/20 train/test split**: Automatic dataset splitting with evaluation on test set
- ✅ **Progress monitoring**: Real-time loss, auxiliary loss, and batch progress display
- ✅ **Regularization**: Weight decay, dropout, early stopping, learning rate scheduling, gradient clipping

#### Device Selection

```bash
# Automatic (recommended)
python train_real.py --multimodal-jsonl processed/multimodal_dataset.jsonl --device auto

# Explicit CPU
python train_real.py --multimodal-jsonl processed/multimodal_dataset.jsonl --device cpu

# Explicit GPU
python train_real.py --multimodal-jsonl processed/multimodal_dataset.jsonl --device cuda --batch-size 16
```

#### Resume from Checkpoint

```bash
# Resume from epoch 5
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --epochs 20 \
  --resume-from 5

# Automatically detects latest checkpoint if --resume-from not specified
```

#### Train with Different Expert Counts

```bash
# Default: 2 experts in shared pool
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --epochs 10

# Train with 4 experts in shared pool
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --num-experts 4 \
  --epochs 10

# Train with 8 experts (more capacity)
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --num-experts 8 \
  --epochs 10 \
  --batch-size 4  # May need smaller batch size for more experts
```

**Expert Configuration:**
- `--num-experts N`: Number of experts in the shared pool (default: 2)
- **All experts share the same architecture**: No modality-specific experts
- **Top-k routing**: Selects top-2 experts per sample (k=2, hardcoded)
- **Load balancing**: Entropy-based loss encourages uniform expert utilization

**Note:** The model uses a single shared expert pool. All experts have the same architecture and can handle any input. The gating network learns to route inputs to the most appropriate experts.

#### Full Options

```bash
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --text-jsonl processed/text_dataset.jsonl \          # Alternative: separate files
  --image-jsonl processed/image_dataset.jsonl \         # Alternative: separate files
  --results evaluation/results.json \                   # Results output path
  --outputs outputs \                                   # Output directory
  --checkpoints checkpoints \                           # Checkpoint directory
  --epochs 10 \                                         # Number of epochs
  --batch-size 8 \                                      # Batch size
  --learning-rate 0.0001 \                            # Learning rate (default: 0.0001)
  --device auto \                                       # Device: auto, cpu, or cuda
  --disable-diagrams \                                  # Disable diagram generation
  --resume-from 5 \                                     # Resume from epoch (optional)
  --num-experts 2 \                                     # Number of experts in shared pool (default: 2)
  --vocab-path outputs/vocabulary.json                 # (optional) Vocabulary file path
  --early-stopping-patience 5 \                        # Early stopping patience (default: 5)
```

---

### 2. Compare Expert Configurations (Recommended Workflow)

**File**: `compare_expert_configs.py` - Trains multiple expert configurations and selects the best

This script automates the process of finding the optimal number of experts by training multiple configurations and comparing their performance on the test set.

#### Quick Start

```bash
# Compare different expert counts (1, 2, 4, 8 experts in shared pool)
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 1,2,4,8 \
  --epochs 10 \
  --selection-metric combined

# Compare fewer configurations for quick testing
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 2,4 \
  --epochs 5 \
  --selection-metric combined
```

**Expert Configuration Format:**
- `"N"` → N experts in shared pool (e.g., `"2"` = 2 experts, `"4"` = 4 experts)
- All experts share the same architecture and can handle any input
- The model uses top-k routing (k=2) to select experts per sample

#### What It Does

1. Trains models with different expert counts (1, 2, 3, 4, etc.)
2. **Auto-detects and resumes** from checkpoints if training was interrupted
3. Compares performance metrics (loss, BLEU, BERTScore) on test set
4. Selects the best configuration based on chosen metric
5. Saves results for visualization and evaluation
6. Evaluation notebook automatically uses best config

#### Resume Features

```bash
# Auto-resume from latest checkpoint (automatic detection)
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 1,2,3,4 \
  --epochs 10

# Manually resume all configurations from epoch 5
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 1,2,3,4 \
  --epochs 10 \
  --resume-from 5

# Resume only a specific configuration
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 2 \
  --epochs 10 \
  --resume-config 2expert \
  --resume-from 7
```

#### Selection Metrics

- `loss`: Lowest test loss (best for minimizing error)
- `bleu`: Highest test BLEU score (best for text quality)
- `bertscore`: Highest test BERTScore (best for semantic similarity)
- `combined`: Weighted combination of all metrics (default, recommended)

#### Full Options

```bash
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 1,2,3,4 \           # Expert counts to test (comma-separated)
  --epochs 10 \                         # Epochs per configuration
  --batch-size 8 \                      # Batch size
  --learning-rate 0.001 \              # Learning rate
  --device auto \                       # Device: auto, cpu, or cuda
  --selection-metric combined \         # Selection metric
  --resume-from 5 \                     # (optional) Resume from epoch
  --resume-config 2expert \              # (optional) Resume specific config
  --outputs-dir outputs \               # Base outputs directory
  --checkpoints-dir checkpoints \        # Base checkpoints directory
  --comparison-output evaluation/expert_comparison.json \  # Comparison results
  --best-config-output evaluation/best_config.json \      # Best config info
  --disable-diagrams                    # Disable diagram generation
```

#### Output Files

- **Individual results**: `outputs/config_{N}expert_results.json` - Metrics for each configuration
- **Comparison summary**: `evaluation/expert_comparison.json` - All configurations compared
- **Best config info**: `evaluation/best_config.json` - Used automatically by evaluation notebook
- **Model checkpoints**: `checkpoints/config_{N}expert/model_epoch_{N}.pt` - Per-config checkpoints

#### Example Output

```
🔬 Expert Configuration Comparison
======================================================================
Configurations to test: [1, 2, 3, 4]
Selection metric: combined

📊 Dataset Split:
   Total samples: 3000
   Train samples: 2400 (80%)
   Test samples: 600 (20%)

🔬 Training Configuration: 1expert
   Experts: 1 text + 1 image = 2 total
...
✅ Configuration 1expert complete!
   Train Loss: 4.23
   Test Loss:  4.45
   Test BLEU:  0.28
   Test BERTScore:  0.82

🏆 Best configuration by COMBINED TEST SCORE: 2expert
   Combined Score: 0.7234
   Test Loss: 3.45
   Test BLEU: 0.32
   Test BERTScore: 0.85
```

#### Tips

- Start with fewer configurations (e.g., `--expert-configs 1,2`) for quick testing
- Use `--disable-diagrams` for faster training
- Monitor test metrics - they determine the best configuration
- Check `evaluation/expert_comparison.json` for detailed metrics
- Visualize results using `notebooks/training_visualization.ipynb`

---

**Note:** The visualization notebook (`training_visualization.ipynb`) is fully compatible with the new simplified architecture:
- ✅ Works with single shared expert pool
- ✅ Loads checkpoints from `checkpoints/` or `checkpoints/config_*/`
- ✅ Handles missing BLEU scores gracefully (BLEU removed from training)
- ✅ Displays auxiliary loss and active expert counts from training logs
- ✅ Compatible with expert comparison results (`compare_expert_configs.py`)

---

### Training Workflow: When to Use What?

| Task | Use | Why |
|------|-----|-----|
| **Learn model parameters** | `train_real.py` | Real PyTorch training with optimizers |
| **Find best expert count** | `compare_expert_configs.py` | Automated comparison and selection |
| **Visualize training progress** | `notebooks/training_visualization.ipynb` | Training curves, metrics, and comparisons |

## 📊 Training Visualization

The training visualization notebook provides comprehensive analysis of training progress:

### Features

- **Loss Curves**: Train vs test loss over epochs
- **BERTScore Tracking**: Semantic similarity metrics over time
- **Expert Configuration Comparison**: Compare multiple expert counts side-by-side
- **Auxiliary Loss Tracking**: Monitor load-balancing effectiveness
- **Active Experts Tracking**: See how many experts are actually utilized
- **Per-Configuration Plots**: Separate loss curves for each expert configuration

### Usage

1. Train models using `train_real.py` or `compare_expert_configs.py`
2. Open the notebook: `cd notebooks && jupyter notebook training_visualization.ipynb`
3. Run all cells to generate visualizations
4. View saved plots: `training_curves.png`, `per_config_loss_curves.png`, `expert_config_comparison.png`

## 📈 Performance Metrics

### Training Results

| Metric | Value |
|--------|-------|
| Text BLEU | 0.0007 |
| Text BERTScore | 0.5003 |
| Image CLIPScore | 0.4981 |
| Routing Entropy | 0.6819 |

### Baseline Comparison

| Model | Text BLEU | Image CLIPScore |
|-------|-----------|-----------------|
| NeuroSeek-MoE | 0.0007 | 0.4981 |
| BioGPT (text-only) | 0.3000 | N/A |
| Stable Diffusion/BLIP | N/A | 0.3500 |

## 📓 Notebook Analysis

### Starting Jupyter

```bash
# Install Jupyter if needed
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Available Notebooks

**`notebooks/training_visualization.ipynb`** - Comprehensive training analysis
   - Loss curves over epochs (train vs test)
   - BERTScore tracking
   - Per-configuration loss curves (for expert comparison results)
   - Auxiliary load-balancing loss tracking
   - Active experts per epoch tracking
   - Expert configuration comparison visualizations

### Notebook Usage

1. Train models using `train_real.py` or `compare_expert_configs.py`
2. Open the notebook: `cd notebooks && jupyter notebook training_visualization.ipynb`
3. Run all cells to generate visualizations
4. View saved plots in the project root directory

## 📁 Project Structure

```
neuroseek-moe/
├── notebooks/
│   └── training_visualization.ipynb  # Training curves and metrics visualization
├── processed/                   # Processed datasets
│   ├── text_dataset.jsonl
│   ├── image_dataset.jsonl
│   └── multimodal_dataset.jsonl
├── staging/                     # Staging area for downloads
├── outputs/                     # Generated outputs and visualizations
├── checkpoints/                 # Model checkpoints (per epoch)
├── evaluation/                  # Training results and metrics
├── data_pipeline.py             # Main data processing pipeline
├── model_architecture.py       # Model architecture (Disease enum, utilities)
├── train_real.py               # Real PyTorch training script
├── compare_expert_configs.py   # Expert configuration comparison
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🤖 Real Model Implementations

This project uses **real** neural network models (not stubs):

### Text Encoding
- **Primary**: BioBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`)
- **Fallback**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Library**: `transformers` or `sentence-transformers`

### Image Encoding
- **Primary**: OpenAI CLIP (`ViT-B/32`)
- **Fallback**: CLIP via transformers (`openai/clip-vit-base-patch32`)
- **Library**: `clip` or `transformers`

### Image Generation
- **Primary**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- **Fallback**: Stable Diffusion v1.4 (`CompVis/stable-diffusion-v1-4`)
- **Library**: `diffusers`

### Evaluation Metrics
- **BLEU**: `sacrebleu` (real corpus-level BLEU scores)
- **BERTScore**: `bert-score` (semantic similarity via BERT)
- **CLIPScore**: `CLIP` model (image-text alignment)
- **FID**: `torch-fidelity` (Fréchet Inception Distance)

All models have graceful fallbacks if libraries are unavailable.

## 🔬 Research Applications

- **Clinical Decision Support**: Assist healthcare professionals with disease analysis
- **Medical Education**: Generate educational content with visual aids
- **Research Acceleration**: Rapid analysis of multimodal biomedical data
- **Drug Discovery**: Identify patterns in disease progression and treatment response

## 🛠️ Development

### Training and Evaluation

Training is performed using `train_real.py` which provides:
1. **Real PyTorch training**: Actual parameter updates with optimizers
2. **Expert configuration comparison**: Use `compare_expert_configs.py` to find optimal expert count
3. **Visualization**: Use `notebooks/training_visualization.ipynb` to analyze results

### Customizing Data Sources

The pipeline is configurable via command-line arguments:
- `--max-text`: Number of text documents per disease
- `--max-images`: Number of images to download per disease
- `--max-pages`: Pages to fetch from NeuroVault (50 images/page)
- Caching: Automatically skips already processed data

### Key Parameters

- **Text Fetching**: PubMed abstracts, biomedical keywords
- **Image Fetching**: NeuroVault with disease-specific search terms
- **Search Terms**: Comprehensive disease-specific keywords (Alzheimer's, dementia, amyloid, etc.)
- **Download Accuracy**: Counts only reflect successfully downloaded images

## 📚 Citation

```bibtex
@software{neuroseek_moe_2024,
  title={NeuroSeek-MoE: Multimodal Mixture-of-Experts for Neurodegenerative Disease Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/neuroseek-moe},
  note={Built with DeepSeek-MoE architecture}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek** for the MoE architecture inspiration
- **NeuroVault** for research neuroimaging datasets
- **PubMed** for biomedical literature access

## ⚙️ Complete Example

Here's a full workflow from dataset creation to training:

```bash
# Step 1: Build the multimodal dataset
python data_pipeline.py biomed-nemo-build \
  --staging ./staging \
  --processed ./processed \
  --max-text 1000 \
  --max-images 200 \
  --max-pages 10 \
  --tar-out ./image_shards \
  --shard-size 100

# Step 2: Check the generated dataset
ls -lh processed/*.jsonl

# Step 3: Train the model (or compare expert configurations)
# Option A: Train single configuration
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --outputs outputs \
  --checkpoints checkpoints \
  --epochs 10 \
  --batch-size 8 \
  --device auto

# Option B: Compare expert configurations (recommended)
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 1,2,3,4 \
  --epochs 10

# Step 4: View results
cat evaluation/results.json | python -m json.tool
```

## ⚙️ Known Issues

### Image Download Success Rate

- **Issue**: Many NeuroVault URLs fail to download
- **Impact**: Actual downloaded images may be fewer than requested
- **Solution**: The counter now reflects successful downloads only

### Dataset Caching

- **Feature**: Datasets are automatically cached
- **Behavior**: Re-running skips already processed data
- **Override**: Delete processed files to force regeneration

---

**NeuroSeek-MoE**: Advancing neurodegenerative disease research through multimodal AI 🧠✨
