# NeuroSeek-MoE: Multimodal Mixture-of-Experts for Neurodegenerative Disease Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized Mixture-of-Experts model for multimodal neurodegenerative disease analysis, supporting Alzheimer's Disease (AD), Parkinson's Disease (PD), ALS, Huntington's Disease (HD), and Multiple Sclerosis (MS).

## 🧠 Overview

NeuroSeek-MoE integrates text and image reasoning for comprehensive disease analysis:
- **Multimodal Input**: Text queries and neuroimaging data
- **Shared Expert Pool**: Single unified expert architecture with sparse top-k routing
- **Load Balancing**: Entropy-based auxiliary loss for uniform expert utilization
- **5 Diseases**: AD, PD, ALS, HD, MS with disease-specific training data

## 🏗️ Architecture

### Model Components

**Shared Expert Pool** (default: 2 experts)
- Standard 2-layer MLP with 4× expansion: `Linear(dim → 4×dim) → ReLU → Linear(4×dim → dim)`
- All experts share same architecture (no modality-specific separation)

**Gating Network**
- Linear gate: `nn.Linear(embedding_dim → num_experts)`
- Top-k sparse routing: Selects top-2 experts per sample (k=2)
- Temperature scaling for better routing control

**Routing Process**
1. Input → Embedding → Mean pooling → Fixed-size representation
2. Gate computes logits → Apply temperature → Top-k selection
3. Compute only selected experts (sparse activation)
4. Weighted combination → Joint fusion → Decoder → Vocabulary logits

**Load Balancing**
- Auxiliary loss: Entropy-based (maximize entropy of expert probabilities)
- Weight: 0.01× aux_loss added to main cross-entropy loss
- Logs active experts per epoch

### Training Details

- **Regularization**: Weight decay (1e-5), dropout (0.1), early stopping (patience=5), LR scheduling, gradient clipping
- **Loss**: `CrossEntropyLoss + 0.01 × entropy_loss`
- **Split**: 80/20 train/test (automatic)
- **Metrics**: BERTScore (BLEU removed)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/neuroseek-moe.git
cd neuroseek-moe

# Core dependencies
pip install -r requirements.txt

# Metrics and models
pip install sacrebleu bert-score torch-fidelity
pip install git+https://github.com/openai/CLIP.git
pip install diffusers transformers accelerate sentence-transformers

# Optional: NeMo Curator (recommended for advanced data curation)
pip install "nemo-curator[all]"

# Optional: Jupyter for notebooks
pip install jupyter notebook
```

### Option 1: Colab Notebook (Recommended)

For easy experimentation:
- Open `notebooks/NeuroSeek_MoE_Complete_Pipeline.ipynb` in Google Colab
- Complete pipeline: setup → data → training → visualization
- GPU support included

### Option 2: Local Setup

**1. Build Dataset**
```bash
python data_pipeline.py biomed-nemo-build \
  --staging ./staging \
  --processed ./processed \
  --max-text 500 \
  --max-images 50 \
  --max-pages 2
```

**2. Train Model**
```bash
# Single configuration
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --device auto

# Compare expert configurations (recommended)
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 1,2,4 \
  --epochs 10
```

**3. Visualize Results**
```bash
cd notebooks
jupyter notebook training_visualization.ipynb
```

## 📊 Data Pipeline

### Data Sources

- **Text**: PubMed abstracts (filtered for Reviews, Meta-Analyses; excludes Case Reports, Letters)
- **Images**: NeuroVault neuroimaging data (NIfTI format)

### Processing Options

**NeMo Curator (Optional, Recommended)**
- **Text**: Advanced curation (normalize, dedupe, PII redaction) when available
- **Image**: Enhanced image pipeline when available
- **Fallback**: Basic processing if NeMo Curator not installed

**Article Quality Filters** (default enabled):
- Minimum abstract length (default: 200 chars)
- Preferred types: Review, Meta-Analysis, Systematic Review
- Excluded types: Case Reports, Letter, Editorial, Retracted Publication
- Optional: Minimum publication year filter

### Dataset Structure

```
processed/
├── text_dataset.jsonl          # PubMed abstracts
├── image_dataset.jsonl         # NeuroVault images
└── multimodal_dataset.jsonl    # Combined for training
```

Each line is JSON: `{"text": "...", "disease": "AD", "modality": "text"}` or `{"image_path": "...", "caption": "...", "disease": "AD", "modality": "image"}`

### Pipeline Options

```bash
# Basic dataset
python data_pipeline.py biomed-nemo-build \
  --staging ./staging \
  --processed ./processed \
  --max-text 500 \
  --max-images 50

# With article quality filters
python data_pipeline.py biomed-nemo-build \
  --staging ./staging \
  --processed ./processed \
  --max-text 500 \
  --article-types "Review" "Meta-Analysis" \
  --exclude-types "Case Reports" "Letter" \
  --min-year 2015 \
  --min-abstract-length 300
```

**Note**: Datasets are cached automatically. Re-running skips already processed data.

## 📝 Training

### Basic Training

```bash
python train_real.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --outputs outputs \
  --checkpoints checkpoints \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --device auto \
  --early-stopping-patience 5
```

### Expert Configuration Comparison

```bash
# Compare multiple expert counts
python compare_expert_configs.py \
  --multimodal-jsonl processed/multimodal_dataset.jsonl \
  --expert-configs 1,2,4,8 \
  --epochs 10 \
  --selection-metric combined
```

**Outputs**:
- `checkpoints/config_{N}expert/` - Per-configuration checkpoints
- `evaluation/expert_comparison.json` - Comparison results
- `evaluation/best_config.json` - Best configuration selected

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num-experts` | Number of experts in shared pool | 2 |
| `--learning-rate` | Learning rate | 0.0001 |
| `--early-stopping-patience` | Early stopping patience | 5 |
| `--resume-from` | Resume from epoch | None |
| `--device` | Device (auto/cpu/cuda) | auto |

## 📊 Visualization

### Training Visualization Notebook

```bash
cd notebooks
jupyter notebook training_visualization.ipynb
```

**Features**:
- Loss curves (train vs test) over epochs
- BERTScore tracking
- Per-configuration loss curves (for expert comparison)
- Auxiliary load-balancing loss tracking
- Active experts per epoch
- Expert configuration comparison plots

**Outputs**: `training_curves.png`, `per_config_loss_curves.png`, `expert_config_comparison.png`

## 📁 Project Structure

```
neuroseek-moe/
├── notebooks/
│   ├── NeuroSeek_MoE_Complete_Pipeline.ipynb  # Complete Colab pipeline
│   └── training_visualization.ipynb            # Training analysis
├── processed/                   # Processed datasets (JSONL)
├── checkpoints/                 # Model checkpoints
│   └── config_*/               # Per-configuration checkpoints
├── evaluation/                  # Training results
├── data_pipeline.py             # Data processing pipeline
├── train_real.py               # Training script
├── compare_expert_configs.py   # Expert comparison
├── model_architecture.py       # Model architecture
└── requirements.txt            # Dependencies
```

## 🔧 Configuration

### Data Pipeline Filters

**Article Quality Filters** (default enabled):
- `--min-abstract-length`: Minimum abstract length (default: 200)
- `--article-types`: Preferred types (default: Review, Meta-Analysis, Systematic Review)
- `--exclude-types`: Excluded types (default: Case Reports, Letter, Editorial, Retracted Publication)
- `--min-year`: Minimum publication year (optional)

### Training Parameters

- **Model**: `vocab_size=10007`, `embedding_dim=128`, `num_experts=2` (configurable)
- **Regularization**: Weight decay (1e-5), dropout (0.1), gradient clipping (max_norm=1.0)
- **Optimization**: Adam optimizer, LR scheduler (ReduceLROnPlateau), early stopping

## 📚 Reference

### Model Implementations

- **Text Encoding**: BioBERT / sentence-transformers
- **Image Encoding**: OpenAI CLIP (ViT-B/32)
- **Metrics**: BERTScore, sacrebleu, torch-fidelity

### Citation

```bibtex
@software{neuroseek_moe_2024,
  title={NeuroSeek-MoE: Multimodal Mixture-of-Experts for Neurodegenerative Disease Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/neuroseek-moe}
}
```

## ⚠️ Known Issues

- **NeuroVault Downloads**: Some URLs fail to download (normal behavior)
- **Dataset Caching**: Datasets cached automatically; delete files to force regeneration

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek** for MoE architecture inspiration
- **NeuroVault** for neuroimaging datasets
- **PubMed** for biomedical literature access

---

**NeuroSeek-MoE**: Advancing neurodegenerative disease research through multimodal AI 🧠✨
