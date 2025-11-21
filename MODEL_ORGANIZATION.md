# Model Organization Guide

This document describes the recommended directory structure for organizing saved models to ensure easy loading in analysis notebooks and evaluation scripts.

## Recommended Directory Structure

```
checkpoints/
├── pretrained/                    # MoE model checkpoints
│   ├── step_5000.pt
│   ├── step_10000.pt
│   ├── step_15000.pt
│   ├── ...
│   └── step_50000.pt              # Final MoE checkpoint
│
├── baseline/                      # Baseline model checkpoints
│   ├── encoder/                   # Encoder-only (BERT-style) baseline
│   │   ├── baseline_encoder_step_5000.pt
│   │   ├── baseline_encoder_step_10000.pt
│   │   ├── ...
│   │   └── baseline_encoder_final.pt
│   │
│   └── decoder/                   # Decoder-only (GPT-style) baseline
│       ├── baseline_decoder_step_5000.pt
│       ├── baseline_decoder_step_10000.pt
│       ├── ...
│       └── baseline_decoder_final.pt
│
└── baseline/                      # Legacy: backward compatibility
    └── baseline_final.pt          # (if you have old baseline without model_type)

evaluations/
├── pretrained/                    # MoE model evaluation results
│   ├── eval_results.json
│   ├── expert_activations.npz
│   └── example_predictions.json
│
├── baseline_encoder/              # Encoder baseline evaluation results
│   ├── eval_results.json
│   └── baseline_encoder_predictions.json
│
├── baseline_decoder/              # Decoder baseline evaluation results
│   ├── eval_results.json
│   └── baseline_decoder_predictions.json
│
└── baseline_results.json          # Legacy: for backward compatibility
                                    # (can be symlink or copy of encoder results)
```

## Model File Naming Conventions

### MoE Models
- **Checkpoints**: `checkpoints/pretrained/step_{N}.pt` where N is the step number
- **Final Model**: `checkpoints/pretrained/step_50000.pt` (or your final step)
- **Evaluation Results**: `evaluations/pretrained/eval_results.json`

### Baseline Models

#### Encoder-Only (BERT-style)
- **Checkpoints**: `checkpoints/baseline/encoder/baseline_encoder_step_{N}.pt`
- **Final Model**: `checkpoints/baseline/encoder/baseline_encoder_final.pt`
- **Results**: `evaluations/baseline_encoder_results.json` (from training)
- **Evaluation**: `evaluations/baseline_encoder/eval_results.json` (from evaluation script)

#### Decoder-Only (GPT-style)
- **Checkpoints**: `checkpoints/baseline/decoder/baseline_decoder_step_{N}.pt`
- **Final Model**: `checkpoints/baseline/decoder/baseline_decoder_final.pt`
- **Results**: `evaluations/baseline_decoder_results.json` (from training)
- **Evaluation**: `evaluations/baseline_decoder/eval_results.json` (from evaluation script)

## Loading Models in Analysis Notebooks

### For MoE Models
```python
CHECKPOINT_BASE = Path("./checkpoints/pretrained")
MODEL_CHECKPOINT_PATH = CHECKPOINT_BASE / "step_50000.pt"
EVAL_BASE = Path("./evaluations/pretrained")
```

### For Baseline Models
```python
# Encoder baseline
ENCODER_CHECKPOINT = Path("./checkpoints/baseline/encoder/baseline_encoder_final.pt")
ENCODER_RESULTS = Path("./evaluations/baseline_encoder_results.json")

# Decoder baseline
DECODER_CHECKPOINT = Path("./checkpoints/baseline/decoder/baseline_decoder_final.pt")
DECODER_RESULTS = Path("./evaluations/baseline_decoder_results.json")
```

## Training Commands

### MoE Model
```bash
python train_colab.py \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --output-dir ./checkpoints \
    --max-steps 50000
```

### Encoder Baseline
```bash
python train_baseline.py \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluations \
    --checkpoint-dir ./checkpoints/baseline \
    --model-type encoder \
    --max-steps 50000
```

### Decoder Baseline
```bash
python train_baseline.py \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluations \
    --checkpoint-dir ./checkpoints/baseline \
    --model-type decoder \
    --max-steps 50000
```

## Evaluation Commands

### MoE Model
```bash
python evaluate.py \
    --model-checkpoint ./checkpoints/pretrained/step_50000.pt \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluations/pretrained
```

### Encoder Baseline
```bash
python evaluate.py \
    --model-checkpoint ./checkpoints/baseline/encoder/baseline_encoder_final.pt \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluations/baseline_encoder
```

### Decoder Baseline
```bash
python evaluate.py \
    --model-checkpoint ./checkpoints/baseline/decoder/baseline_decoder_final.pt \
    --dataset-text-dir ./data/arxiv/texts \
    --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
    --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
    --output-dir ./evaluations/baseline_decoder
```

## Backward Compatibility

If you have existing baseline models saved in the old format (`checkpoints/baseline/baseline_final.pt`), the evaluation script will automatically detect them. However, for clarity and to support both encoder and decoder baselines, we recommend:

1. **Rename old baseline**: If you have an old baseline, it's likely encoder-only. Rename it:
   ```bash
   mkdir -p checkpoints/baseline/encoder
   mv checkpoints/baseline/baseline_final.pt checkpoints/baseline/encoder/baseline_encoder_final.pt
   ```

2. **Create symlink for compatibility** (optional):
   ```bash
   cd evaluations
   ln -s baseline_encoder_results.json baseline_results.json
   ```

## Google Drive Organization (Colab)

When using Google Drive, the structure should be:

```
/content/drive/MyDrive/neuroMOE_results/
├── checkpoints/
│   ├── pretrained/
│   └── baseline/
│       ├── encoder/
│       └── decoder/
├── evaluations/
│   ├── pretrained/
│   ├── baseline_encoder/
│   └── baseline_decoder/
└── data/
    └── arxiv/
```

## Benefits of This Structure

1. **Clear Separation**: Each model type has its own directory
2. **Easy Loading**: Analysis notebooks can easily find models by type
3. **Scalable**: Easy to add more baseline types in the future
4. **Backward Compatible**: Old code still works with detection logic
5. **Organized**: Checkpoints, results, and evaluations are clearly separated

## Tips for Analysis Notebooks

When loading models in analysis notebooks, use path detection:

```python
# Auto-detect baseline type from checkpoint path
checkpoint_path = Path("./checkpoints/baseline/encoder/baseline_encoder_final.pt")
if "encoder" in str(checkpoint_path):
    model_type = "encoder"
elif "decoder" in str(checkpoint_path):
    model_type = "decoder"
else:
    model_type = "encoder"  # Default to encoder for backward compatibility
```

This ensures your notebooks work with both encoder and decoder baselines automatically.

