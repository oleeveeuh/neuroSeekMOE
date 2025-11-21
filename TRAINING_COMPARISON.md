# Training Process Comparison: Baseline vs MoE Models

This document compares the training configurations for baseline transformers and MoE models to help understand differences and ensure fair comparisons.

## Learning Rate Configuration

### Default Learning Rate
Both models use the **same default learning rate**: `5e-4` (0.0005)

### Learning Rate Schedule

Both models use **warmup + cosine annealing**, but with **slight differences**:

#### Baseline Transformers (`train_baseline.py`) - **UPDATED to match MoE**
```python
# Warmup: Linear from 10% to 100% of learning rate (matches MoE)
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,  # Starts at 10% of LR (matches MoE)
    end_factor=1.0,
    total_iters=2000
)

# Cosine annealing: Decay to 10% of learning rate
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max_steps - 2000,
    eta_min=learning_rate * 0.1  # Decays to 10% of LR
)
```

#### MoE Models (`train_colab.py`)
```python
# Warmup: Linear from 10% to 100% of learning rate
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,  # Starts at 10% of LR (10x higher than baseline!)
    end_factor=1.0,
    total_iters=2000
)

# Cosine annealing: Decay to 10% of learning rate
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max_steps - 2000,
    eta_min=learning_rate * 0.1  # Decays to 10% of LR
)
```

**Key Difference**: 
- Baseline starts warmup at **1% of LR** (more conservative)
- MoE starts warmup at **10% of LR** (more aggressive)

## Optimizer Configuration

### Baseline Transformers - **UPDATED to match MoE**
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate,  # Default: 5e-4
    weight_decay=0.01  # Matches MoE (stronger regularization)
)
```

### MoE Models
```python
optimizer = AdamW(
    model.parameters(), 
    lr=learning_rate,  # Default: 5e-4
    weight_decay=0.01   # Higher weight decay (1000x difference!)
)
```

**Key Difference**:
- Baseline: `weight_decay=1e-5` (0.00001)
- MoE: `weight_decay=0.01` (0.01)

This is a **1000x difference** in weight decay, which significantly affects regularization.

## Training Process Comparison

### Similarities ✅

1. **Same Learning Rate**: Both use `5e-4` by default
2. **Same Warmup Duration**: Both use 2000 warmup steps
3. **Same Cosine Decay**: Both decay to 10% of initial LR
4. **Same Optimizer Type**: Both use AdamW
5. **Same Batch Size**: Both default to batch_size=6
6. **Same Gradient Accumulation**: Both use 4 steps (effective batch size = 24)
7. **Same Max Steps**: Both default to 50,000 steps
8. **Same Gradient Clipping**: Both clip gradients at norm 1.0

### Differences ⚠️

| Aspect | Baseline Transformers | MoE Models |
|--------|----------------------|------------|
| **Weight Decay** | ✅ `0.01` (matches MoE) | ✅ `0.01` |
| **Warmup Start** | ✅ 10% of LR (matches MoE) | ✅ 10% of LR |
| **Mixed Precision** | ❌ No | ✅ Yes (GradScaler) |
| **Gradient Checkpointing** | ❌ No | ✅ Yes (if available) |
| **Auxiliary Losses** | ❌ No | ✅ Yes (load balance, z-loss, capacity) |

## Impact on Training

### ✅ Settings Now Aligned

As of the latest update, baseline training has been updated to match MoE settings:

- **Weight Decay**: Both use `0.01` - same regularization strength
- **Warmup Start**: Both use `10%` - same early learning behavior

This ensures:
- ✅ Same regularization strength (fair comparison)
- ✅ Same early training dynamics (fair comparison)
- ✅ Only architectural differences remain (MoE-specific features)

## ✅ Settings Aligned

**Status**: Baseline training has been updated to match MoE settings.

The following changes were made to `train_baseline.py`:

1. **Weight Decay**: Changed from `1e-5` to `0.01` (matches MoE)
2. **Warmup Start**: Changed from `0.01` (1%) to `0.1` (10%) (matches MoE)

This ensures a fair comparison where the only differences are:
- MoE-specific features (auxiliary losses, mixed precision)
- Architecture differences (dense vs sparse)

All other training hyperparameters are now identical.

## Current Training Commands

### Baseline (Current Settings - **Aligned with MoE**)
```bash
python train_baseline.py \
    --learning-rate 5e-4 \
    --max-steps 50000 \
    --batch-size 6 \
    --gradient-accumulation 4
    # Uses: weight_decay=0.01, warmup_start=0.1 (matches MoE)
```

### MoE (Current Settings)
```bash
python train_colab.py \
    --learning-rate 5e-4 \
    --max-steps 50000 \
    --batch-size 6 \
    --gradient-accumulation 4
    # Uses: weight_decay=0.01, warmup_start=0.1
```

## Summary

**Are they the same?** **Yes, now they match!** ✅

**As of the latest update**, baseline training has been aligned with MoE training settings:

**Aligned Settings:**
1. ✅ **Weight Decay**: Both use `0.01` (stronger regularization)
2. ✅ **Warmup Start**: Both start at `10%` of learning rate
3. ✅ **Learning Rate**: Both use `5e-4`
4. ✅ **Schedule**: Both use warmup (2000 steps) + cosine annealing

**Remaining Differences (Architectural):**
1. **Mixed Precision**: Only MoE uses it (GradScaler)
2. **Auxiliary Losses**: Only MoE has them (load balance, z-loss, capacity)
3. **Gradient Checkpointing**: Only MoE uses it (if available)

**For Fair Comparison:**
- ✅ Learning rate and schedule are now identical
- ✅ Regularization strength is now identical
- ✅ Warmup behavior is now identical
- The only differences are MoE-specific features (auxiliary losses, mixed precision)
- This ensures a fair comparison between dense and sparse architectures

**Status**: Baseline training now matches MoE training settings for optimal comparison fairness.

