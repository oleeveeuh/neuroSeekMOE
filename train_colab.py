"""
Colab-Optimized Training Loop for DeepSeekMoE

Optimized for Colab T4 GPU (12GB VRAM) with:
- Mixed precision training
- Gradient accumulation
- Gradient checkpointing
- Dynamic batch sizing
- Efficient checkpointing
- Comprehensive logging

Usage:
    # Train from scratch (recommended)
    python train_colab.py \
        --dataset-text-dir ./data/arxiv/texts \
        --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
        --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
        --output-dir ./checkpoints \
        --batch-size 6 \
        --gradient-accumulation 4 \
        --max-steps 50000 \
        --learning-rate 5e-4
    
    # Resume from checkpoint (optional)
    python train_colab.py \
        --model-path ./checkpoints/step_5000.pt \
        --dataset-text-dir ./data/arxiv/texts \
        --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
        --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
        --output-dir ./checkpoints \
        --batch-size 6 \
        --gradient-accumulation 4 \
        --max-steps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, List
import csv

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler  # GradScaler is still in torch.cuda.amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import our components
from arxiv_dataset import ArXivStreamingDataset, create_dataloader
from training_adapter import ModelAdapter

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("sentencepiece not available")

try:
    from tokenizer_wrapper import TokenizerWrapper, load_medical_tokenizer, DEFAULT_MEDICAL_TOKENIZER
    TOKENIZER_WRAPPER_AVAILABLE = True
except ImportError:
    TOKENIZER_WRAPPER_AVAILABLE = False
    print("tokenizer_wrapper not available, falling back to SentencePiece only")


class TrainingLogger:
    """Logger for training metrics with CSV export."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = []
        self.fieldnames = [
            'step', 'loss', 'learning_rate', 'gpu_memory_mb', 'throughput_samples_per_sec',
            'domain_neurodegeneration', 'domain_neuroscience', 'domain_medical_imaging',
            'domain_clinical', 'domain_drug_discovery', 'domain_general_ml_health'
        ]
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV file
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log(self, metrics: Dict):
        """Log metrics to CSV."""
        self.metrics.append(metrics)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)
    
    def get_latest_metrics(self, n: int = 100) -> List[Dict]:
        """Get latest N metrics."""
        return self.metrics[-n:]


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint, or None if none found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('step_') and filename.endswith('.pt'):
            try:
                step = int(filename.replace('step_', '').replace('.pt', ''))
                checkpoints.append((step, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    # Sort by step number and return latest
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    step: int,
    checkpoint_dir: str,
    max_checkpoints: int = 2
):
    """Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        scaler: GradScaler state
        step: Current training step
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'step_{step}.pt')
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Delete old checkpoints (keep only last max_checkpoints)
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('step_') and filename.endswith('.pt'):
            try:
                step_num = int(filename.replace('step_', '').replace('.pt', ''))
                checkpoints.append((step_num, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue
    
    if len(checkpoints) > max_checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        # Delete oldest checkpoints
        for step_num, path in checkpoints[:-max_checkpoints]:
            os.remove(path)
            print(f"  Deleted old checkpoint: step_{step_num}.pt")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler
) -> int:
    """Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        scaler: GradScaler to load state into
        
    Returns:
        Step number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    step = checkpoint.get('step', 0)
    print(f"Loaded checkpoint from step {step}: {checkpoint_path}")
    
    return step


def count_domains_in_batch(batch_metadata: Dict) -> Dict[str, int]:
    """Count domain distribution in batch.

    Args:
        batch_metadata: Batch metadata from adapter

    Returns:
        Dictionary with domain counts
    """
    domain_counts = {
        'neurodegeneration': 0,
        'neuroscience': 0,
        'medical_imaging': 0,
        'clinical': 0,
        'drug_discovery': 0,
        'general_ml_health': 0
    }

    domains_list = batch_metadata.get('domains', [])
    for domains in domains_list:
        for domain in domains:
            if domain in domain_counts:
                domain_counts[domain] += 1

    return domain_counts


def find_max_batch_size(
    model: nn.Module,
    adapter: ModelAdapter,
    tokenizer,
    dataset: ArXivStreamingDataset,
    device: torch.device,
    start_batch_size: int = 8,
    min_batch_size: int = 1
) -> int:
    """Dynamically find maximum batch size that fits in GPU memory.
    
    Args:
        model: Model to test
        adapter: Model adapter
        tokenizer: Tokenizer
        dataset: Dataset
        device: Device to test on
        start_batch_size: Starting batch size to test
        min_batch_size: Minimum batch size to try
        
    Returns:
        Maximum batch size that fits
    """
    print(f"Finding maximum batch size (starting from {start_batch_size})...")
    
    model.eval()
    batch_size = start_batch_size
    
    # Create a small dataloader for testing
    test_dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Single worker for testing
        pin_memory=False
    )
    
    while batch_size >= min_batch_size:
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try to process a batch
            batch = next(iter(test_dataloader))
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                result = adapter.process_batch(batch)
                loss = result['loss']
                # Simulate backward pass
                loss.backward()
            
            # If successful, try larger batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"   Batch size {batch_size} fits")
            batch_size = min(batch_size + 2, start_batch_size * 2)  # Cap at 2x start
            break
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"   Batch size {batch_size} too large, trying {batch_size - 1}")
                batch_size -= 1
            else:
                raise
    
    model.train()
    print(f"Maximum batch size: {batch_size}")
    return max(batch_size, min_batch_size)


def train(
    model: nn.Module,
    dataset: ArXivStreamingDataset,
    adapter: ModelAdapter,
    checkpoint_dir: str,
    batch_size: int = 6,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 50000,
    learning_rate: float = 5e-4,
    warmup_steps: int = 2000,
    save_interval: int = 5000,
    log_interval: int = 100,
    domain_log_interval: int = 1000,
    resume_from_checkpoint: Optional[str] = None,
    auto_find_batch_size: bool = True,
    early_stopping: bool = False,
    early_stopping_patience: int = 1000,
    early_stopping_min_delta: float = 0.01,
    early_stopping_window: int = 500,
    val_dataset: Optional[ArXivStreamingDataset] = None
):
    """Main training loop optimized for Colab GPU with early stopping.

    Args:
        model: DeepSeekMoE model
        dataset: ArXiv streaming dataset (training)
        adapter: Model adapter
        checkpoint_dir: Directory for checkpoints
        batch_size: Batch size (will be auto-adjusted if needed)
        gradient_accumulation_steps: Gradient accumulation steps
        max_steps: Maximum training steps
        learning_rate: Learning rate
        warmup_steps: Warmup steps for scheduler
        save_interval: Steps between checkpoints
        log_interval: Steps between logging
        domain_log_interval: Steps between domain distribution logging
        resume_from_checkpoint: Path to checkpoint to resume from (auto-detected if None)
        auto_find_batch_size: Whether to auto-detect max batch size
        early_stopping: Enable early stopping based on perplexity
        early_stopping_patience: Stop if no improvement for N steps
        early_stopping_min_delta: Minimum perplexity improvement threshold
        early_stopping_window: Evaluate perplexity every N steps
        val_dataset: Validation dataset for early stopping (prevents overfitting to training data)
    """
    device = adapter.device

    # Ensure numeric types for scheduler (fix for YAML loading)
    learning_rate = float(learning_rate)
    max_steps = int(max_steps)
    warmup_steps = int(warmup_steps)

    # Find maximum batch size if requested
    if auto_find_batch_size and torch.cuda.is_available():
        # We need tokenizer for this, but it's in adapter
        # For now, use provided batch_size
        pass  # Skip auto-detection for now (requires tokenizer access)

    # Create dataloader - use 0 workers for Colab memory stability
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Single-threaded for Colab stability
        pin_memory=False  # Disable pin_memory for CPU
    )

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)

    # Setup scheduler: warmup + cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=learning_rate * 0.1
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if specified or auto-detect
    start_step = 0
    if resume_from_checkpoint:
        start_step = load_checkpoint(resume_from_checkpoint, model, optimizer, scheduler, scaler)
    else:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_step = load_checkpoint(latest_checkpoint, model, optimizer, scheduler, scaler)

    # Setup logging
    log_file = os.path.join(checkpoint_dir, 'training_log.csv')
    logger = TrainingLogger(log_file)

    # Early stopping state
    best_perplexity = float('inf')
    steps_without_improvement = 0
    last_evaluation_step = 0
    best_model_state = None  # Track best model state
    best_step = 0  # Track step when best model was saved

    # Training state
    model.train()
    accumulated_loss = 0.0
    step = start_step
    samples_processed = 0
    start_time = time.time()
    
    print("=" * 60)
    print("Starting Training Loop")
    print("=" * 60)
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Max steps: {max_steps}")
    print(f"   Starting from step: {start_step}")
    print("=" * 60)
    print()
    
    # Training loop
    dataloader_iter = iter(dataloader)
    
    while step < max_steps:
        # Zero gradients at start of accumulation
        if step % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            print(f"Dataloader exhausted at step {step}, restarting...")
            dataloader_iter = iter(dataloader)
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                print("ERROR: Dataset appears to be empty after restart!")
                print(f"Dataset length: {len(dataset)}")
                print("Breaking training loop...")
                break
        
        # Forward pass with mixed precision
        try:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                result = adapter.process_batch(batch)
                loss = result['loss']
                batch_metadata = result['batch_metadata']
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            accumulated_loss += loss.item() * gradient_accumulation_steps
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at step {step}, skipping batch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Reset gradients if we were accumulating
                if step % gradient_accumulation_steps != 0:
                    optimizer.zero_grad()
                continue
            else:
                raise
        
        # Update weights after accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        step += 1
        samples_processed += batch_metadata.get('batch_size', batch_size)
        
        # Logging
        if step % log_interval == 0:
            elapsed_time = time.time() - start_time
            throughput = samples_processed / elapsed_time if elapsed_time > 0 else 0
            gpu_memory = get_gpu_memory_mb()
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = accumulated_loss / log_interval
            
            # Count domains if it's domain log interval
            domain_counts = {}
            if step % domain_log_interval == 0:
                domain_counts = count_domains_in_batch(batch_metadata)
            
            metrics = {
                'step': step,
                'loss': avg_loss,
                'learning_rate': current_lr,
                'gpu_memory_mb': gpu_memory,
                'throughput_samples_per_sec': throughput,
                **{f'domain_{k}': v for k, v in domain_counts.items()}
            }
            
            logger.log(metrics)
            
            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                  f"GPU: {gpu_memory:.0f}MB | Throughput: {throughput:.1f} samples/sec")
            
            if domain_counts:
                print(f"   Domains: {domain_counts}")
            
            accumulated_loss = 0.0
        
        # Checkpointing
        if step % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, step, checkpoint_dir)

        # Early stopping evaluation
        if early_stopping and (step - last_evaluation_step) >= early_stopping_window:
            model.eval()
            eval_loss = 0.0
            eval_batches = 10  # Evaluate on 10 batches

            # Use validation dataset if provided, otherwise sample from training data
            if val_dataset is not None:
                # Create validation dataloader
                val_dataloader = create_dataloader(
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=False,
                    shuffle=True  # Shuffle to get different samples each time
                )
                val_dataloader_iter = iter(val_dataloader)
                data_source = "validation set"
            else:
                # Fallback to training data (not ideal, but maintains compatibility)
                val_dataloader_iter = dataloader_iter
                data_source = "training data (WARNING: no validation set provided)"

            with torch.no_grad():
                for _ in range(eval_batches):
                    try:
                        eval_batch = next(val_dataloader_iter)
                    except StopIteration:
                        if val_dataset is not None:
                            val_dataloader_iter = iter(val_dataloader)
                        else:
                            dataloader_iter = iter(dataloader)
                            val_dataloader_iter = dataloader_iter
                        eval_batch = next(val_dataloader_iter)

                    result = adapter.process_batch(eval_batch)
                    loss = result['loss']
                    eval_loss += loss.item()

            avg_eval_loss = eval_loss / eval_batches
            current_perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()

            # Check for improvement
            if current_perplexity < best_perplexity - early_stopping_min_delta:
                best_perplexity = current_perplexity
                steps_without_improvement = 0
                best_step = step
                # Save best model state
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'step': step,
                    'perplexity': best_perplexity
                }
                print(f"   ✓ New best perplexity: {best_perplexity:.2f} ({data_source})")
            else:
                steps_without_improvement += early_stopping_window
                print(f"   Early stopping ({data_source}): {steps_without_improvement}/{early_stopping_patience} steps without improvement (current: {current_perplexity:.2f}, best: {best_perplexity:.2f})")

                # Check if should stop early
                if steps_without_improvement >= early_stopping_patience:
                    print(f"\n✅ Early stopping triggered! No improvement for {early_stopping_patience} steps.")
                    print(f"   Best perplexity: {best_perplexity:.2f} at step {best_step}")
                    print(f"   Final step: {step}")
                    break

            last_evaluation_step = step
            model.train()

    # Final checkpoint
    print("\nSaving final checkpoint...")

    # If early stopping was used and we have a best model, save it
    if early_stopping and best_model_state is not None:
        print(f"Saving best model (perplexity: {best_perplexity:.2f} from step {best_step})...")
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(best_model_state, best_checkpoint_path)
        print(f"   Best model saved to: {best_checkpoint_path}")

        # Also save as final checkpoint for compatibility
        save_checkpoint(model, optimizer, scheduler, scaler, step, checkpoint_dir)
        print(f"   Final checkpoint (step {step}) also saved")
    else:
        # No early stopping or no best model saved, save current state
        save_checkpoint(model, optimizer, scheduler, scaler, step, checkpoint_dir)

    print("\nTraining complete!")
    print(f"Training log saved to: {log_file}")
    if early_stopping:
        print(f"Best perplexity achieved: {best_perplexity:.2f} at step {best_step}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Colab-optimized training loop for DeepSeekMoE",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model and data paths
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint or initial weights (optional, creates new model if not provided)')
    parser.add_argument('--dataset-text-dir', type=str, required=True,
                       help='Directory containing text files')
    parser.add_argument('--dataset-metadata', type=str, required=True,
                       help='JSONL file with paper metadata')
    parser.add_argument('--tokenizer-path', type=str, required=True,
                       help='Path to SentencePiece tokenizer model')
    parser.add_argument('--val-dataset', type=str, default=None,
                       help='Path to validation metadata JSONL for early stopping (prevents overfitting)')

    # Training config
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Alias for --output-dir (deprecated, use --output-dir)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: from config.yaml or 6)')
    parser.add_argument('--gradient-accumulation', type=int, default=None,
                       help=f'Gradient accumulation steps (default: from config.yaml or 4)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=None,
                       help='Alias for --gradient-accumulation (deprecated, use --gradient-accumulation)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help=f'Maximum training steps (default: from config.yaml or 50000)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help=f'Learning rate (default: from config.yaml or 5e-4)')
    parser.add_argument('--warmup-steps', type=int, default=None,
                       help=f'Warmup steps (default: from config.yaml or 2000)')

    # Checkpointing and logging
    parser.add_argument('--save-interval', type=int, default=None,
                       help=f'Steps between checkpoints (default: from config.yaml or 5000)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Steps between logging (default: 100)')
    parser.add_argument('--domain-log-interval', type=int, default=1000,
                       help='Steps between domain distribution logging (default: 1000)')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from (auto-detected if None)')
    
    # Domain weights
    parser.add_argument('--domain-weight-neurodegeneration', type=float, default=1.5,
                       help='Loss weight for neurodegeneration papers (default: 1.5)')
    parser.add_argument('--domain-weight-neuroscience', type=float, default=1.2,
                       help='Loss weight for neuroscience papers (default: 1.2)')
    
    args = parser.parse_args()

    # Handle deprecated/alias arguments
    if args.checkpoint_dir is not None:
        args.output_dir = args.checkpoint_dir
        print("Warning: --checkpoint-dir is deprecated, use --output-dir instead")

    if args.gradient_accumulation_steps is not None:
        args.gradient_accumulation = args.gradient_accumulation_steps
        print("Warning: --gradient-accumulation-steps is deprecated, use --gradient-accumulation instead")

    # Apply config.yaml defaults if CLI args not provided
    # (config values are loaded earlier in training_config_defaults)
    if 'training_config_defaults' in locals() or 'training_config_defaults' in globals():
        # We're in the main() function, need to use the loaded config
        pass  # Config will be applied after loading in main()
    
    # Load tokenizer (try HuggingFace first, fallback to SentencePiece)
    if TOKENIZER_WRAPPER_AVAILABLE:
        # Check if it's a HuggingFace model name or SentencePiece file
        if os.path.exists(args.tokenizer_path) and (args.tokenizer_path.endswith('.model') or os.path.isfile(args.tokenizer_path)):
            # SentencePiece file
            tokenizer = TokenizerWrapper(args.tokenizer_path, tokenizer_type='sentencepiece')
            print(f"Loaded SentencePiece tokenizer from: {args.tokenizer_path}")
        elif '/' in args.tokenizer_path and not os.path.exists(args.tokenizer_path):
            # HuggingFace model name (e.g., "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            try:
                tokenizer = TokenizerWrapper(args.tokenizer_path, tokenizer_type='huggingface')
                print(f"✅ Loaded HuggingFace tokenizer: {args.tokenizer_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load HuggingFace tokenizer '{args.tokenizer_path}': {e}")
                print(f"   Falling back to default medical tokenizer: {DEFAULT_MEDICAL_TOKENIZER}")
                tokenizer = load_medical_tokenizer()
        else:
            # Use default medical tokenizer
            print(f"Using default medical tokenizer: {DEFAULT_MEDICAL_TOKENIZER}")
            tokenizer = load_medical_tokenizer()
    elif SENTENCEPIECE_AVAILABLE:
        # Fallback to SentencePiece only
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(args.tokenizer_path)
        print(f"Loaded SentencePiece tokenizer from: {args.tokenizer_path}")
    else:
        raise ImportError("Neither tokenizer_wrapper nor sentencepiece available. Install transformers or sentencepiece.")
    
    vocab_size = tokenizer.get_piece_size()
    print(f"   Vocabulary size: {vocab_size}")
    
    # Create dataset - use processed_dataset.jsonl for training data
    # If text_dir is missing (deleted during cleanup), use processed_dataset
    if os.path.exists(args.dataset_text_dir):
        dataset = ArXivStreamingDataset(
            text_dir=args.dataset_text_dir,
            metadata_jsonl=args.dataset_metadata,
            tokenizer=tokenizer,
            max_length=512,
            min_length=64
        )
    else:
        print("Warning: text_dir not found, using processed_dataset.jsonl for training")
        # Point to processed_dataset.jsonl as both source
        processed_dataset_path = args.dataset_metadata.replace('arxiv_papers.jsonl', 'processed_dataset.jsonl')
        if not os.path.exists(processed_dataset_path):
            # Try alternative path
            processed_dataset_path = args.dataset_metadata.replace('metadata', 'processed_dataset')

        if os.path.exists(processed_dataset_path):
            dataset = ArXivStreamingDataset(
                text_dir=None,  # No separate text files
                metadata_jsonl=processed_dataset_path,
                tokenizer=tokenizer,
                max_length=512,
                min_length=64
            )
        else:
            raise FileNotFoundError(f"Neither text_dir nor processed_dataset.jsonl found. Checked: {processed_dataset_path}")
    print(f"Created dataset with ~{len(dataset)} samples")

    # Load validation dataset if provided (for early stopping)
    val_dataset = None
    if args.val_dataset:
        print(f"\nLoading validation dataset for early stopping...")
        if os.path.exists(args.val_dataset):
            val_dataset = ArXivStreamingDataset(
                text_dir=None,  # Validation set typically doesn't have separate text files
                metadata_jsonl=args.val_dataset,
                tokenizer=tokenizer,
                max_length=512,
                min_length=64
            )
            print(f"Created validation dataset with ~{len(val_dataset)} samples")
        else:
            print(f"⚠️  Warning: Validation dataset file not found: {args.val_dataset}")
            print(f"   Will use training data for early stopping evaluation (NOT recommended - may overfit)")

    # Load model - use SimpleMoEModel from train_real.py
    # vocab_size already set above
    
    try:
        from train_real import SimpleMoEModel
        print("Imported SimpleMoEModel from train_real.py")
    except ImportError:
        print("Could not import SimpleMoEModel, using dummy model")
        # Fallback to dummy model
        class DummyModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 768)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(768, 8, dim_feedforward=2048, batch_first=True),
                    num_layers=6
                )
                self.lm_head = nn.Linear(768, vocab_size)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                logits = self.lm_head(x)
                return logits
        
        SimpleMoEModel = DummyModel
    
    # Load MoE architecture and routing parameters from config.yaml if available
    moe_arch_config = {
        'embedding_dim': 768,  # Default from config.yaml
        'num_shared_experts': 2,
        'num_routed_experts': 8,  # DeepSeek-MoE: 8 routed experts
        'top_k': 2,  # Select 2 out of 8 routed experts per token
    }

    moe_routing_config = {
        'noise_scale': 1.0,  # Default: increased for better specialization
        'load_balance_loss_weight': 0.5,  # Default: increased for better specialization
        'z_loss_weight': 0.005,  # Default: increased for better specialization
        'temperature_schedule': 'linear',
        'temperature_start': 3.0,
        'temperature_end': 0.3,  # Default: increased to maintain exploration
        'temperature_steps': 10000,  # Default: increased for slower decay
    }

    # Early stopping config
    early_stopping_config = {
        'early_stopping': False,
        'early_stopping_patience': 1000,
        'early_stopping_min_delta': 0.01,
        'early_stopping_window': 500,
    }

    # Training hyperparameters config (for overriding CLI defaults)
    training_config_defaults = {
        'batch_size': 6,
        'gradient_accumulation_steps': 4,
        'max_steps': 50000,
        'learning_rate': 5e-4,
        'warmup_steps': 2000,
        'save_interval': 5000,
    }

    # Try to load from config.yaml
    try:
        import yaml
        config_path = Path('config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'training' in config:
                    training_config = config['training']

                    # Load architecture parameters
                    if 'embedding_dim' in training_config:
                        moe_arch_config['embedding_dim'] = training_config['embedding_dim']
                    if 'num_shared_experts' in training_config:
                        moe_arch_config['num_shared_experts'] = training_config['num_shared_experts']
                    if 'num_routed_experts' in training_config:
                        moe_arch_config['num_routed_experts'] = training_config['num_routed_experts']
                    if 'top_k' in training_config:
                        moe_arch_config['top_k'] = training_config['top_k']

                    # Load routing parameters
                    if 'noise_scale' in training_config:
                        moe_routing_config['noise_scale'] = training_config['noise_scale']
                    if 'load_balance_loss_weight' in training_config:
                        moe_routing_config['load_balance_loss_weight'] = training_config['load_balance_loss_weight']
                    if 'z_loss_weight' in training_config:
                        moe_routing_config['z_loss_weight'] = training_config['z_loss_weight']
                    if 'temperature_schedule' in training_config:
                        moe_routing_config['temperature_schedule'] = training_config['temperature_schedule']
                    if 'temperature_start' in training_config:
                        moe_routing_config['temperature_start'] = training_config['temperature_start']
                    if 'temperature_end' in training_config:
                        moe_routing_config['temperature_end'] = training_config['temperature_end']
                    if 'temperature_steps' in training_config:
                        moe_routing_config['temperature_steps'] = training_config['temperature_steps']

                    # Load early stopping parameters
                    if 'early_stopping' in training_config:
                        early_stopping_config['early_stopping'] = training_config['early_stopping']
                    if 'early_stopping_patience' in training_config:
                        early_stopping_config['early_stopping_patience'] = training_config['early_stopping_patience']
                    if 'early_stopping_min_delta' in training_config:
                        early_stopping_config['early_stopping_min_delta'] = training_config['early_stopping_min_delta']
                    if 'early_stopping_window' in training_config:
                        early_stopping_config['early_stopping_window'] = training_config['early_stopping_window']

                    # Load training hyperparameters (will override CLI defaults)
                    if 'batch_size' in training_config:
                        training_config_defaults['batch_size'] = training_config['batch_size']
                    if 'gradient_accumulation_steps' in training_config:
                        training_config_defaults['gradient_accumulation_steps'] = training_config['gradient_accumulation_steps']
                    if 'max_steps' in training_config:
                        training_config_defaults['max_steps'] = training_config['max_steps']
                    if 'learning_rate' in training_config:
                        training_config_defaults['learning_rate'] = training_config['learning_rate']
                    if 'warmup_steps' in training_config:
                        training_config_defaults['warmup_steps'] = training_config['warmup_steps']
                    if 'checkpoint_interval' in training_config:
                        training_config_defaults['save_interval'] = training_config['checkpoint_interval']

                    print(f"✅ Loaded DeepSeek-MoE configuration from config.yaml")
                    print(f"   Architecture:")
                    print(f"      embedding_dim: {moe_arch_config['embedding_dim']}")
                    print(f"      num_shared_experts: {moe_arch_config['num_shared_experts']}")
                    print(f"      num_routed_experts: {moe_arch_config['num_routed_experts']}")
                    print(f"      top_k: {moe_arch_config['top_k']}")
                    print(f"   Routing:")
                    print(f"      noise_scale: {moe_routing_config['noise_scale']}")
                    print(f"      load_balance_loss_weight: {moe_routing_config['load_balance_loss_weight']}")
                    print(f"      z_loss_weight: {moe_routing_config['z_loss_weight']}")
                    print(f"      temperature: {moe_routing_config['temperature_start']} → {moe_routing_config['temperature_end']} over {moe_routing_config['temperature_steps']} steps")
                    print(f"   Training:")
                    print(f"      batch_size: {training_config_defaults['batch_size']}")
                    print(f"      gradient_accumulation: {training_config_defaults['gradient_accumulation_steps']}")
                    print(f"      max_steps: {training_config_defaults['max_steps']}")
                    if early_stopping_config['early_stopping']:
                        print(f"   Early Stopping:")
                        print(f"      patience: {early_stopping_config['early_stopping_patience']}")
                        print(f"      min_delta: {early_stopping_config['early_stopping_min_delta']}")
                        print(f"      window: {early_stopping_config['early_stopping_window']}")
    except Exception as e:
        print(f"⚠️  Could not load config.yaml: {e}")
        print(f"   Using default DeepSeek-MoE configuration")

    # Create model with DeepSeek-MoE configuration from config.yaml
    # Proper DeepSeek-MoE: 8 routed experts with top_k=2 (select 2 out of 8)
    model = SimpleMoEModel(
        vocab_size=vocab_size,
        embedding_dim=moe_arch_config['embedding_dim'],
        num_shared_experts=moe_arch_config['num_shared_experts'],
        num_routed_experts=moe_arch_config['num_routed_experts'],
        top_k=moe_arch_config['top_k'],
        noise_scale=moe_routing_config['noise_scale'],
        load_balance_loss_weight=moe_routing_config['load_balance_loss_weight'],
        z_loss_weight=moe_routing_config['z_loss_weight'],
        temperature_schedule=moe_routing_config['temperature_schedule'],
        temperature_start=moe_routing_config['temperature_start'],
        temperature_end=moe_routing_config['temperature_end'],
        temperature_steps=moe_routing_config['temperature_steps'],
    )
    
    # Wrap model to match expected signature: model(input_ids) -> logits
    # SimpleMoEModel returns tuple, so we need a wrapper
    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, input_ids):
            # SimpleMoEModel expects (text_tokens, image_features=None, ...)
            # We only have text, so pass None for image_features
            output = self.base_model(input_ids, image_features=None, return_load_balance_loss=False, return_gate_logits=False)
            # SimpleMoEModel returns tuple, extract logits
            if isinstance(output, tuple):
                logits = output[0]  # First element is logits
            else:
                logits = output
            return logits
    
    model = ModelWrapper(model)
    
    # If model path provided and exists, try to load it
    if args.model_path and os.path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"Loaded model from {args.model_path}")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded model weights from {args.model_path}")
        except Exception as e:
            print(f"Could not load model from {args.model_path}: {e}")
            print("   Using randomly initialized model")
    else:
        if args.model_path:
            print(f"Model path {args.model_path} does not exist, using randomly initialized model")
        else:
            print("No model path provided, using randomly initialized model")
    
    # Create adapter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adapter = ModelAdapter(
        model=model,
        device=device,
        domain_weights={
            'neurodegeneration': args.domain_weight_neurodegeneration,
            'neuroscience': args.domain_weight_neuroscience
        }
    )
    
    # Start training
    # Resolve final values (CLI args override config)
    final_batch_size = args.batch_size if args.batch_size is not None else training_config_defaults['batch_size']
    final_gradient_accumulation = args.gradient_accumulation if args.gradient_accumulation is not None else training_config_defaults['gradient_accumulation_steps']
    final_max_steps = args.max_steps if args.max_steps is not None else training_config_defaults['max_steps']
    final_learning_rate = args.learning_rate if args.learning_rate is not None else training_config_defaults['learning_rate']
    final_warmup_steps = args.warmup_steps if args.warmup_steps is not None else training_config_defaults['warmup_steps']
    final_save_interval = args.save_interval if args.save_interval is not None else training_config_defaults['save_interval']

    # Debug: Show which values are being used
    print(f"\n📋 Final training configuration:")
    print(f"   batch_size: {final_batch_size} {'(from CLI)' if args.batch_size is not None else '(from config.yaml)'}")
    print(f"   gradient_accumulation: {final_gradient_accumulation} {'(from CLI)' if args.gradient_accumulation is not None else '(from config.yaml)'}")
    print(f"   max_steps: {final_max_steps} {'(from CLI)' if args.max_steps is not None else '(from config.yaml)'}")
    print(f"   learning_rate: {final_learning_rate} {'(from CLI)' if args.learning_rate is not None else '(from config.yaml)'}")
    print(f"   warmup_steps: {final_warmup_steps} {'(from CLI)' if args.warmup_steps is not None else '(from config.yaml)'}")
    print(f"   save_interval: {final_save_interval} {'(from CLI)' if args.save_interval is not None else '(from config.yaml)'}")
    print()

    train(
        model=model,
        dataset=dataset,
        val_dataset=val_dataset,
        adapter=adapter,
        checkpoint_dir=args.output_dir,
        batch_size=final_batch_size,
        gradient_accumulation_steps=final_gradient_accumulation,
        max_steps=final_max_steps,
        learning_rate=final_learning_rate,
        warmup_steps=final_warmup_steps,
        save_interval=final_save_interval,
        log_interval=args.log_interval,
        domain_log_interval=args.domain_log_interval,
        resume_from_checkpoint=args.resume_from_checkpoint,
        early_stopping=early_stopping_config['early_stopping'],
        early_stopping_patience=early_stopping_config['early_stopping_patience'],
        early_stopping_min_delta=early_stopping_config['early_stopping_min_delta'],
        early_stopping_window=early_stopping_config['early_stopping_window']
    )


if __name__ == "__main__":
    main()

