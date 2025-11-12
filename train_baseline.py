"""
Train Baseline Model (Standard Transformer without MoE)

This script trains a baseline transformer model without MoE routing
and evaluates it to generate baseline_results.json for comparison.

Usage:
    python train_baseline.py \
        --dataset-text-dir ./data/arxiv/texts \
        --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
        --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
        --output-dir ./evaluations \
        --checkpoint-dir ./checkpoints/baseline \
        --epochs 10 \
        --batch-size 8 \
        --learning-rate 5e-4
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("Warning: sentencepiece not available")

# Import evaluation utilities
from evaluate import (
    compute_perplexity, compute_domain_classification_accuracy,
    compute_mrr_at_k, compute_section_classification_accuracy,
    extract_embeddings, get_drive_results_path
)
from arxiv_dataset import ArXivStreamingDataset, create_dataloader
from training_adapter import ModelAdapter


class BaselineTransformer(nn.Module):
    """Standard Transformer model without MoE routing.
    
    This is a baseline model that uses a standard feedforward network
    instead of MoE routing, for comparison with the MoE model.
    """
    
    def __init__(
        self,
        vocab_size: int = 10007,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, image_features=None, return_load_balance_loss=False, return_gate_logits=False):
        """Forward pass.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            image_features: Ignored (for compatibility)
            return_load_balance_loss: Ignored (for compatibility)
            return_gate_logits: Ignored (for compatibility)
            
        Returns:
            logits: [batch, seq_len, vocab_size] logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits


def train_baseline_model(
    dataset_text_dir: str,
    dataset_metadata: str,
    tokenizer_path: str,
    output_dir: str,
    checkpoint_dir: str = "./checkpoints/baseline",
    epochs: int = 10,  # Default: 10 epochs to match train_real.py
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    embedding_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    ff_dim: int = 1024,
    device: str = "auto",
    test_split: float = 0.1,
    save_interval: int = 5000,
    max_steps: int = None,
) -> str:
    """Train baseline transformer model and evaluate it.
    
    Args:
        dataset_text_dir: Directory containing text files
        dataset_metadata: JSONL file with paper metadata
        tokenizer_path: Path to SentencePiece tokenizer
        output_dir: Output directory for results
        checkpoint_dir: Directory for model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        embedding_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feedforward dimension
        device: Device to use ('auto', 'cuda', 'cpu')
        test_split: Fraction of data for testing
        save_interval: Steps between checkpoints
        max_steps: Maximum training steps (None = use epochs)
        
    Returns:
        Path to baseline_results.json
    """
    print("=" * 60)
    print("Training Baseline Transformer Model")
    print("=" * 60)
    print()
    
    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    if not SENTENCEPIECE_AVAILABLE:
        raise ImportError("sentencepiece package required")
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    print(f"Loaded tokenizer (vocab_size={vocab_size})")
    
    # Create dataset
    full_dataset = ArXivStreamingDataset(
        text_dir=dataset_text_dir,
        metadata_jsonl=dataset_metadata,
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )
    
    # Split into train/test
    all_files = full_dataset.text_files
    n_test = int(len(all_files) * test_split)
    test_files = all_files[-n_test:] if n_test > 0 else []
    train_files = all_files[:-n_test] if n_test > 0 else all_files
    
    # Create train/test datasets
    class SplitDataset(ArXivStreamingDataset):
        def __init__(self, text_files, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.text_files = text_files
    
    train_dataset = SplitDataset(
        train_files,
        text_dir=dataset_text_dir,
        metadata_jsonl=dataset_metadata,
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )
    
    test_dataset = SplitDataset(
        test_files,
        text_dir=dataset_text_dir,
        metadata_jsonl=dataset_metadata,
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )
    
    print(f"Train dataset: {len(train_files)} papers")
    print(f"Test dataset: {len(test_files)} papers")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating baseline transformer model...")
    model = BaselineTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    model.train()
    global_step = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Reshape for loss calculation
            # logits: [batch, seq_len, vocab_size]
            # target_ids: [batch, seq_len]
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids.view(-1)
            
            # Compute loss
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Save checkpoint
            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"baseline_step_{global_step}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
            
            # Check max_steps
            if max_steps is not None and global_step >= max_steps:
                print(f"\nReached max_steps={max_steps}, stopping training")
                break
        
        if max_steps is not None and global_step >= max_steps:
            break
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} completed: avg_loss={avg_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "baseline_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'step': global_step,
    }, final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Evaluating Baseline Model")
    print("=" * 60)
    
    # Wrap model for evaluation
    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, input_ids):
            output = self.base_model(input_ids, image_features=None, return_load_balance_loss=False, return_gate_logits=False)
            if isinstance(output, tuple):
                return output[0]
            return output
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Create adapter
    adapter = ModelAdapter(wrapped_model, device=device)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # 1. Perplexity (without activation hook for baseline)
    print("   Computing perplexity...")
    perplexity, domain_metrics = compute_perplexity(wrapped_model, adapter, test_dataloader, activation_hook=None)
    print(f"   Perplexity: {perplexity:.2f}")
    
    # Print domain-specific results
    if domain_metrics:
        print("\n   Domain-Specific Perplexity:")
        for domain in sorted(domain_metrics.keys()):
            metrics = domain_metrics[domain]
            print(f"     {domain}: {metrics['perplexity']:.2f} ({metrics['num_papers']} papers)")
    
    # 2. Extract embeddings
    print("   Extracting embeddings...")
    embeddings, metadata = extract_embeddings(
        wrapped_model, adapter, test_dataloader, max_samples=None, activation_hook=None
    )
    print(f"   Extracted embeddings: {embeddings.shape}")
    
    # 3. Domain classification accuracy
    print("   Computing domain classification accuracy...")
    domain_accuracy = compute_domain_classification_accuracy(embeddings, metadata)
    print(f"   Domain accuracy: {domain_accuracy:.4f}")
    
    # 4. Neurodegeneration relevance ranking (MRR@20)
    print("   Computing neurodegeneration relevance ranking (MRR@20)...")
    query_indices = [
        i for i, meta in enumerate(metadata)
        if meta.get('has_neurodegeneration', False)
    ]
    mrr_20 = compute_mrr_at_k(embeddings, metadata, query_indices, k=20)
    print(f"   MRR@20: {mrr_20:.4f}")
    
    # 5. Section classification accuracy
    print("   Computing section classification accuracy...")
    section_accuracy = compute_section_classification_accuracy(
        wrapped_model, adapter, test_dataloader, num_samples=min(100, len(test_files))
    )
    print(f"   Section accuracy: {section_accuracy:.4f}")
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_checkpoint': final_model_path,
        'test_samples': len(metadata),
        'metrics': {
            'perplexity': float(perplexity),
            'domain_classification_accuracy': float(domain_accuracy),
            'neurodegeneration_mrr_at_20': float(mrr_20),
            'section_classification_accuracy': float(section_accuracy),
        },
        'domain_metrics': domain_metrics
    }
    
    # Save baseline results
    results_dir = get_drive_results_path(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📁 Saving results to: {results_dir}")
    
    baseline_results_path = os.path.join(results_dir, "baseline_results.json")
    
    # Add baseline-specific metadata
    baseline_results = {
        'model_type': 'baseline_transformer',
        'timestamp': datetime.now().isoformat(),
        'model_checkpoint': final_model_path,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'embedding_dim': embedding_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'total_steps': global_step,
        },
        'test_samples': results['test_samples'],
        'metrics': results['metrics'],
        'domain_metrics': results['domain_metrics'],
    }
    
    with open(baseline_results_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"✅ Baseline results saved to: {baseline_results_path}")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    return baseline_results_path


def main():
    parser = argparse.ArgumentParser(description="Train baseline transformer model")
    
    parser.add_argument(
        "--dataset-text-dir",
        type=str,
        required=True,
        help="Directory containing text files"
    )
    parser.add_argument(
        "--dataset-metadata",
        type=str,
        required=True,
        help="JSONL file with paper metadata"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to SentencePiece tokenizer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluations",
        help="Output directory for results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/baseline",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10 to match train_real.py MoE model)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=1024,
        help="Feedforward dimension"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for testing"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Steps between checkpoints"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (None = use epochs). If set, overrides epochs. Use 50000 to match train_colab.py"
    )
    
    args = parser.parse_args()
    
    baseline_results_path = train_baseline_model(
        dataset_text_dir=args.dataset_text_dir,
        dataset_metadata=args.dataset_metadata,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        device=args.device,
        test_split=args.test_split,
        save_interval=args.save_interval,
        max_steps=args.max_steps,
    )
    
    print(f"\n✅ Baseline training complete!")
    print(f"Results saved to: {baseline_results_path}")


if __name__ == "__main__":
    main()

