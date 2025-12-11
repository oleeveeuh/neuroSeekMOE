#!/usr/bin/env python3
"""
Simple Baseline Training Script - matches train_colab.py logic

This is a simplified version of train_baseline.py that uses the exact same
approach as train_colab.py for dataset creation and dataloader setup.
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

try:
    from tokenizer_wrapper import TokenizerWrapper, load_medical_tokenizer, DEFAULT_MEDICAL_TOKENIZER
    TOKENIZER_WRAPPER_AVAILABLE = True
except ImportError:
    TOKENIZER_WRAPPER_AVAILABLE = False
    print("tokenizer_wrapper not available, falling back to SentencePiece only")

from arxiv_dataset import ArXivStreamingDataset, create_dataloader


class BaselineTransformer(nn.Module):
    """Standard Transformer encoder model without MoE routing."""

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

        # Transformer encoder layers (bidirectional attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass with bidirectional attention (encoder-style)."""
        batch_size, seq_len = input_ids.size()

        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_emb = self.embedding(input_ids)
        pos_emb = self.pos_embedding(position_ids)
        embeddings = token_emb + pos_emb

        # Transformer encoder (bidirectional attention)
        if attention_mask is not None:
            # Convert to transformer format (0 for keep, 1 for mask)
            attention_mask = ~attention_mask.bool()

        hidden_states = self.transformer(embeddings, src_key_padding_mask=attention_mask)

        # Output projection
        logits = self.output_proj(hidden_states)

        return logits


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer model (GPT-style) with causal attention."""

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

        # Transformer decoder layers (causal attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass with causal attention (GPT-style)."""
        batch_size, seq_len = input_ids.size()

        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_emb = self.embedding(input_ids)
        pos_emb = self.pos_embedding(position_ids)
        embeddings = token_emb + pos_emb

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)

        # Transformer decoder (causal attention)
        hidden_states = self.transformer(
            tgt=embeddings,
            memory=embeddings,  # Self-attention for decoder-only
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask if attention_mask is not None else None
        )

        # Output projection
        logits = self.output_proj(hidden_states)

        return logits


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer with same logic as train_colab.py"""

    print("Loading tokenizer...")

    # Try to load as HuggingFace model name first (like train_colab.py)
    if not os.path.exists(tokenizer_path) and '/' in tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"✅ Loaded HuggingFace tokenizer: {tokenizer_path}")
            return tokenizer
        except Exception as e:
            print(f"⚠️  Warning: Could not load HuggingFace tokenizer '{tokenizer_path}': {e}")

    # Try tokenizer_wrapper
    if TOKENIZER_WRAPPER_AVAILABLE:
        try:
            tokenizer = load_medical_tokenizer(tokenizer_path)
            print(f"✅ Loaded medical tokenizer: {tokenizer_path}")
            return tokenizer
        except Exception as e:
            print(f"⚠️  Could not load medical tokenizer: {e}")

    # Try SentencePiece
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(tokenizer_path)
            print(f"✅ Loaded SentencePiece tokenizer: {tokenizer_path}")
            return tokenizer
        except Exception as e:
            print(f"⚠️  Could not load SentencePiece tokenizer: {e}")

    raise ImportError("Could not load any tokenizer. Please install transformers or sentencepiece")


def train_baseline_model(
    dataset_text_dir: str,
    dataset_metadata: str,
    tokenizer_path: str,
    output_dir: str,
    checkpoint_dir: str,
    model_type: str = "encoder",
    epochs: int = 10,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    max_steps: int = None,
    learning_rate: float = 5e-4,
    embedding_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    ff_dim: int = 1024,
    device: str = "auto",
    test_split: float = 0.1,
    save_interval: int = 5000,
    keep_last_n_checkpoints: int = 2
):
    """Train baseline transformer model using the same approach as train_colab.py"""

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output and checkpoint directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Get vocab size (same logic as train_colab.py)
    if hasattr(tokenizer, 'get_piece_size'):  # SentencePiece
        vocab_size = tokenizer.get_piece_size()
    elif hasattr(tokenizer, 'vocab_size'):  # HuggingFace
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'get_vocab'):  # TokenizerWrapper
        vocab_size = len(tokenizer.get_vocab())
    else:
        vocab_size = 30522  # Default for BERT-style tokenizers

    print(f"Loaded tokenizer (vocab_size={vocab_size})")

    # Create dataset - EXACT SAME LOGIC AS TRAIN_COLAB.PY
    print("Creating dataset...")
    if os.path.exists(dataset_text_dir):
        dataset = ArXivStreamingDataset(
            text_dir=dataset_text_dir,
            metadata_jsonl=dataset_metadata,
            tokenizer=tokenizer,
            max_length=512,
            min_length=64
        )
    else:
        print("Warning: text_dir not found, using processed_dataset.jsonl for training")
        # Point to processed_dataset.jsonl as both source (same as train_colab.py)
        processed_dataset_path = dataset_metadata.replace('arxiv_papers.jsonl', 'processed_dataset.jsonl')
        if not os.path.exists(processed_dataset_path):
            # Try alternative path (same as train_colab.py)
            processed_dataset_path = dataset_metadata.replace('metadata', 'processed_dataset')

        if os.path.exists(processed_dataset_path):
            dataset = ArXivStreamingDataset(
                text_dir=None,  # No separate text files (same as train_colab.py)
                metadata_jsonl=processed_dataset_path,
                tokenizer=tokenizer,
                max_length=512,
                min_length=64
            )
        else:
            raise FileNotFoundError(f"Neither text_dir nor processed_dataset.jsonl found. Checked: {processed_dataset_path}")

    print(f"Created dataset with ~{len(dataset)} samples")

    # Create dataloader - SAME LOGIC AS TRAIN_COLAB.PY
    print("Creating dataloader...")
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Single-threaded for Colab stability (same as train_colab.py)
        pin_memory=False  # Disable pin_memory for CPU (same as train_colab.py)
    )

    # Create model
    print(f"\nCreating baseline transformer model...")
    print(f"  Model parameters: vocab_size={vocab_size}, embedding_dim={embedding_dim}, num_layers={num_layers}, num_heads={num_heads}, ff_dim={ff_dim}")

    if model_type == "decoder":
        print(f"  Model type: Decoder-only (GPT-style, causal attention)")
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
    else:
        print(f"  Model type: Encoder-only (bidirectional attention)")
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
    print(f"✅ Model created successfully: {total_params:,} total, {trainable_params:,} trainable")

    # Setup optimizer and loss (matching MoE training settings)
    print("Setting up optimizer and loss...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    print("  Optimizer and loss created successfully")

    # Simple training loop (same as train_colab.py - no complex evaluation)
    print("\nStarting training...")
    model.train()
    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)

            # Compute loss (language modeling - predict next token)
            # Shift input_ids for causal models or use input_ids directly for encoder
            if model_type == "decoder":
                # For decoder, predict next token
                targets = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
            else:
                # For encoder, use same input (reconstruction or MLM style)
                targets = input_ids

            # Flatten for loss computation
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Backward pass
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Print progress
            if global_step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                print(f"  Step {global_step}: loss={loss.item():.4f}, lr={current_lr:.6f}, time={elapsed:.1f}s")

            # Save checkpoint
            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'loss': loss.item(),
                    'model_type': model_type,
                    'vocab_size': vocab_size,
                    'embedding_dim': embedding_dim,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'ff_dim': ff_dim,
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Print epoch summary
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} completed - avg loss: {avg_loss:.4f}")

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f"baseline_{model_type}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'model_type': model_type,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'ff_dim': ff_dim,
    }, final_model_path)

    print(f"\n✅ Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Total steps: {global_step}")
    print(f"Total time: {time.time() - start_time:.1f}s")

    return final_model_path


def main():
    parser = argparse.ArgumentParser(description="Train simple baseline transformer model (matches train_colab.py)")

    parser.add_argument('--dataset-text-dir', type=str, required=True, help="Directory containing text files")
    parser.add_argument('--dataset-metadata', type=str, required=True, help="JSONL file with paper metadata")
    parser.add_argument('--tokenizer-path', type=str, required=True, help="Path to tokenizer")
    parser.add_argument('--output-dir', type=str, default='./evaluations', help="Output directory for results")
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/baseline', help="Directory for model checkpoints")

    parser.add_argument('--model-type', type=str, default='encoder', choices=['encoder', 'decoder'], help="Model type")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--gradient-accumulation', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--max-steps', type=int, default=None, help="Maximum training steps")
    parser.add_argument('--learning-rate', type=float, default=5e-4, help="Learning rate")

    parser.add_argument('--embedding-dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--num-layers', type=int, default=6, help="Number of transformer layers")
    parser.add_argument('--num-heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--ff-dim', type=int, default=1024, help="Feed-forward dimension")

    parser.add_argument('--device', type=str, default='auto', help="Device (auto, cpu, cuda)")
    parser.add_argument('--save-interval', type=int, default=5000, help="Save checkpoint every N steps")

    args = parser.parse_args()

    print("=" * 60)
    print("Simple Baseline Transformer Training (matches train_colab.py)")
    print("=" * 60)

    final_model_path = train_baseline_model(
        dataset_text_dir=args.dataset_text_dir,
        dataset_metadata=args.dataset_metadata,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        device=args.device,
        save_interval=args.save_interval,
    )

    print(f"\n✅ Training complete!")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()