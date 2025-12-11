"""
Train Baseline Model (Standard Transformer without MoE)

This script trains a baseline transformer model without MoE routing
and evaluates it to generate baseline_results.json for comparison.

Supports two model types:
- encoder: Bidirectional transformer (BERT-style) with full attention
- decoder: Decoder-only transformer (GPT-style) with causal attention

Usage:
    # Train encoder-only baseline (bidirectional)
    python train_baseline.py \
        --dataset-text-dir ./data/arxiv/texts \
        --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
        --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
        --output-dir ./evaluations \
        --checkpoint-dir ./checkpoints/baseline \
        --model-type encoder \
        --epochs 10 \
        --batch-size 8 \
        --learning-rate 5e-4
    
    # Train decoder-only baseline (GPT-style, causal)
    python train_baseline.py \
        --dataset-text-dir ./data/arxiv/texts \
        --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
        --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
        --output-dir ./evaluations \
        --checkpoint-dir ./checkpoints/baseline \
        --model-type decoder \
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

try:
    from tokenizer_wrapper import TokenizerWrapper, load_medical_tokenizer, DEFAULT_MEDICAL_TOKENIZER
    TOKENIZER_WRAPPER_AVAILABLE = True
except ImportError:
    TOKENIZER_WRAPPER_AVAILABLE = False
    print("tokenizer_wrapper not available, falling back to SentencePiece only")

# Import evaluation utilities
from evaluate import (
    compute_perplexity, compute_domain_classification_accuracy,
    compute_mrr_at_k, compute_section_classification_accuracy,
    extract_embeddings
)
from arxiv_dataset import ArXivStreamingDataset, create_dataloader
from training_adapter import ModelAdapter


class BaselineTransformer(nn.Module):
    """Standard Transformer encoder model without MoE routing.
    
    This is a baseline model that uses a standard feedforward network
    instead of MoE routing, for comparison with the MoE model.
    Uses bidirectional attention (encoder-style).
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
        
        # Transformer encoder layers (bidirectional attention)
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
        
        # Transformer encoding (bidirectional)
        x = self.transformer(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer model (GPT-style) without MoE routing.
    
    This is a decoder-only baseline model that uses causal (unidirectional) attention,
    similar to GPT models. This mimics LLM behavior for autoregressive language modeling.
    Uses TransformerEncoderLayer with causal masking for efficiency.
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
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer encoder layers (we'll apply causal mask in forward)
        # Using encoder layers but with causal masking gives us decoder-only behavior
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
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask for decoder-only model.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask: [seq_len, seq_len] where True means "masked" (can't attend)
        """
        # Create upper triangular mask (causal)
        # True = masked (can't attend), False = can attend
        # Position i can only attend to positions <= i
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask
    
    def forward(self, input_ids, image_features=None, return_load_balance_loss=False, return_gate_logits=False):
        """Forward pass with causal attention.
        
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
        
        # Generate causal mask for decoder-only attention
        # True = masked (can't attend), False = can attend
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # Transformer encoding with causal mask (decoder-only behavior)
        # The mask ensures each position can only attend to previous positions
        x = self.transformer(x, mask=causal_mask)
        
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
    batch_size: int = 6,  # Changed from 8 to match MoE training
    gradient_accumulation_steps: int = 4,  # Added to match MoE training
    learning_rate: float = 5e-4,
    embedding_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    ff_dim: int = 1024,
    device: str = "auto",
    test_split: float = 0.1,
    save_interval: int = 5000,
    max_steps: int = None,
    model_type: str = "encoder",  # "encoder" or "decoder"
    keep_last_n_checkpoints: int = 2,  # Number of checkpoints to keep (delete older ones)
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
        gradient_accumulation_steps: Gradient accumulation steps (for fair comparison with MoE)
        learning_rate: Learning rate
        embedding_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feedforward dimension
        device: Device to use ('auto', 'cuda', 'cpu')
        test_split: Fraction of data for testing
        save_interval: Steps between checkpoints
        max_steps: Maximum training steps (None = use epochs)
        model_type: Type of baseline model ('encoder' or 'decoder')
            - 'encoder': Bidirectional transformer (BERT-style) with full attention
            - 'decoder': Decoder-only transformer (GPT-style) with causal attention
        keep_last_n_checkpoints: Number of checkpoints to keep (older ones are deleted)
        
    Returns:
        Path to baseline_results.json (includes model_type in filename)
    """
    print("=" * 60)
    model_type_display = "Decoder-only (GPT-style)" if model_type == "decoder" else "Encoder-only"
    print(f"Training Baseline Transformer Model ({model_type_display})")
    print("=" * 60)
    print()
    
    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer (try HuggingFace first, fallback to SentencePiece)
    if TOKENIZER_WRAPPER_AVAILABLE:
        # Check if it's a HuggingFace model name or SentencePiece file
        if os.path.exists(tokenizer_path) and (tokenizer_path.endswith('.model') or os.path.isfile(tokenizer_path)):
            # SentencePiece file
            tokenizer = TokenizerWrapper(tokenizer_path, tokenizer_type='sentencepiece')
            print(f"Loaded SentencePiece tokenizer from: {tokenizer_path}")
        elif '/' in tokenizer_path and not os.path.exists(tokenizer_path):
            # HuggingFace model name (e.g., "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            try:
                tokenizer = TokenizerWrapper(tokenizer_path, tokenizer_type='huggingface')
                print(f"✅ Loaded HuggingFace tokenizer: {tokenizer_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load HuggingFace tokenizer '{tokenizer_path}': {e}")
                print(f"   Falling back to default medical tokenizer: {DEFAULT_MEDICAL_TOKENIZER}")
                tokenizer = load_medical_tokenizer()
        else:
            # Use default medical tokenizer
            print(f"Using default medical tokenizer: {DEFAULT_MEDICAL_TOKENIZER}")
            tokenizer = load_medical_tokenizer()
    elif SENTENCEPIECE_AVAILABLE:
        # Fallback to SentencePiece only
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        print(f"Loaded SentencePiece tokenizer from: {tokenizer_path}")
    else:
        raise ImportError("Neither tokenizer_wrapper nor sentencepiece available. Install transformers or sentencepiece.")
    
    vocab_size = tokenizer.get_piece_size()
    print(f"Loaded tokenizer (vocab_size={vocab_size})")
    
    # Create dataset - handle missing text directory like train_colab.py
    if dataset_text_dir and not os.path.exists(dataset_text_dir):
        print(f"Warning: text_dir not found ({dataset_text_dir}), using processed_dataset.jsonl for training")
        # Try to find processed_dataset.jsonl
        processed_dataset_path = dataset_metadata.replace('arxiv_papers.jsonl', 'processed_dataset.jsonl')
        if not os.path.exists(processed_dataset_path):
            processed_dataset_path = dataset_metadata.replace('metadata', 'processed_dataset')

        if os.path.exists(processed_dataset_path):
            print(f"Using processed_dataset.jsonl: {processed_dataset_path}")
            full_dataset = ArXivStreamingDataset(
                text_dir=None,  # No separate text files
                metadata_jsonl=processed_dataset_path,
                tokenizer=tokenizer,
                max_length=512,
                min_length=64
            )
        else:
            raise FileNotFoundError(f"Neither text_dir nor processed_dataset.jsonl found. Checked: {processed_dataset_path}")
    else:
        full_dataset = ArXivStreamingDataset(
            text_dir=dataset_text_dir,
            metadata_jsonl=dataset_metadata,
            tokenizer=tokenizer,
            max_length=512,
            min_length=64
        )
    
    # Load metadata for stratified split (same as evaluate.py)
    import json
    from collections import defaultdict
    metadata = {}
    if os.path.exists(dataset_metadata):
        with open(dataset_metadata, 'r') as f:
            for line in f:
                if line.strip():
                    paper = json.loads(line)
                    arxiv_id = paper.get('arxiv_id', '')
                    if arxiv_id:
                        metadata[arxiv_id] = paper
    
    # Import classify_paper_domain from evaluate.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from evaluate import classify_paper_domain
    except ImportError:
        # Fallback: define it here if import fails
        def classify_paper_domain(paper: dict) -> str:
            categories = paper.get('categories', paper.get('domains', []))
            if not isinstance(categories, list):
                categories = [categories] if categories else []
            title = paper.get('title', '').lower() if paper.get('title') else ''
            abstract = paper.get('abstract', '').lower() if paper.get('abstract') else ''
            text = title + ' ' + abstract
            
            has_cs = any((isinstance(cat, str) and (cat.startswith('cs.') or 'stat.' in cat.lower())) for cat in categories)
            has_bio = any((isinstance(cat, str) and ('q-bio' in cat or 'bio' in cat.lower())) for cat in categories)
            healthcare_domain_labels = ['medical_imaging', 'neuroscience', 'clinical', 'drug_discovery', 'neurodegeneration']
            has_healthcare_domain = any((isinstance(cat, str) and cat in healthcare_domain_labels) for cat in categories)
            
            ml_keywords = ['neural network', 'deep learning', 'machine learning', 'convolutional', 'transformer']
            healthcare_keywords = ['patient', 'clinical', 'medical', 'diagnosis', 'disease', 'treatment', 'brain', 'imaging']
            ml_keyword_count = sum(1 for kw in ml_keywords if kw in text)
            healthcare_keyword_count = sum(1 for kw in healthcare_keywords if kw in text)
            
            has_ml = has_cs or ml_keyword_count >= 1
            has_healthcare = has_bio or has_healthcare_domain or healthcare_keyword_count >= 1
            
            if has_ml and has_healthcare:
                return 'Both'
            elif has_ml:
                return 'ML'
            elif has_healthcare:
                return 'Healthcare'
            else:
                return 'Other'
    
    # Classify all papers by domain (same as evaluate.py)
    print("Creating train/test split...")
    all_files = full_dataset.text_files
    print(f"Total text files: {len(all_files)}")

    # If using processed_dataset.jsonl with text_dir=None, text_files might be metadata IDs
    if not all_files and actual_metadata is not None:
        print("No text files found (using processed_dataset.jsonl), creating file list from metadata...")
        # Use the metadata entries as our "files" - create tuples (arxiv_id, None) since we don't have file paths
        # Limit to 5000 for Colab memory constraints
        all_files = [(arxiv_id, None) for arxiv_id in list(metadata.keys())[:5000]]
        print(f"Created {len(all_files)} file entries from metadata (limited for Colab)")

    print("Classifying papers by domain...")
    file_domains = []
    for item in all_files:
        # Handle both tuple format (arxiv_id, file_path) and string format (just arxiv_id)
        if isinstance(item, tuple):
            arxiv_id, file_path = item
        else:
            arxiv_id, file_path = item, None
        meta = metadata.get(arxiv_id, {})
        domains = meta.get('domains', [])
        categories = meta.get('categories', [])
        title = meta.get('title', '')
        abstract = meta.get('abstract', '')
        
        all_cats = []
        if categories:
            all_cats.extend(categories if isinstance(categories, list) else [categories])
        if domains:
            all_cats.extend(domains if isinstance(domains, list) else [domains])
        
        paper_dict = {
            'categories': all_cats,
            'domains': domains if isinstance(domains, list) else [domains] if domains else [],
            'title': title,
            'abstract': abstract
        }
        domain = classify_paper_domain(paper_dict)
        file_domains.append((arxiv_id, file_path, domain))
    
    # Group files by domain
    domain_groups = defaultdict(list)
    for arxiv_id, file_path, domain in file_domains:
        domain_groups[domain].append((arxiv_id, file_path))
    
    # Print domain distribution
    print(f"\nFull dataset domain distribution:")
    for domain, files in sorted(domain_groups.items()):
        print(f"  {domain}: {len(files)} papers ({len(files)/len(all_files)*100:.1f}%)")
    
    # Stratified sampling: sample proportionally from each domain (same as evaluate.py)
    import random
    random.seed(42)  # For reproducibility - same seed as evaluate.py
    
    test_files = []
    train_files = []
    
    for domain, files in domain_groups.items():
        n_domain_test = max(1, int(len(files) * test_split))  # At least 1 per domain
        random.shuffle(files)
        domain_test = files[:n_domain_test]
        domain_train = files[n_domain_test:]
        test_files.extend(domain_test)
        train_files.extend(domain_train)
    
    # Shuffle test files
    random.shuffle(test_files)
    
    print(f"\nStratified test split:")
    print(f"  Test set: {len(test_files)} papers")
    print(f"  Train set: {len(train_files)} papers")
    
    # Verify test set domain distribution
    test_domains = defaultdict(int)
    for arxiv_id, _ in test_files:
        meta = metadata.get(arxiv_id, {})
        domains = meta.get('domains', [])
        categories = meta.get('categories', [])
        all_cats = []
        if categories:
            all_cats.extend(categories if isinstance(categories, list) else [categories])
        if domains:
            all_cats.extend(domains if isinstance(domains, list) else [domains])
        paper_dict = {
            'categories': all_cats,
            'domains': domains if isinstance(domains, list) else [domains] if domains else [],
            'title': meta.get('title', ''),
            'abstract': meta.get('abstract', '')
        }
        domain = classify_paper_domain(paper_dict)
        test_domains[domain] += 1
    
    print(f"Test set domain distribution:")
    for domain, count in sorted(test_domains.items()):
        print(f"  {domain}: {count} papers ({count/len(test_files)*100:.1f}%)")
    
    # Create train/test datasets - handle missing text directory
    class SplitDataset(ArXivStreamingDataset):
        def __init__(self, text_files, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.text_files = text_files

    # Determine the correct metadata path (processed_dataset.jsonl if text_dir is missing)
    actual_metadata = dataset_metadata
    actual_text_dir = dataset_text_dir

    if dataset_text_dir and not os.path.exists(dataset_text_dir):
        # Use processed_dataset.jsonl instead
        processed_dataset_path = dataset_metadata.replace('arxiv_papers.jsonl', 'processed_dataset.jsonl')
        if not os.path.exists(processed_dataset_path):
            processed_dataset_path = dataset_metadata.replace('metadata', 'processed_dataset')

        if os.path.exists(processed_dataset_path):
            actual_metadata = processed_dataset_path
            actual_text_dir = None
        else:
            raise FileNotFoundError(f"Neither text_dir nor processed_dataset.jsonl found. Checked: {processed_dataset_path}")

    train_dataset = SplitDataset(
        train_files,
        text_dir=actual_text_dir,
        metadata_jsonl=actual_metadata,
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )

    test_dataset = SplitDataset(
        test_files,
        text_dir=actual_text_dir,
        metadata_jsonl=actual_metadata,
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )
    
    print(f"Train dataset: {len(train_files)} papers")
    print(f"Test dataset: {len(test_files)} papers")

    # Create dataloaders - use num_workers=0 for Colab compatibility
    print("Creating dataloaders...")

    # Test if we can access the first sample before creating dataloaders
    print("Testing dataset access...")
    try:
        sample_iter = iter(train_dataset)
        first_sample = next(sample_iter)
        print(f"✅ Successfully accessed first sample: {type(first_sample)}")
        if isinstance(first_sample, dict):
            print(f"   Keys: {list(first_sample.keys())}")
            if 'input_ids' in first_sample:
                print(f"   Input IDs shape: {first_sample['input_ids'].shape}")
    except Exception as e:
        print(f"❌ Error accessing dataset: {e}")
        return None, None, None, None, None, None, None, None, None

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for Colab to avoid multiprocessing issues
        pin_memory=True
    )

    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for Colab to avoid multiprocessing issues
        pin_memory=True
    )
    print("Dataloaders created successfully")

    # Create model
    print(f"\nCreating baseline transformer model...")
    print(f"  Model parameters: vocab_size={vocab_size}, embedding_dim={embedding_dim}, num_layers={num_layers}, num_heads={num_heads}, ff_dim={ff_dim}")
    print(f"  Device: {device}")

    try:
        if model_type == "decoder":
            print(f"  Model type: Decoder-only (GPT-style, causal attention)")
            print("  Initializing DecoderOnlyTransformer...")
            model = DecoderOnlyTransformer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
            )
            print("  DecoderOnlyTransformer created successfully")
        else:
            print(f"  Model type: Encoder-only (bidirectional attention)")
            print("  Initializing BaselineTransformer...")
            model = BaselineTransformer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
            )
            print("  BaselineTransformer created successfully")

        print("  Moving model to device...")
        model = model.to(device)
        print("  Model moved to device successfully")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created successfully: {total_params:,} total, {trainable_params:,} trainable")

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None
    
    # Setup optimizer and loss (matching MoE training settings for fair comparison)
    print("Setting up optimizer and loss...")
    try:
        # Match MoE training settings for fair comparison:
        # - weight_decay=0.01 (same as MoE, stronger regularization)
        # - warmup_start=0.1 (same as MoE, 10% of LR instead of 1%)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        print("  Optimizer and loss created successfully")

        # Setup learning rate scheduler (warmup + cosine decay) to match MoE training
        # This ensures fair comparison with MoE model
        warmup_steps = 2000 if max_steps is not None else 0
        print(f"  Setting up scheduler with warmup_steps={warmup_steps}, max_steps={max_steps}")
        if max_steps is not None and max_steps > warmup_steps:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,  # Start at 10% of learning rate (matches MoE training)
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
            print(f"  Using learning rate schedule: warmup ({warmup_steps} steps, start=10% LR) + cosine decay (matches MoE training)")
        else:
            scheduler = None
            print("  Using constant learning rate (no scheduler)")
    except Exception as e:
        print(f"❌ Error setting up optimizer/scheduler: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None

    # Create checkpoint directory (include model type in path)
    # Handle case where checkpoint_dir might be a file path instead of directory
    if os.path.isfile(checkpoint_dir):
        # If it's an existing file, use its parent directory
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    elif os.path.exists(checkpoint_dir) and not os.path.isdir(checkpoint_dir):
        # If it exists but is not a directory (e.g., a file), use parent directory
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    elif not os.path.exists(checkpoint_dir) and os.path.splitext(checkpoint_dir)[1]:
        # If it doesn't exist but looks like a file path (has extension), extract directory
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    
    # Add model type to checkpoint directory to avoid overwriting encoder/decoder models
    checkpoint_dir = os.path.join(checkpoint_dir, model_type)
    
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Helper function to clean up old checkpoints
    def cleanup_old_checkpoints(checkpoint_dir: str, model_type: str, keep_last_n: int):
        """Delete old checkpoints, keeping only the most recent N."""
        if keep_last_n <= 0:
            return
        
        # Find all checkpoints matching the pattern
        checkpoint_pattern = f"baseline_{model_type}_step_"
        checkpoints = []
        
        if not os.path.exists(checkpoint_dir):
            return
        
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith(checkpoint_pattern) and filename.endswith('.pt'):
                try:
                    # Extract step number from filename: baseline_{model_type}_step_{N}.pt
                    step_str = filename.replace(checkpoint_pattern, '').replace('.pt', '')
                    step_num = int(step_str)
                    checkpoints.append((step_num, os.path.join(checkpoint_dir, filename)))
                except ValueError:
                    continue
        
        # Sort by step number and keep only the last N
        if len(checkpoints) > keep_last_n:
            checkpoints.sort(key=lambda x: x[0])
            # Delete oldest checkpoints (all except the last N)
            for step_num, path in checkpoints[:-keep_last_n]:
                try:
                    os.remove(path)
                    print(f"  Deleted old checkpoint: baseline_{model_type}_step_{step_num}.pt")
                except OSError as e:
                    print(f"  Warning: Could not delete checkpoint {path}: {e}")
    
    # Training loop
    print(f"\nStarting training...")
    if max_steps is not None:
        print(f"Max steps: {max_steps}")
        print(f"Epochs: unlimited (will stop at {max_steps} steps)")
    else:
        print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    
    model.train()
    global_step = 0
    start_time = time.time()
    
    # Determine training mode: step-based or epoch-based
    if max_steps is not None:
        # Step-based training: loop until max_steps reached
        epoch = 0
        while global_step < max_steps:
            epoch += 1
            epoch_loss = 0.0
            num_batches = 0
            
            # Calculate remaining steps for progress bar
            remaining_steps = max_steps - global_step
            progress_bar = tqdm(train_dataloader, desc=f"Step {global_step}/{max_steps} (Epoch {epoch})")
            
            for batch in progress_bar:
                if batch is None:
                    continue
                
                # Check if we've reached max_steps before processing this batch
                if global_step >= max_steps:
                    break
                    
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
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if global_step % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                loss.backward()
                
                # Update weights only after accumulating gradients
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                
                epoch_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
                num_batches += 1
                global_step += 1
                
                # Check again after incrementing (in case we just hit max_steps)
                if global_step >= max_steps:
                    break
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 'step': global_step})
                
                # Save checkpoint
                if global_step % save_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"baseline_{model_type}_step_{global_step}.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': global_step,
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
                    
                    # Clean up old checkpoints
                    cleanup_old_checkpoints(checkpoint_dir, model_type, keep_last_n_checkpoints)
            
            if global_step >= max_steps:
                print(f"\nReached max_steps={max_steps}, stopping training")
                break
            
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch} completed: avg_loss={avg_loss:.4f}, total_steps={global_step}")
    else:
        # Epoch-based training: loop for specified epochs
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
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if num_batches % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                loss.backward()
                
                # Update weights only after accumulating gradients
                if (num_batches + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                epoch_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
                
                # Save checkpoint
                if global_step % save_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"baseline_{model_type}_step_{global_step}.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': global_step,
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
                    
                    # Clean up old checkpoints
                    cleanup_old_checkpoints(checkpoint_dir, model_type, keep_last_n_checkpoints)
            
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1} completed: avg_loss={avg_loss:.4f}")
    
    # Save final model (include model type in filename)
    final_model_path = os.path.join(checkpoint_dir, f"baseline_{model_type}_final.pt")
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
    
    # Debug: Check if perplexity seems suspiciously low
    if perplexity < 2.0:
        print(f"   ⚠️  WARNING: Perplexity {perplexity:.2f} is suspiciously low!")
        print(f"      This may indicate:")
        print(f"      1. Data leakage (test set contains training data)")
        print(f"      2. Very small/easy test set")
        print(f"      3. Loss calculation bug")
        print(f"      Expected perplexity for a baseline model: 50-500+")
        print(f"      Inference perplexities (50,000+) are more realistic.")
    
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
    
    # Save baseline results (include model type in filename)
    # Use the output_dir parameter directly (respect user's choice)
    # Convert to absolute path to ensure consistency
    results_dir = os.path.abspath(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📁 Saving results to: {results_dir}")
    
    baseline_results_path = os.path.join(results_dir, f"baseline_{model_type}_results.json")
    
    # Add baseline-specific metadata
    baseline_results = {
        'model_type': f'baseline_transformer_{model_type}',  # 'baseline_transformer_encoder' or 'baseline_transformer_decoder'
        'architecture': model_type,  # 'encoder' or 'decoder'
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
            'model_type': model_type,
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
        default=6,
        help="Batch size for training (default: 6 to match MoE training)"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4 to match MoE training, effective batch size = 24)"
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
        default=50000,
        help="Maximum training steps (default: 50000 to match MoE training). If None, uses epochs instead"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="encoder",
        choices=["encoder", "decoder"],
        help="Type of baseline model: 'encoder' (bidirectional, BERT-style) or 'decoder' (causal, GPT-style). Default: encoder"
    )
    parser.add_argument(
        "--keep-last-n-checkpoints",
        type=int,
        default=2,
        help="Number of checkpoints to keep (older ones are deleted during training). Default: 2"
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
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        device=args.device,
        test_split=args.test_split,
        save_interval=args.save_interval,
        max_steps=args.max_steps,
        model_type=args.model_type,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
    )
    
    print(f"\n✅ Baseline training complete!")
    print(f"Results saved to: {baseline_results_path}")


if __name__ == "__main__":
    main()

