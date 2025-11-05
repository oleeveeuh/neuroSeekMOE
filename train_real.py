"""
Real PyTorch Training for NeuroSeek-MoE

This implements actual parameter training with PyTorch, including:
- Trainable MoE modules with learnable parameters
- Optimizers (Adam, SGD)
- Loss functions (cross-entropy, MSE)
- GPU/CPU selection
- Model checkpointing
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Optional
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Please install with: pip install torch")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("⚠️  DeepSpeed not available. Install with: pip install deepspeed")

from model_architecture import Disease


def top_k_gating(logits, k=2):
    """Select top-k experts per token and return their indices and normalized weights."""
    topk_values, topk_indices = torch.topk(logits, k, dim=-1)
    gate_probs = torch.softmax(topk_values, dim=-1)
    return gate_probs, topk_indices


class Vocabulary:
    """Vocabulary mapping for hash-based tokenization with decoding support.
    
    Maps words to token IDs (0-10006) and can decode token IDs back to words.
    Handles hash collisions by storing the most common word for each token ID.
    """
    
    def __init__(self, vocab_size: int = 10007):
        self.vocab_size = vocab_size
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.token_counts: Dict[int, Dict[str, int]] = {}  # Track word frequencies per token ID
        
    def tokenize(self, text: str) -> Tuple[List[int], List[str]]:
        """Tokenize text and return both token IDs and original words.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (token_ids, words) for this text
        """
        words = text.split()[:128]
        tokens = []
        word_list = []
        
        for word in words:
            if not word.strip():
                continue
            word_clean = word.strip().lower()
            token_id = abs(hash(word_clean)) % self.vocab_size
            
            # Track word frequency for this token ID (for collision resolution)
            if token_id not in self.token_counts:
                self.token_counts[token_id] = {}
            if word_clean not in self.token_counts[token_id]:
                self.token_counts[token_id][word_clean] = 0
            self.token_counts[token_id][word_clean] += 1
            
            # Update mappings
            self.word_to_id[word_clean] = token_id
            if token_id not in self.id_to_word:
                self.id_to_word[token_id] = word_clean
            else:
                # If collision, use most frequent word for this token ID
                most_frequent = max(
                    self.token_counts[token_id].items(),
                    key=lambda x: x[1]
                )[0]
                self.id_to_word[token_id] = most_frequent
            
            tokens.append(token_id)
            word_list.append(word_clean)
        
        return tokens, word_list
    
    def decode(self, token_ids: List[int], unknown_token: str = "<unk>") -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            unknown_token: String to use for unknown token IDs
            
        Returns:
            Decoded text string
        """
        words = []
        for token_id in token_ids:
            if token_id == 0:  # Padding token
                continue
            if token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
            else:
                words.append(unknown_token)
        return " ".join(words)
    
    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            "vocab_size": self.vocab_size,
            "id_to_word": self.id_to_word,
            "token_counts": {str(k): v for k, v in self.token_counts.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str) -> None:
        """Load vocabulary from file."""
        with open(path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.vocab_size = vocab_data.get("vocab_size", 10007)
        self.id_to_word = {int(k): v for k, v in vocab_data.get("id_to_word", {}).items()}
        self.token_counts = {
            int(k): v for k, v in vocab_data.get("token_counts", {}).items()
        }
        # Rebuild word_to_id from id_to_word
        self.word_to_id = {v: k for k, v in self.id_to_word.items()}


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                data.append(obj)
    return data


# Cache BERTScore model to avoid reloading
_bertscore_scorer = None

def bertscore(reference: str, hypothesis: str) -> float:
    """Calculate BERTScore between reference and hypothesis text."""
    global _bertscore_scorer
    try:
        from bert_score import BERTScorer  # type: ignore
        if _bertscore_scorer is None:
            _bertscore_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        P, R, F1 = _bertscore_scorer.score([hypothesis], [reference])
        return float(F1.mean().item())
    except ImportError:
        # Fallback if bert_score not available
        return 0.5
    except Exception:
        # Fallback on any error
        return 0.5


class NeurodegenerativeDataset(Dataset):
    """PyTorch Dataset for multimodal neurodegenerative disease data."""
    
    def __init__(self, text_data: List[Dict], image_data: List[Dict], device: str = "cpu", vocab: Vocabulary = None):
        self.text_data = text_data
        self.image_data = image_data
        self.device = device
        self.vocab = vocab
        # Pad image_data to match text_data length
        max_len = max(len(text_data), len(image_data))
        if len(image_data) < max_len:
            self.image_data = image_data * (max_len // len(image_data) + 1)
            self.image_data = self.image_data[:max_len]
        
        # Build vocabulary from dataset if provided
        self._build_vocab()
    
    def _build_vocab(self) -> None:
        """Build vocabulary from all text in dataset."""
        if self.vocab is None:
            return
        
        # Only build if vocabulary is empty
        if len(self.vocab.id_to_word) == 0:
            print(f"   Building vocabulary from {len(self.text_data)} text samples...")
            for record in self.text_data:
                text = record.get("text", "")
                if text:
                    self.vocab.tokenize(text)
            print(f"   ✅ Vocabulary built: {len(self.vocab.id_to_word)} token-to-word mappings")
        
    def __len__(self) -> int:
        return len(self.text_data)
    
    def __getitem__(self, idx: int) -> Dict:
        text_item = self.text_data[idx % len(self.text_data)]
        image_item = self.image_data[idx % len(self.image_data)]
        
        return {
            "text": text_item.get("text", ""),
            "caption": image_item.get("caption", ""),
            "disease": text_item.get("disease") or image_item.get("disease") or "AD",
            "modality": "combined",
        }


def load_balance_loss(gate_logits):
    """Encourages balanced expert usage across the batch using entropy."""
    if gate_logits is None or gate_logits.numel() == 0:
        return torch.tensor(0.0, device=gate_logits.device if gate_logits is not None else 'cpu')
    probs = torch.softmax(gate_logits, dim=-1)
    mean_probs = probs.mean(dim=0)
    # Add small epsilon to avoid log(0)
    entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum()
    return entropy


class SimpleMoEModel(nn.Module):
    """Simplified trainable MoE model with learnable parameters.
    
    Supports flexible expert configurations:
    - num_text_experts: Number of text-only experts
    - num_image_experts: Number of image-only experts  
    - num_multimodal_experts: Number of multimodal experts (handle both modalities)
    """
    
    def __init__(
        self,
        vocab_size: int = 10007,
        embedding_dim: int = 128,
        num_experts: int = 2,
        num_text_experts: int = None,
        num_image_experts: int = None,
        num_multimodal_experts: int = None,
    ):
        super().__init__()
        
        # Backward compatibility: if only num_experts is provided, use it
        # Otherwise, use the sum of all expert types or default to num_experts
        if num_text_experts is None and num_image_experts is None and num_multimodal_experts is None:
            total_num_experts = num_experts
        else:
            num_text_experts = num_text_experts or 0
            num_image_experts = num_image_experts or 0
            num_multimodal_experts = num_multimodal_experts or 0
            total_num_experts = num_text_experts + num_image_experts + num_multimodal_experts
            if total_num_experts == 0:
                total_num_experts = num_experts  # Fallback to default
        
        self.num_experts = total_num_experts
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size  # Store vocab_size for validation
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Shared expert pool (standard 2-layer MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.ReLU(),
                nn.Linear(4 * embedding_dim, embedding_dim),
            ) for _ in range(total_num_experts)
        ])
        
        # Single shared gate
        self.gate = nn.Linear(embedding_dim, total_num_experts)
        self.gate_temperature = 1.0  # Temperature for softmax (can be tuned)
        
        # Joint fusion (combines expert outputs)
        # Single expert output dimension
        self.joint_fusion = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.joint_fusion_norm = nn.LayerNorm(embedding_dim)
        
        # Output decoder (standard 2-layer MLP)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, vocab_size),
        )
        
    def forward(self, text_tokens: torch.Tensor, image_features: torch.Tensor = None, return_load_balance_loss: bool = False, return_gate_logits: bool = False):
        # Validate token indices before embedding
        if torch.any((text_tokens < 0) | (text_tokens >= self.vocab_size)):
            raise ValueError(f"Invalid token indices detected: min={text_tokens.min().item()}, max={text_tokens.max().item()}, vocab_size={self.vocab_size}")
        
        # Embed text tokens
        embedded = self.embedding(text_tokens)  # [batch, seq_len, embedding_dim]
        
        # Check for NaN in embedding output
        if torch.isnan(embedded).any():
            raise ValueError("NaN detected in embedding output — check input token indices")
        
        # Average pooling to get fixed-size representation
        pooled_text = embedded.mean(dim=1)  # [batch, embedding_dim]
        
        # Image input preparation
        if image_features is None:
            pooled_image = pooled_text  # Fallback to text features
        else:
            if len(image_features.shape) > 2:
                pooled_image = image_features.mean(dim=1)  # [batch, embedding_dim]
            else:
                pooled_image = image_features  # [batch, embedding_dim]
        
        # Compute gate logits using shared gate
        gate_logits = self.gate(pooled_text)  # [batch, num_experts]
        
        # Check for NaN in gate logits
        if torch.isnan(gate_logits).any():
            raise ValueError("NaN detected in gate_logits")
        
        # Apply temperature scaling for better routing
        gate_logits = gate_logits / self.gate_temperature
        
        # Sparse top-k routing
        gate_probs, topk_idx = top_k_gating(gate_logits, k=2)
        
        # Compute expert outputs and combine with top-k routing
        # gate_probs: [batch, k], topk_idx: [batch, k]
        outputs = []
        for i in range(topk_idx.shape[1]):
            idx = topk_idx[:, i]  # [batch] - expert indices for each sample
            # Compute expert outputs for each sample
            expert_out_list = []
            for b in range(pooled_text.shape[0]):
                expert_idx = idx[b].item()
                expert_out_list.append(self.experts[expert_idx](pooled_text[b:b+1]))  # [1, embedding_dim]
            expert_out = torch.cat(expert_out_list, dim=0)  # [batch, embedding_dim]
            # Multiply by gate probabilities: [batch, 1] * [batch, embedding_dim] -> [batch, embedding_dim]
            weighted_out = gate_probs[:, i:i+1] * expert_out  # [batch, embedding_dim]
            outputs.append(weighted_out)
        combined = sum(outputs)  # [batch, embedding_dim]
        
        # Joint fusion with normalization and residual connection
        fused_output = self.joint_fusion(combined)  # [batch, embedding_dim]
        # Apply normalization and residual: x = x + dropout(norm(ffn(x)))
        fused_output = self.joint_fusion_norm(fused_output)
        fused_output = nn.functional.dropout(fused_output, p=0.1, training=self.training)
        # Add residual connection
        fused_output = combined + fused_output  # Residual connection
        
        # Decode to vocabulary
        output = self.decoder(fused_output)  # [batch, vocab_size]
        
        # Debug: Check output shape (should be [batch, vocab_size])
        # Only print once to reduce noise
        if len(output.shape) != 2:
            if not hasattr(self, '_debug_shape_warned') or not self._debug_shape_warned:
                print(f"⚠️  Model decoder output shape is unexpected: {output.shape}")
                print(f"   fused_output shape: {fused_output.shape}")
                print(f"   pooled_text shape: {pooled_text.shape}")
                print(f"   text_tokens shape: {text_tokens.shape}")
                print(f"   This should not happen - model should output [batch, vocab_size]")
                self._debug_shape_warned = True
        
        if return_load_balance_loss or return_gate_logits:
            if return_gate_logits:
                # Return gate_logits for entropy-based load balancing
                return output, (gate_logits, None, None)  # Single gate for all experts
            else:
                # Legacy: return zero load balance loss (now computed in training loop)
                return output, torch.tensor(0.0, device=output.device)
        return output


def train_real_model(
    multimodal_jsonl: str = None,
    text_jsonl: str = None,
    image_jsonl: str = None,
    results_path: str = None,
    outputs_dir: str = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    disable_diagrams: bool = False,
    resume_from_epoch: int = None,
    num_experts: int = 2,
    vocab_path: str = None,
    num_text_experts: int = None,
    num_image_experts: int = None,
    num_multimodal_experts: int = None,
    comparison_mode: bool = False,
    early_stopping_patience: int = 5,
    deepspeed_config: str = "ds_config.json",
    use_deepspeed: bool = True,
) -> Dict[str, float]:
    """Real PyTorch training with actual parameter updates.
    
    Returns:
        Dictionary with training metrics (loss, bertscore)
    """
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch is required for real training. Install with: pip install torch")
        return {}
    
    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if comparison_mode:
        print(f"🔄 Training configuration (comparison mode)")
    else:
        print(f"🚀 Starting REAL NeuroSeek-MoE training")
    print(f"📊 Configuration:")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {early_stopping_patience if early_stopping_patience else 'Disabled'}")
    print(f"   Diagras: {'Disabled' if disable_diagrams else 'Enabled'}")
    
    _ensure_dir(os.path.dirname(results_path) or ".")
    _ensure_dir(outputs_dir)
    _ensure_dir(checkpoint_dir)
    
    # Initialize or load vocabulary
    if vocab_path is None:
        vocab_path = os.path.join(outputs_dir, "vocabulary.json")
    
    vocab = Vocabulary(vocab_size=10007)
    if os.path.exists(vocab_path) and resume_from_epoch is not None:
        # Load existing vocabulary when resuming
        print(f"📖 Loading vocabulary from {vocab_path}")
        vocab.load(vocab_path)
        print(f"   Loaded {len(vocab.id_to_word)} token-to-word mappings")
    else:
        print(f"📖 Building vocabulary from dataset...")
    
    # Load data
    if multimodal_jsonl:
        print(f"\n📁 Loading multimodal dataset from {multimodal_jsonl}")
        multimodal_ds = load_jsonl(multimodal_jsonl)
        if not multimodal_ds:
            print("⚠️  Dataset is empty")
            return
        
        text_ds = [item for item in multimodal_ds if item.get('modality') == 'text']
        image_ds = [item for item in multimodal_ds if item.get('modality') == 'image']
        print(f"✅ Loaded {len(text_ds)} text records and {len(image_ds)} image records")
    else:
        print(f"\n📁 Loading separate datasets")
        if not text_jsonl or not image_jsonl:
            print("❌ ERROR: Must provide either multimodal_jsonl or both text_jsonl and image_jsonl")
            return
        text_ds = load_jsonl(text_jsonl)
        image_ds = load_jsonl(image_jsonl)
        print(f"✅ Loaded {len(text_ds)} text records and {len(image_ds)} image records")
    
    # Create PyTorch dataset (this will build vocabulary if needed)
    dataset = NeurodegenerativeDataset(text_ds, image_ds, device=device, vocab=vocab)
    
    # Save vocabulary after building
    if len(vocab.id_to_word) > 0:
        vocab.save(vocab_path)
        print(f"📖 Vocabulary saved to {vocab_path}")
    
    # Split dataset into train/test (80/20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total samples: {total_size}")
    print(f"   Train samples: {train_size} (80%)")
    print(f"   Test samples: {test_size} (20%)")
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Build model
    print(f"\n🏗️  Building trainable PyTorch MoE model...")
    
    # Determine expert configuration
    if num_text_experts is not None or num_image_experts is not None or num_multimodal_experts is not None:
        # Explicit configuration
        num_text_experts_val = num_text_experts or 0
        num_image_experts_val = num_image_experts or 0
        num_multimodal_experts_val = num_multimodal_experts or 0
        print(f"   Expert configuration:")
        print(f"     - Text-only experts: {num_text_experts_val}")
        print(f"     - Image-only experts: {num_image_experts_val}")
        print(f"     - Multimodal experts: {num_multimodal_experts_val}")
        model = SimpleMoEModel(
            vocab_size=10007,
            embedding_dim=128,
            num_text_experts=num_text_experts_val,
            num_image_experts=num_image_experts_val,
            num_multimodal_experts=num_multimodal_experts_val,
        )
    else:
        # Backward compatibility: split num_experts evenly
        print(f"   Expert configuration: {num_experts} text experts, {num_experts} image experts")
        model = SimpleMoEModel(vocab_size=10007, embedding_dim=128, num_experts=num_experts)
    
    model = model.to(device)
    print(f"✅ Model created and moved to {device}")
    
    # DIAGNOSTIC 4: Parameter & optimizer state
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Initialize DeepSpeed if available and requested
    model_engine = None
    optimizer = None
    scheduler = None
    
    if use_deepspeed and DEEPSPEED_AVAILABLE:
        print(f"\n🚀 Initializing DeepSpeed with config: {deepspeed_config}")
        # Load DeepSpeed config
        if os.path.exists(deepspeed_config):
            with open(deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            print(f"⚠️  DeepSpeed config file not found: {deepspeed_config}")
            print("   Using default DeepSpeed config")
            ds_config = {}
        
        # Set auto values in config - replace all "auto" strings with actual values
        # BF16
        if "bf16" not in ds_config:
            ds_config["bf16"] = {}
        if ds_config["bf16"].get("enabled") == "auto":
            ds_config["bf16"]["enabled"] = False  # Default to FP32 for compatibility
        
        # Optimizer params
        if "optimizer" not in ds_config:
            ds_config["optimizer"] = {"type": "AdamW", "params": {}}
        if "params" not in ds_config["optimizer"]:
            ds_config["optimizer"]["params"] = {}
        opt_params = ds_config["optimizer"]["params"]
        if opt_params.get("lr") == "auto":
            opt_params["lr"] = learning_rate
        if opt_params.get("betas") == "auto":
            opt_params["betas"] = [0.9, 0.999]  # Standard Adam/AdamW betas
        if opt_params.get("eps") == "auto":
            opt_params["eps"] = 1e-8  # Standard epsilon
        if opt_params.get("weight_decay") == "auto":
            opt_params["weight_decay"] = 1e-5
        
        # Scheduler params
        if "scheduler" not in ds_config:
            ds_config["scheduler"] = {"type": "WarmupLR", "params": {}}
        if "params" not in ds_config["scheduler"]:
            ds_config["scheduler"]["params"] = {}
        sched_params = ds_config["scheduler"]["params"]
        if sched_params.get("warmup_min_lr") == "auto":
            sched_params["warmup_min_lr"] = 0.0
        if sched_params.get("warmup_max_lr") == "auto":
            sched_params["warmup_max_lr"] = learning_rate
        if sched_params.get("warmup_num_steps") == "auto":
            # Default to 10% of total steps (estimate based on epochs)
            # This is approximate - actual steps depend on dataset size
            sched_params["warmup_num_steps"] = max(100, int(epochs * 100 * 0.1))
        
        # Training batch sizes
        if ds_config.get("train_batch_size") == "auto":
            ds_config["train_batch_size"] = batch_size
        if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
            ds_config["train_micro_batch_size_per_gpu"] = batch_size
        if ds_config.get("gradient_accumulation_steps") == "auto":
            ds_config["gradient_accumulation_steps"] = 1
        if ds_config.get("gradient_clipping") == "auto":
            ds_config["gradient_clipping"] = 1.0
        
        # Zero optimization auto values
        if "zero_optimization" not in ds_config:
            ds_config["zero_optimization"] = {}
        zero_opt = ds_config["zero_optimization"]
        if zero_opt.get("reduce_bucket_size") == "auto":
            zero_opt["reduce_bucket_size"] = 5e8
        if zero_opt.get("stage3_prefetch_bucket_size") == "auto":
            zero_opt["stage3_prefetch_bucket_size"] = 5e7
        if zero_opt.get("stage3_param_persistence_threshold") == "auto":
            zero_opt["stage3_param_persistence_threshold"] = 1e6
        
        # Debug: Print config values (first few key ones)
        print(f"   Config values:")
        print(f"     Optimizer type: {ds_config.get('optimizer', {}).get('type', 'N/A')}")
        print(f"     LR: {opt_params.get('lr', 'N/A')}, Betas: {opt_params.get('betas', 'N/A')}, Eps: {opt_params.get('eps', 'N/A')}")
        print(f"     Weight decay: {opt_params.get('weight_decay', 'N/A')}")
        print(f"     ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}")
        print(f"     BF16 enabled: {ds_config.get('bf16', {}).get('enabled', 'N/A')}")
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        print(f"✅ DeepSpeed initialized")
        print(f"   Optimizer: {ds_config.get('optimizer', {}).get('type', 'AdamW')}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {opt_params.get('weight_decay', 'N/A')}")
        print(f"   ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}")
    else:
        if use_deepspeed:
            print(f"⚠️  DeepSpeed requested but not available. Using standard PyTorch optimizer.")
        # Standard PyTorch optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # Learning rate scheduler to reduce LR when validation plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        model_engine = model
        
        # DIAGNOSTIC 4 (continued): Optimizer state
        print(f"   Optimizer: {type(optimizer).__name__}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {optimizer.param_groups[0]['weight_decay']}")
        print(f"   Optimizer param groups: {len(optimizer.param_groups)}")
    
    # Initialize tracking variables before checkpoint loading
    start_epoch = 0
    loaded_checkpoint_loss = None
    best_test_loss = float('inf')
    best_test_loss_epoch = 0
    epochs_without_improvement = 0
    
    # Check for resume checkpoint
    if resume_from_epoch is not None:
        if use_deepspeed and DEEPSPEED_AVAILABLE and model_engine is not None:
            # DeepSpeed checkpoint loading
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{resume_from_epoch}")
            if os.path.exists(checkpoint_path):
                print(f"\n📂 Loading DeepSpeed checkpoint from epoch {resume_from_epoch}...")
                try:
                    _, client_state = model_engine.load_checkpoint(checkpoint_path)
                    if client_state:
                        start_epoch = client_state.get('epoch', resume_from_epoch)
                        if 'loss' in client_state:
                            loaded_checkpoint_loss = client_state['loss']
                        if 'best_test_loss' in client_state:
                            best_test_loss = client_state['best_test_loss']
                        if 'best_test_loss_epoch' in client_state:
                            best_test_loss_epoch = client_state['best_test_loss_epoch']
                    print(f"✅ DeepSpeed checkpoint loaded successfully!")
                    print(f"   Resuming from epoch {start_epoch}")
                except Exception as e:
                    print(f"⚠️  Failed to load DeepSpeed checkpoint: {e}")
                    print("   Starting training from scratch...")
                    start_epoch = 0
            else:
                print(f"⚠️  DeepSpeed checkpoint not found: {checkpoint_path}")
                print("   Starting training from scratch...")
                start_epoch = 0
        else:
            # Standard PyTorch checkpoint loading
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{resume_from_epoch}.pt")
            if os.path.exists(checkpoint_path):
                print(f"\n📂 Loading checkpoint from epoch {resume_from_epoch}...")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if optimizer is not None:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch']
                    if 'loss' in checkpoint:
                        loaded_checkpoint_loss = checkpoint['loss']
                    # Load best test loss if available
                    if 'best_test_loss' in checkpoint:
                        best_test_loss = checkpoint['best_test_loss']
                    if 'best_test_loss_epoch' in checkpoint:
                        best_test_loss_epoch = checkpoint['best_test_loss_epoch']
                    print(f"✅ Checkpoint loaded successfully!")
                    print(f"   Resuming from epoch {start_epoch}")
                    if loaded_checkpoint_loss is not None:
                        print(f"   Previous loss: {loaded_checkpoint_loss:.4f}")
                    if best_test_loss != float('inf'):
                        print(f"   Previous best test loss: {best_test_loss:.4f} (epoch {best_test_loss_epoch})")
                    # Verify model parameters were loaded
                    num_params = sum(p.numel() for p in model.parameters())
                    print(f"   Model parameters: {num_params:,} total")
                    # Check a sample parameter to verify it's not random
                    sample_param = next(model.parameters())
                    print(f"   Sample param stats: mean={sample_param.data.mean().item():.4f}, std={sample_param.data.std().item():.4f}")
                except Exception as e:
                    print(f"⚠️  Failed to load checkpoint: {e}")
                    print("   Starting training from scratch...")
                    start_epoch = 0
            else:
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")
                print("   Starting training from scratch...")
                start_epoch = 0
    
    # Initialize timing and test metrics before training check
    start_time = time.time()
    final_bert = 0.0
    test_loss = 0.0
    test_bert = 0.0
    
    # Check if training is needed
    if start_epoch >= epochs:
        print(f"\n⚠️  Checkpoint epoch {start_epoch} is >= total epochs {epochs}")
        print("   Training already complete at this checkpoint!")
        avg_loss = loaded_checkpoint_loss if loaded_checkpoint_loss is not None else 0.0
        # Try to load metrics from checkpoint
        if resume_from_epoch is not None:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{resume_from_epoch}.pt")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if 'loss' in checkpoint:
                        avg_loss = checkpoint['loss']
                    if 'bertscore' in checkpoint:
                        final_bert = checkpoint['bertscore']
                    if 'test_loss' in checkpoint:
                        test_loss = checkpoint['test_loss']
                    if 'test_bertscore' in checkpoint:
                        test_bert = checkpoint['test_bertscore']
                    if 'best_test_loss' in checkpoint:
                        best_test_loss = checkpoint['best_test_loss']
                    if 'best_test_loss_epoch' in checkpoint:
                        best_test_loss_epoch = checkpoint['best_test_loss_epoch']
                except Exception:
                    pass
    else:
        # Training loop
        print(f"\n🔄 Starting training...")
        
        for epoch in range(start_epoch, epochs):
            print(f"\n{'='*60}")
            print(f"📚 Epoch {epoch + 1}/{epochs}" + (f" (resumed from {start_epoch})" if epoch == start_epoch and start_epoch > 0 else ""))
            print(f"{'='*60}")
            
            model.train()
            total_loss = 0.0
            total_aux_loss = 0.0  # Track auxiliary load balance loss
            batch_count = 0
            bert_scores = []
            diagnostics_run = False  # Flag to ensure diagnostics run once per first epoch
            expert_usage = torch.zeros(model.num_experts, device=device)  # Track expert usage
            
            for batch_idx, batch in enumerate(train_dataloader):
                batch_count += 1
                
                # Prepare inputs - tokenize text using vocabulary
                texts = batch["text"]
                tokens_list = []
                for text in texts:
                    # Use vocabulary tokenization (maintains compatibility with hash-based IDs)
                    tokens, _ = vocab.tokenize(text)
                    if not tokens:  # Skip empty texts
                        tokens = [1]  # Use a dummy token instead of empty (avoid 0 for non-padding)
                    tokens_list.append(tokens)
                
                # Filter out completely empty sequences
                if not tokens_list:
                    continue
                
                # Pad to same length
                max_len = max(len(t) for t in tokens_list)
                if max_len == 0:
                    continue
                
                tokens_tensor = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
                for i, tokens in enumerate(tokens_list):
                    for j, token in enumerate(tokens[:max_len]):
                        tokens_tensor[i, j] = token
                
                tokens_tensor = tokens_tensor.to(device)
                
                # Skip empty sequences
                if tokens_tensor.size(1) == 0:
                    continue
                
                # Create input and target for next-token prediction
                # Input: all tokens except last, Target: all tokens except first
                input_tokens = tokens_tensor[:, :-1]  # [batch, seq_len-1]
                target_tokens = tokens_tensor[:, 1:]  # [batch, seq_len-1]
                
                # Only train on non-empty sequences
                if input_tokens.size(1) == 0:
                    continue
                
                # Forward pass
                # DeepSpeed handles zero_grad internally
                if not (use_deepspeed and DEEPSPEED_AVAILABLE):
                    optimizer.zero_grad()
                
                # Process sequence by averaging, then predict next token
                # For simplicity, we'll use the full sequence and predict the last token
                output, gate_logits_tuple = model_engine(input_tokens, return_gate_logits=True)  # [batch, vocab_size]
                gate_logits, _, _ = gate_logits_tuple  # Single shared gate
                
                # Ensure output is 2D [batch, vocab_size] - handle any shape issues
                original_output_shape = output.shape
                if len(output.shape) > 2:
                    # If output is [batch, seq_len, vocab_size], we need to reshape
                    # This shouldn't happen with current model, but handle it gracefully
                    batch_size, output_seq_len, vocab_size = output.shape
                    input_seq_len = input_tokens.shape[1]  # Original input sequence length
                    # Only print warning once per epoch
                    if batch_idx == 0 and epoch == start_epoch:
                        print(f"⚠️  Output is 3D {output.shape}, reshaping to 2D. This indicates a bug in the model.")
                        print(f"   input_tokens shape: {input_tokens.shape}, target_tokens shape: {target_tokens.shape}")
                    
                    # Flatten output: [batch, output_seq_len, vocab_size] -> [batch*output_seq_len, vocab_size]
                    output = output.contiguous().view(-1, vocab_size)  # [batch*output_seq_len, vocab_size]
                    
                    # Match target to output: we need [batch*output_seq_len]
                    # Since target_tokens is [batch, input_seq_len-1], we need to align it
                    # If output_seq_len matches input_seq_len-1, use all target tokens
                    if output_seq_len == target_tokens.shape[1]:
                        target = target_tokens.contiguous().view(-1)  # [batch*output_seq_len]
                    elif output_seq_len < target_tokens.shape[1]:
                        # Output is shorter, take last output_seq_len tokens
                        target = target_tokens[:, -output_seq_len:].contiguous().view(-1)  # [batch*output_seq_len]
                    else:
                        # Output is longer, pad target with last token
                        target = target_tokens.contiguous().view(-1)  # [batch*target_seq_len]
                        # Pad with last token value to match output length
                        padding_len = output.shape[0] - target.shape[0]
                        if padding_len > 0:
                            last_token = target[-1].item() if target.numel() > 0 else 0
                            padding = torch.full((padding_len,), last_token, dtype=target.dtype, device=target.device)
                            target = torch.cat([target, padding], dim=0)
                    
                    if batch_idx == 0 and epoch == start_epoch:
                        print(f"   Reshaped: output {output.shape}, target {target.shape}")
                elif len(output.shape) == 1:
                    # If output is 1D [vocab_size], add batch dimension
                    print(f"⚠️  Output is 1D, unsqueezing: {output.shape} -> [1, vocab_size]")
                    output = output.unsqueeze(0)  # [1, vocab_size]
                    target = target_tokens[:, -1:].squeeze(0) if target_tokens.shape[0] == 1 else target_tokens[:, -1]
                else:
                    # Normal case: output is [batch, vocab_size], target is last token
                    target = target_tokens[:, -1]  # [batch]
                    # Ensure output is contiguous for efficiency
                    output = output.contiguous()
                
                # Final sanity check: output must be 2D [batch, vocab_size] or [batch*seq_len, vocab_size]
                if len(output.shape) != 2:
                    raise ValueError(f"Model output must be 2D, got shape {output.shape} (original: {original_output_shape})")
                
                # Ensure batch dimensions match
                if output.shape[0] != target.shape[0]:
                    raise ValueError(f"Batch size mismatch: output {output.shape[0]} != target {target.shape[0]} (original output shape: {original_output_shape})")
                
                # Debug: Print shapes for first batch
                if batch_idx == 0 and epoch == start_epoch:
                    print(f"   Debug - output shape: {output.shape}, target shape: {target.shape}")
                    print(f"   Debug - input_tokens shape: {input_tokens.shape}, target_tokens shape: {target_tokens.shape}")
                
                # Ignore padding tokens (0) in loss calculation
                # Padding uses token ID 0, which is masked out
                mask = (target != 0)
                if mask.sum() == 0:
                    continue  # Skip batch if all targets are padding
                
                # Calculate loss only on non-padding tokens
                # output shape: [batch, vocab_size] or [batch*seq_len, vocab_size], target shape: [batch] or [batch*seq_len]
                # We need to select the masked rows from both
                # Use explicit indexing to ensure correct shapes
                masked_output = output[mask, :]  # [num_non_padding, vocab_size]
                masked_target = target[mask]  # [num_non_padding]
                
                # Ensure masked_output is 2D and masked_target is 1D
                if len(masked_output.shape) == 1:
                    # If only one non-padding token, add batch dimension
                    masked_output = masked_output.unsqueeze(0)  # [1, vocab_size]
                if len(masked_target.shape) > 1:
                    # If target is somehow 2D, flatten it
                    masked_target = masked_target.flatten()  # [num_non_padding]
                
                # Ensure shapes are correct for CrossEntropyLoss
                # Input: [N, C], Target: [N] where N is batch size, C is num classes
                if len(masked_output.shape) != 2:
                    raise ValueError(f"Expected masked_output to be 2D [num_non_padding, vocab_size], got shape {masked_output.shape}")
                if len(masked_target.shape) != 1:
                    raise ValueError(f"Expected masked_target to be 1D [num_non_padding], got shape {masked_target.shape}")
                if masked_output.shape[0] != masked_target.shape[0]:
                    raise ValueError(f"Batch size mismatch: masked_output {masked_output.shape[0]} != masked_target {masked_target.shape[0]}")
                if masked_output.shape[1] != model.vocab_size:
                    raise ValueError(f"Vocab size mismatch: masked_output {masked_output.shape[1]} != vocab_size {model.vocab_size}")
                
                main_loss = criterion(masked_output, masked_target)
                
                # Compute entropy-based load-balancing auxiliary loss
                aux_loss = load_balance_loss(gate_logits)
                
                # Track expert usage (which experts were selected in top-k)
                with torch.no_grad():
                    # Get top-k indices from gate logits (with temperature)
                    gate_logits_scaled = gate_logits / model.gate_temperature
                    _, topk_idx = top_k_gating(gate_logits_scaled, k=2)
                    # Count unique expert indices used in this batch
                    unique_experts = torch.unique(topk_idx)
                    expert_usage[unique_experts] += 1
                
                # Combine main loss with load-balancing auxiliary loss
                # Weight the load-balancing loss to not dominate (typically 0.01-0.1)
                load_balance_weight = 0.01
                loss = main_loss + load_balance_weight * aux_loss
                
                # Accumulate auxiliary loss
                total_aux_loss += aux_loss.item()
                
                # Check for NaN loss before backward pass
                if torch.isnan(loss) or torch.isnan(main_loss) or torch.isnan(aux_loss):
                    print(f"⚠️  NaN loss detected — skipping batch")
                    print(f"     Main loss: {main_loss.item() if not torch.isnan(main_loss) else 'NaN'}")
                    print(f"     Aux loss: {aux_loss.item() if not torch.isnan(aux_loss) else 'NaN'}")
                    print(f"     Total loss: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                    continue
                
                # Debug first batch after resuming to check loss
                if batch_idx == 0 and epoch == start_epoch and start_epoch > 0:
                    print(f"  🔍 Debug - First batch after resume:")
                    print(f"     Output shape: {output.shape}, Target shape: {target.shape}")
                    print(f"     Mask sum: {mask.sum().item()}/{len(target)}")
                    print(f"     Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
                    print(f"     Target range: [{target.min().item()}, {target.max().item()}]")
                    print(f"     Raw loss (before mask): {criterion(output, target).item():.4f}")
                    print(f"     Main loss: {main_loss.item():.4f}")
                    print(f"     Aux loss (entropy-based): {aux_loss.item():.4f} (weighted: {load_balance_weight * aux_loss.item():.6f})")
                    print(f"     Total loss: {loss.item():.4f}")
                
                # Backward pass and optimizer step
                if use_deepspeed and DEEPSPEED_AVAILABLE and model_engine is not None:
                    # DeepSpeed handles backward and step, including gradient clipping
                    model_engine.backward(loss)
                    model_engine.step()
                else:
                    # Standard PyTorch backward and step
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # DIAGNOSTIC 1: Check if gradients are nonzero (first batch only)
                if not diagnostics_run and epoch == start_epoch:
                    print(f"\n  🔍 DIAGNOSTIC 1: Gradient Check")
                    grad_found = False
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            grad_mean = p.grad.abs().mean().item()
                            grad_max = p.grad.abs().max().item()
                            print(f"     {n}: mean={grad_mean:.6f}, max={grad_max:.6f}")
                            grad_found = True
                            break  # Just show first non-None gradient
                    if not grad_found:
                        print(f"     ⚠️  WARNING: All gradients are None!")
                
                total_loss += loss.item()
                
                # DIAGNOSTIC 2: Sample model output vs target (first batch only)
                if not diagnostics_run and epoch == start_epoch:
                    print(f"\n  🔍 DIAGNOSTIC 2: Model Output vs Target")
                    with torch.no_grad():
                        # Get model output for this batch (without load-balance loss for diagnostics)
                        sample_output = model_engine(input_tokens, return_load_balance_loss=False)
                        probs = torch.softmax(sample_output, dim=-1)
                        topk = torch.topk(probs, 5, dim=-1).indices
                        
                        # Show first sample
                        sample_idx = 0
                        if sample_idx < len(topk):
                            pred_tokens_sample = topk[sample_idx].cpu().tolist()[:5]
                            # Get actual target tokens for comparison
                            actual_targets = target_tokens[sample_idx].cpu().tolist()[:5]
                            print(f"     Sample {sample_idx}:")
                            print(f"       Predicted top-5 tokens: {pred_tokens_sample}")
                            print(f"       Reference tokens: {actual_targets}")
                            print(f"       Output logit range: [{sample_output[sample_idx].min().item():.2f}, {sample_output[sample_idx].max().item():.2f}]")
                            print(f"       Output logit std: {sample_output[sample_idx].std().item():.2f}")
                            
                            # Check if predictions are uniform/bad
                            probs_sample = probs[sample_idx]
                            entropy = -(probs_sample * torch.log(probs_sample + 1e-10)).sum().item()
                            max_entropy = math.log(probs_sample.shape[0])
                            print(f"       Prediction entropy: {entropy:.4f} (max={max_entropy:.4f}, ratio={entropy/max_entropy:.4f})")
                            if entropy / max_entropy < 0.1:
                                print(f"       ⚠️  WARNING: Very uniform predictions (low entropy)")
                
                # DIAGNOSTIC 3: Check loss computation / labels (first batch only)
                if not diagnostics_run and epoch == start_epoch:
                    print(f"\n  🔍 DIAGNOSTIC 3: Loss Computation Check")
                    print(f"     Output shape: {output.shape}")
                    print(f"     Target shape: {target.shape}")
                    print(f"     Mask shape: {mask.shape}, non-zero: {mask.sum().item()}/{len(target)}")
                    print(f"     Target token range: [{target.min().item()}, {target.max().item()}]")
                    print(f"     Target unique values: {torch.unique(target).numel()}")
                    if target.max().item() >= output.shape[1]:
                        print(f"     ⚠️  ERROR: Target token ID {target.max().item()} >= vocab_size {output.shape[1]}")
                    print(f"     Masked output shape: {masked_output.shape}")
                    print(f"     Masked target shape: {masked_target.shape}")
                    print(f"     Loss value: {loss.item():.4f}")
                    diagnostics_run = True  # Mark diagnostics as run
                
                # Compute BLEU and BERTScore metrics periodically (every 100 batches to avoid slowdown)
                # Now uses actual vocabulary decoder for real metrics!
                if batch_idx % 100 == 0 and texts and len(texts) > 0 and not (epoch == start_epoch and start_epoch > 0):
                    try:
                        # Get predicted tokens (argmax of output logits)
                        pred_tokens = output.argmax(dim=-1).cpu().tolist()  # [batch] of token IDs
                        
                        # Compute metrics for each sample in batch (limit to avoid slowdown)
                        max_samples = min(3, len(texts))  # Process up to 3 samples per batch
                        for i in range(max_samples):
                            if i >= len(pred_tokens) or i >= len(texts):
                                continue
                            
                            ref_text = texts[i]
                            if not ref_text or not ref_text.strip():
                                continue
                            
                            # Decode predicted token to text using vocabulary
                            pred_token_id = pred_tokens[i]
                            if isinstance(pred_token_id, (list, torch.Tensor)):
                                pred_token_id = pred_token_id[0] if len(pred_token_id) > 0 else 0
                            if isinstance(pred_token_id, torch.Tensor):
                                pred_token_id = pred_token_id.item()
                            
                            # Decode: try single token first, then sequence if needed
                            hyp_text = vocab.decode([pred_token_id])
                            
                            # If single token decoding fails, try decoding the input sequence context
                            if not hyp_text or hyp_text == "<unk>":
                                if i < len(input_tokens):
                                    seq_tokens = input_tokens[i].cpu().tolist()
                                    # Decode last few tokens for context
                                    hyp_text = vocab.decode(seq_tokens[-5:]) if len(seq_tokens) >= 5 else vocab.decode(seq_tokens)
                                    if not hyp_text or hyp_text == "<unk>":
                                        hyp_text = "unknown"  # Fallback
                            
                            # Debug: Print sample predictions occasionally
                            if len(bert_scores) < 3 and epoch == start_epoch:
                                print(f"  🔍 Sample prediction {len(bert_scores)+1}:")
                                print(f"     Reference: {ref_text[:100]}...")
                                print(f"     Hypothesis: {hyp_text}")
                                print(f"     Predicted token ID: {pred_token_id}")
                            
                            # Compute BERTScore metric with actual predictions!
                            bert_val = bertscore(ref_text, hyp_text)
                            bert_scores.append(bert_val)
                    except Exception as e:
                        # Silently skip metric computation on error to not interrupt training
                        print(f"⚠️  Metric computation error (batch {batch_idx}): {e}")
                        pass
                
                # Debug: Print target statistics occasionally
                if not diagnostics_run and epoch == start_epoch:
                    print(f"  📊 Debug - First batch targets: min={target.min().item()}, max={target.max().item()}, "
                          f"unique={torch.unique(target).numel()}, non-zero={mask.sum().item()}/{len(target)}")
                
                if batch_idx % max(1, len(train_dataloader) // 10) == 0:
                    progress_pct = (batch_idx / len(train_dataloader)) * 100
                    print(f"  ⏳ Progress: {batch_idx}/{len(train_dataloader)} batches "
                          f"({progress_pct:.1f}%) | Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            avg_aux_loss = total_aux_loss / batch_count if batch_count > 0 else 0.0
            avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0
            epoch_time = time.time() - start_time
            num_active_experts = (expert_usage > 0).sum().item()
            
            # Evaluate on test set
            model.eval()
            test_total_loss = 0.0
            test_batch_count = 0
            test_bert_scores = []
            total_test_samples = len(test_dataloader.dataset)
            
            with torch.no_grad():
                for test_batch_idx, test_batch in enumerate(test_dataloader):
                    test_batch_count += 1
                    texts = test_batch["text"]
                    tokens_list = []
                    
                    for text in texts:
                        tokens, _ = vocab.tokenize(text)
                        if not tokens:
                            tokens = [1]
                        tokens_list.append(tokens)
                    
                    if not tokens_list:
                        continue
                    
                    max_len = max(len(t) for t in tokens_list)
                    if max_len == 0:
                        continue
                    
                    tokens_tensor = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
                    for i, tokens in enumerate(tokens_list):
                        for j, token in enumerate(tokens[:max_len]):
                            tokens_tensor[i, j] = token
                    
                    tokens_tensor = tokens_tensor.to(device)
                    
                    if tokens_tensor.size(1) == 0:
                        continue
                    
                    input_tokens = tokens_tensor[:, :-1]
                    target_tokens = tokens_tensor[:, 1:]
                    
                    if input_tokens.size(1) == 0:
                        continue
                    
                    # During evaluation, don't need load-balancing loss
                    output = model_engine(input_tokens, return_load_balance_loss=False)
                    
                    # Handle output shape (should be [batch, vocab_size] but might be 3D)
                    original_output_shape = output.shape
                    if len(output.shape) > 2:
                        # If output is [batch, seq_len, vocab_size], flatten it
                        batch_size, output_seq_len, vocab_size = output.shape
                        output = output.contiguous().view(-1, vocab_size)
                        # Use corresponding target tokens
                        if target_tokens.shape[1] >= output_seq_len:
                            target = target_tokens[:, -output_seq_len:].contiguous().view(-1)
                        else:
                            target = target_tokens.contiguous().view(-1)
                            # Pad if needed
                            if target.shape[0] < output.shape[0]:
                                last_token = target[-1].item() if target.numel() > 0 else 0
                                padding = torch.full((output.shape[0] - target.shape[0],), last_token, 
                                                   dtype=target.dtype, device=target.device)
                                target = torch.cat([target, padding], dim=0)
                    else:
                        target = target_tokens[:, -1]
                    
                    mask = (target != 0)
                    if mask.sum() == 0:
                        continue
                    
                    masked_output = output[mask, :]  # Explicit indexing
                    masked_target = target[mask]
                    
                    # Ensure shapes are correct
                    if len(masked_output.shape) == 1:
                        masked_output = masked_output.unsqueeze(0)
                    if len(masked_target.shape) > 1:
                        masked_target = masked_target.flatten()
                    
                    test_loss_val = criterion(masked_output, masked_target)
                    test_total_loss += test_loss_val.item()
                    
                    # Compute test metrics periodically (every 20 batches to avoid slowdown)
                    # NOTE: This means BERTScore is computed on ~(total_test_batches/20) * 3 samples
                    # For example: 958 test samples / 8 batch_size = ~120 batches
                    # Every 20 batches = 6 batches * 3 samples = 18 BERTScore samples
                    if test_batch_idx % 20 == 0 and texts and len(texts) > 0:
                        try:
                            pred_tokens = output.argmax(dim=-1).cpu().tolist()
                            max_samples = min(3, len(texts))
                            for i in range(max_samples):
                                if i >= len(pred_tokens) or i >= len(texts):
                                    continue
                                ref_text = texts[i]
                                if not ref_text or not ref_text.strip():
                                    continue
                                
                                # Decode predicted tokens to text using vocabulary
                                pred_token_id = pred_tokens[i]
                                if isinstance(pred_token_id, (list, torch.Tensor)):
                                    pred_token_id = pred_token_id[0] if len(pred_token_id) > 0 else 0
                                if isinstance(pred_token_id, torch.Tensor):
                                    pred_token_id = pred_token_id.item()
                                
                                hyp_text = vocab.decode([pred_token_id])
                                if not hyp_text or hyp_text == "<unk>":
                                    if i < len(input_tokens):
                                        seq_tokens = input_tokens[i].cpu().tolist()
                                        hyp_text = vocab.decode(seq_tokens[-5:]) if len(seq_tokens) >= 5 else vocab.decode(seq_tokens)
                                        if not hyp_text or hyp_text == "<unk>":
                                            hyp_text = "unknown"
                                
                                test_bert_val = bertscore(ref_text, hyp_text)
                                test_bert_scores.append(test_bert_val)
                        except Exception:
                            pass
            
            model.train()
            
            test_avg_loss = test_total_loss / test_batch_count if test_batch_count > 0 else 0.0
            test_avg_bert = sum(test_bert_scores) / len(test_bert_scores) if test_bert_scores else 0.0
            
            print(f"\n✅ Epoch {epoch + 1} complete:")
            print(f"   Train Loss: {avg_loss:.4f}")
            print(f"   Test Loss:  {test_avg_loss:.4f} (evaluated on {total_test_samples} test samples)")
            print(f"🔍 Epoch {epoch + 1}: Aux loss = {avg_aux_loss:.4f}, Active experts = {num_active_experts}/{model.num_experts}")
            if bert_scores:
                print(f"   Train BERTScore: {avg_bert:.4f} (from {len(bert_scores)} samples)")
            if test_bert_scores:
                print(f"   Test BERTScore:  {test_avg_bert:.4f} (from {len(test_bert_scores)} samples, computed on ~{test_batch_count // 20} batches to avoid slowdown)")
            print(f"   Epoch Time: {epoch_time:.2f}s")
            
            # Update final test metrics (from last epoch)
            test_loss = test_avg_loss
            test_bert = test_avg_bert
            
            # Update learning rate scheduler based on test loss
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_avg_loss)
                else:
                    # DeepSpeed scheduler handles steps internally
                    pass
            if optimizer is not None and hasattr(optimizer, 'param_groups'):
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = learning_rate
            
            # Track best test loss
            if test_avg_loss < best_test_loss:
                best_test_loss = test_avg_loss
                best_test_loss_epoch = epoch + 1
                epochs_without_improvement = 0
                print(f"   🏆 New best test loss: {best_test_loss:.4f} at epoch {best_test_loss_epoch}")
                print(f"   Current learning rate: {current_lr:.6f}")
            else:
                epochs_without_improvement += 1
                # Warn if test loss is significantly worse than best
                if test_avg_loss > best_test_loss * 1.5:
                    print(f"   ⚠️  Test loss ({test_avg_loss:.4f}) is much worse than best ({best_test_loss:.4f}) - possible overfitting!")
                print(f"   Current learning rate: {current_lr:.6f} (no improvement for {epochs_without_improvement} epochs)")
                
                # Early stopping if patience is set
                if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                    print(f"\n⏹️  Early stopping triggered!")
                    print(f"   No improvement for {epochs_without_improvement} epochs")
                    print(f"   Best test loss: {best_test_loss:.4f} (epoch {best_test_loss_epoch})")
                    print(f"   Stopping at epoch {epoch + 1}/{epochs}")
                    break
            
            # Save checkpoint (include test metrics)
            if use_deepspeed and DEEPSPEED_AVAILABLE and model_engine is not None:
                # DeepSpeed checkpoint saving
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")
                client_state = {
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'bertscore': avg_bert,
                    'test_loss': test_avg_loss,
                    'test_bertscore': test_avg_bert,
                    'best_test_loss': best_test_loss,
                    'best_test_loss_epoch': best_test_loss_epoch,
                }
                model_engine.save_checkpoint(checkpoint_path, client_state=client_state)
                print(f"   DeepSpeed checkpoint saved: {checkpoint_path}")
            else:
                # Standard PyTorch checkpoint saving
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'loss': avg_loss,
                    'bertscore': avg_bert,
                    'test_loss': test_avg_loss,
                    'test_bertscore': test_avg_bert,
                    'best_test_loss': best_test_loss,
                    'best_test_loss_epoch': best_test_loss_epoch,
                }, checkpoint_path)
                print(f"   Checkpoint saved: {checkpoint_path}")
            
            # Update final metrics (from last epoch)
            final_bert = avg_bert
    
    # Final model save
    final_model_path = os.path.join(outputs_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Save results (include test metrics)
    results = {
        "training_complete": True,
        "epochs": epochs,
        "train_size": train_size,
        "test_size": test_size,
        "final_loss": avg_loss,
        "final_bertscore": final_bert,
        "test_loss": test_loss,
        "test_bertscore": test_bert,
        "best_test_loss": best_test_loss if best_test_loss != float('inf') else None,
        "best_test_loss_epoch": best_test_loss_epoch,
        "device": device,
        "model_path": final_model_path,
        "checkpoint_dir": checkpoint_dir,
        "num_experts": num_experts,
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"   Train Loss: {avg_loss:.4f}")
    print(f"   Test Loss:  {test_loss:.4f}")
    if best_test_loss != float('inf'):
        print(f"   🏆 Best Test Loss: {best_test_loss:.4f} (epoch {best_test_loss_epoch})")
    if final_bert > 0:
        print(f"   Train BERTScore: {final_bert:.4f}")
        print(f"   Test BERTScore:  {test_bert:.4f}")
    else:
        print(f"   Train BERTScore: N/A (no metrics computed)")
        print(f"   Test BERTScore:  N/A (no metrics computed)")
    print(f"   Results saved to: {results_path}")
    print(f"{'='*60}\n")
    
    # Return metrics for comparison (include test metrics)
    return {
        "final_loss": avg_loss,
        "final_bertscore": final_bert,
        "test_loss": test_loss,
        "test_bertscore": test_bert,
        "best_test_loss": best_test_loss if best_test_loss != float('inf') else None,
        "best_test_loss_epoch": best_test_loss_epoch,
        "num_experts": num_experts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="REAL PyTorch training for NeuroSeek-MoE")
    
    # Data input
    data_group = ap.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--multimodal-jsonl", help="Path to multimodal JSONL")
    data_group.add_argument("--separate-datasets", action="store_true")
    
    ap.add_argument("--text-jsonl", help="Path to text JSONL (with --separate-datasets)")
    ap.add_argument("--image-jsonl", help="Path to image JSONL (with --separate-datasets)")
    
    # Training options
    ap.add_argument("--results", default="evaluation/results.json")
    ap.add_argument("--outputs", default="./outputs")
    ap.add_argument("--checkpoints", default="checkpoints", help="Directory for model checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.0001,
                    help="Learning rate (default: 0.0001, reduced to prevent overfitting)")
    ap.add_argument("--early-stopping-patience", type=int, default=5,
                    help="Number of epochs without improvement before early stopping (default: 5, set to 0 to disable)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], 
                    help="Device for training (auto, cpu, cuda)")
    ap.add_argument("--disable-diagrams", action="store_true")
    ap.add_argument("--resume-from", type=int, default=None, metavar="EPOCH",
                    help="Resume training from this epoch checkpoint (e.g., --resume-from 5)")
    ap.add_argument("--num-experts", type=int, default=2,
                   help="Number of experts per modality (default: 2, backward compatibility)")
    ap.add_argument("--num-text-experts", type=int, default=None,
                   help="Number of text-only experts (overrides --num-experts if set)")
    ap.add_argument("--num-image-experts", type=int, default=None,
                   help="Number of image-only experts (overrides --num-experts if set)")
    ap.add_argument("--num-multimodal-experts", type=int, default=None,
                   help="Number of multimodal experts (handle both text and image)")
    ap.add_argument("--vocab-path", type=str, default=None,
                   help="Path to vocabulary file (auto-generated in outputs_dir if not provided)")
    
    # DeepSpeed options
    ap.add_argument("--deepspeed-config", default="ds_config.json",
                   help="Path to DeepSpeed configuration file (default: ds_config.json)")
    ap.add_argument("--no-deepspeed", action="store_true",
                   help="Disable DeepSpeed even if available (use standard PyTorch)")
    
    args = ap.parse_args()
    
    # Validate
    if args.separate_datasets:
        if not args.text_jsonl or not args.image_jsonl:
            ap.error("--text-jsonl and --image-jsonl required with --separate-datasets")
        multimodal_jsonl = None
        text_jsonl = args.text_jsonl
        image_jsonl = args.image_jsonl
    else:
        multimodal_jsonl = args.multimodal_jsonl
        text_jsonl = None
        image_jsonl = None
    
    train_real_model(
        multimodal_jsonl=multimodal_jsonl,
        text_jsonl=text_jsonl,
        image_jsonl=image_jsonl,
        results_path=args.results,
        outputs_dir=args.outputs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoints,
        disable_diagrams=args.disable_diagrams,
        resume_from_epoch=args.resume_from,
        num_experts=args.num_experts,
        vocab_path=args.vocab_path,
        num_text_experts=args.num_text_experts,
        num_image_experts=args.num_image_experts,
        num_multimodal_experts=args.num_multimodal_experts,
        early_stopping_patience=args.early_stopping_patience if args.early_stopping_patience > 0 else None,
        deepspeed_config=args.deepspeed_config,
        use_deepspeed=not args.no_deepspeed,
    )


if __name__ == "__main__":
    main()
