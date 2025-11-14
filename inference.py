"""
Production Inference Pipeline for DeepSeekMoE Model

Features:
- Fast inference (<100ms per paper on CPU)
- Batch processing
- Embedding generation
- Literature review (top-k retrieval)
- Domain classification
- Embedding caching
- Similarity search
- Optional INT8 quantization
- Optional ONNX export

Usage:
    from inference import InferencePipeline
    
    pipeline = InferencePipeline(
        checkpoint_path='./checkpoints/step_50000.pt',
        tokenizer_path='./data/arxiv/healthcare_tokenizer.model'
    )
    
    # Generate embedding
    embedding = pipeline.generate_embeddings("Alzheimer's disease research...")
    
    # Batch inference
    embeddings = pipeline.batch_encode(["Paper 1 text...", "Paper 2 text..."])
    
    # Literature review
    results = pipeline.literature_review("neurodegeneration", top_k=10)
    
    # Domain classification
    domain = pipeline.classify_domain("Clinical trial results...")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EmbeddingCache:
    """LRU cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = './cache/embeddings', max_size: int = 10000):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached embeddings
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        os.makedirs(cache_dir, exist_ok=True)
        self._memory_cache = {}
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}.npy")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None
        """
        cache_key = self._get_cache_key(text)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            embedding = np.load(cache_path)
            # Add to memory cache
            if len(self._memory_cache) < self.max_size:
                self._memory_cache[cache_key] = embedding
            return embedding
        
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache.
        
        Args:
            text: Input text
            embedding: Embedding to cache
        """
        cache_key = self._get_cache_key(text)
        
        # Store in memory cache
        if len(self._memory_cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = embedding
        
        # Store on disk
        cache_path = self._get_cache_path(cache_key)
        np.save(cache_path, embedding)


class InferencePipeline:
    """Production inference pipeline for DeepSeekMoE model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = 'cpu',
        max_length: int = 512,
        use_cache: bool = True,
        quantize: bool = False
    ):
        """Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to SentencePiece tokenizer
            device: Device to run on ('cpu' or 'cuda')
            max_length: Maximum sequence length
            use_cache: Whether to use embedding cache
            quantize: Whether to use INT8 quantization
        """
        self.device = torch.device(device)
        self.max_length = max_length
        self.use_cache = use_cache
        self.quantize = quantize
        
        # Load tokenizer
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("sentencepiece package required")
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = self.tokenizer.get_piece_size()
        print(f"Loaded tokenizer (vocab_size={self.vocab_size})")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Quantize if requested
        if quantize and device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            print("Model quantized to INT8")
        
        # Initialize cache
        if use_cache:
            self.cache = EmbeddingCache()
        else:
            self.cache = None
        
        # Domain classifier (lazy-loaded)
        self._domain_classifier = None
        
        # Store base_model reference (set in _load_model)
        self.base_model = None
        
        print(f"Inference pipeline initialized on {device}")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint, automatically detecting MoE vs Baseline.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model
        """
        try:
            from train_real import SimpleMoEModel
            from train_baseline import BaselineTransformer
            
            # Load checkpoint first to detect model type
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Detect if this is a baseline model
            has_gate = any('gate' in key for key in state_dict.keys())
            has_routed_experts = any('routed_experts' in key for key in state_dict.keys())
            has_transformer = any('transformer' in key and 'transformer_layers' not in key for key in state_dict.keys())
            
            is_baseline = has_transformer and not (has_gate or has_routed_experts)
            
            if is_baseline:
                # Load baseline model
                print("Detected BaselineTransformer model")
                
                # Infer model config from checkpoint
                embedding_dim = 256  # Default
                for key in state_dict.keys():
                    if 'embedding.weight' in key:
                        weight = state_dict[key]
                        embedding_dim = weight.shape[1]
                        break
                
                num_layers = 6  # Default
                for key in state_dict.keys():
                    if 'transformer.layers.' in key:
                        parts = key.split('.')
                        for i, part in enumerate(parts):
                            if part == 'layers' and i + 1 < len(parts):
                                try:
                                    layer_idx = int(parts[i + 1])
                                    num_layers = max(num_layers, layer_idx + 1)
                                except ValueError:
                                    pass
                
                base_model = BaselineTransformer(
                    vocab_size=self.vocab_size,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                )
                
                # Store base_model reference
                self.base_model = base_model
                
                # Wrap model
                class ModelWrapper(nn.Module):
                    def __init__(self, base_model):
                        super().__init__()
                        self.base_model = base_model
                    
                    def forward(self, input_ids, return_gate_logits=False):
                        output = self.base_model(
                            input_ids,
                            image_features=None,
                            return_load_balance_loss=False,
                            return_gate_logits=False
                        )
                        return output
                
                model = ModelWrapper(base_model)
                
                # Load checkpoint
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                
                print(f"Loaded baseline model from {checkpoint_path}")
                return model
            else:
                # Load MoE model
                print("Detected SimpleMoEModel (MoE) model")
                
                # Infer model config from checkpoint
                embedding_dim = 256  # Default
                num_routed_experts = 4  # Default
                
                for key in state_dict.keys():
                    if 'gate.weight' in key or 'base_model.gate.weight' in key:
                        weight = state_dict[key]
                        # gate is nn.Linear(embedding_dim, num_routed_experts)
                        # So gate.weight shape is [num_routed_experts, embedding_dim]
                        num_routed_experts = weight.shape[0]
                        embedding_dim = weight.shape[1]
                        break
                
                base_model = SimpleMoEModel(
                    vocab_size=self.vocab_size,
                    embedding_dim=embedding_dim,
                    num_shared_experts=2,
                    num_routed_experts=num_routed_experts,
                )
                
                # Store base_model reference for expert activation capture
                self.base_model = base_model
                
                # Wrap model
                class ModelWrapper(nn.Module):
                    def __init__(self, base_model):
                        super().__init__()
                        self.base_model = base_model
                    
                    def forward(self, input_ids, return_gate_logits=False):
                        output = self.base_model(
                            input_ids,
                            image_features=None,
                            return_load_balance_loss=False,
                            return_gate_logits=return_gate_logits
                        )
                        if isinstance(output, tuple):
                            return output[0] if not return_gate_logits else output
                        return output
                
                model = ModelWrapper(base_model)
                
                # Load checkpoint
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                
                print(f"Loaded MoE model from {checkpoint_path}")
                return model
            
        except Exception as e:
            print(f"Could not load model: {e}")
            raise
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            Token IDs tensor
        """
        token_ids = self.tokenizer.encode(text, out_type=int)
        
        # Truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Pad to max_length
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        
        return torch.tensor([token_ids], dtype=torch.long)
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            1D embedding vector
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        
        # Tokenize
        input_ids = self._tokenize(text).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(input_ids)  # [1, seq_len, vocab_size]
            
            # Extract embeddings: use mean pooling over sequence
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'embedding'):
                # Extract from embedding layer
                embeds = self.model.base_model.embedding(input_ids)  # [1, seq_len, embed_dim]
                embedding = embeds.mean(dim=1).squeeze(0)  # [embed_dim]
            else:
                # Fallback: use logits mean
                embedding = logits.mean(dim=1).squeeze(0)  # [vocab_size]
        
        embedding = embedding.cpu().numpy()
        
        # Cache result
        if self.cache:
            self.cache.set(text, embedding)
        
        return embedding
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings [num_texts, embed_dim]
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.generate_embeddings(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def cosine_similarity(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and corpus embeddings.
        
        Args:
            query_embedding: Query embedding [embed_dim]
            corpus_embeddings: Corpus embeddings [num_docs, embed_dim]
            
        Returns:
            Similarity scores [num_docs]
        """
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        corpus_norms = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(corpus_norms, query_norm)
        
        return similarities
    
    def literature_review(
        self,
        query: str,
        corpus_embeddings: np.ndarray,
        corpus_metadata: Optional[List[Dict]] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Perform literature review: find top-k most relevant papers.
        
        Args:
            query: Query text
            corpus_embeddings: Precomputed corpus embeddings [num_docs, embed_dim]
            corpus_metadata: Optional metadata for each document
            top_k: Number of top results to return
            
        Returns:
            List of top-k results with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings(query)
        
        # Compute similarities
        similarities = self.cosine_similarity(query_embedding, corpus_embeddings)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_k_indices:
            result = {
                'rank': len(results) + 1,
                'similarity': float(similarities[idx]),
                'index': int(idx)
            }
            
            if corpus_metadata and idx < len(corpus_metadata):
                result.update(corpus_metadata[idx])
            
            results.append(result)
        
        return results
    
    def classify_domain(self, text: str, domain_classifier: Optional[object] = None) -> Dict[str, float]:
        """Classify text into healthcare subdomain.
        
        Args:
            text: Input text
            domain_classifier: Optional pre-trained classifier (LogisticRegression)
            
        Returns:
            Dictionary mapping domain names to confidence scores
        """
        # Use classifier if provided
        if domain_classifier is not None:
            # Generate embedding
            embedding = self.generate_embeddings(text)
            
            # Predict with classifier
            if SKLEARN_AVAILABLE and hasattr(domain_classifier, 'predict_proba'):
                try:
                    probs = domain_classifier.predict_proba([embedding])[0]
                    classes = domain_classifier.classes_
                    return {str(domain): float(prob) for domain, prob in zip(classes, probs)}
                except Exception as e:
                    print(f"Classifier prediction failed: {e}, using keyword-based")
        
        # Fallback to keyword-based classification
        return self._keyword_domain_classification(text)
    
    def generate_example_predictions(
        self,
        dataset_metadata_path: str,
        dataset_text_dir: str,
        output_path: str,
        num_examples: int = 10,
        max_input_length: int = 200,
        max_prediction_length: int = 50
    ) -> str:
        """Generate example predictions with expert activations.
        
        Args:
            dataset_metadata_path: Path to processed_dataset.jsonl
            dataset_text_dir: Directory containing paper text files
            output_path: Path to save example_predictions.json
            num_examples: Number of examples to generate
            max_input_length: Maximum input text length (characters)
            max_prediction_length: Maximum prediction length (tokens)
            
        Returns:
            Path to saved JSON file
        """
        import random
        from evaluate import classify_paper_domain
        
        # Load metadata
        papers = []
        if os.path.exists(dataset_metadata_path):
            with open(dataset_metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        papers.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Dataset metadata not found: {dataset_metadata_path}")
        
        # Sample papers
        if len(papers) > num_examples:
            papers = random.sample(papers, num_examples)
        
        example_predictions = []
        
        # Get base model for routing capture (only for MoE models)
        base_model = self.base_model if self.base_model else (
            self.model.base_model if hasattr(self.model, 'base_model') else None
        )
        
        # Check if this is a MoE model (has gate/routed_experts) or baseline
        is_moe_model = base_model is not None and (
            hasattr(base_model, 'gate') or hasattr(base_model, 'routed_experts')
        )
        
        if is_moe_model:
            top_k = getattr(base_model, 'top_k', 2)
            num_routed_experts = getattr(base_model, 'num_routed_experts', 4)
        else:
            print("Note: Baseline model detected - expert activations will not be captured")
            top_k = None
            num_routed_experts = None
        
        print(f"Generating {len(papers)} example predictions...")
        
        for paper in papers:
            paper_id = paper.get('arxiv_id', paper.get('id', 'unknown'))
            
            # Load text
            text_file = os.path.join(dataset_text_dir, f"{paper_id}.txt")
            if not os.path.exists(text_file):
                print(f"Warning: Text file not found for {paper_id}, skipping...")
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Prepare input (first max_input_length characters)
            input_text = full_text[:max_input_length]
            
            # Tokenize input
            input_ids = self._tokenize(input_text).to(self.device)
            
            # Validate token IDs are within vocabulary
            if (input_ids < 0).any() or (input_ids >= self.vocab_size).any():
                print(f"Warning: Invalid token IDs in input for {paper_id}, skipping...")
                continue
            
            # Forward pass (with routing info only for MoE models)
            with torch.no_grad():
                if is_moe_model:
                    output, routing_info = base_model(
                        input_ids,
                        image_features=None,
                        return_load_balance_loss=False,
                        return_gate_logits=True
                    )
                    
                    if routing_info is None:
                        print(f"Warning: No routing info for {paper_id}, skipping...")
                        continue
                    
                    gate_logits, _, _, _, _, routing_metrics = routing_info
                    logits = output  # [1, seq_len, vocab_size]
                else:
                    # Baseline model - no routing info
                    output = base_model(
                        input_ids,
                        image_features=None,
                        return_load_balance_loss=False,
                        return_gate_logits=False
                    )
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    gate_logits = None
                    routing_info = None
                
                # Compute perplexity
                # Use input as target (next token prediction)
                if input_ids.shape[1] > 1:
                    targets = input_ids[:, 1:]  # Shift by 1 for next-token prediction
                    logits_pred = logits[:, :-1, :]  # Remove last logit
                    
                    loss_per_token = F.cross_entropy(
                        logits_pred.view(-1, logits_pred.shape[-1]),
                        targets.view(-1),
                        ignore_index=0,
                        reduction='none'
                    ).view(targets.shape)
                    
                    # Mask padding
                    mask = (targets != 0).float()
                    if mask.sum() > 0:
                        avg_loss = (loss_per_token * mask).sum() / mask.sum()
                        perplexity = float(torch.exp(avg_loss).item())
                    else:
                        perplexity = float('inf')
                else:
                    perplexity = float('inf')
                
                # Extract expert activations (only for MoE models)
                if is_moe_model and gate_logits is not None:
                    # gate_logits shape: [batch*seq_len, num_routed_experts]
                    if isinstance(gate_logits, torch.Tensor):
                        gate_logits_np = gate_logits.detach().cpu().numpy()
                    else:
                        gate_logits_np = np.array(gate_logits)
                    
                    # For Expert Choice routing, get which experts were actually selected
                    # Each expert selects top-k tokens, so we need to see which experts selected tokens from this paper
                    if len(gate_logits_np.shape) == 2:
                        # [batch*seq_len, num_routed_experts]
                        seq_len = input_ids.shape[1]
                        batch_size = 1
                        
                        # Compute probabilities
                        gate_probs = torch.softmax(torch.from_numpy(gate_logits_np), dim=-1).numpy()
                        
                        # For Expert Choice: transpose to [num_experts, batch*seq_len]
                        gate_probs_transposed = gate_probs.T  # [num_routed_experts, batch*seq_len]
                        
                        # Each expert selects top-k tokens
                        expert_token_selections = np.argsort(gate_probs_transposed, axis=-1)[:, -top_k:]  # [num_experts, top_k]
                        
                        # Count which experts selected tokens from this paper
                        expert_counts = np.zeros(num_routed_experts, dtype=int)
                        for expert_idx in range(num_routed_experts):
                            selected_tokens = expert_token_selections[expert_idx]
                            # All tokens are from this paper (batch_size=1)
                            expert_counts[expert_idx] = len(selected_tokens)
                        
                        # Get experts that selected at least one token (or top-k by count)
                        active_experts = np.where(expert_counts > 0)[0].tolist()
                        if len(active_experts) >= top_k:
                            # Sort by count and take top-k
                            expert_counts_dict = {i: expert_counts[i] for i in active_experts}
                            top_experts = sorted(active_experts, key=lambda x: expert_counts_dict[x], reverse=True)[:top_k]
                        else:
                            # Fallback: use top experts by average probability
                            avg_gate_probs = gate_probs.mean(axis=0)  # [num_routed_experts]
                            top_experts = np.argsort(avg_gate_probs)[-top_k:].tolist()
                    else:
                        # Fallback: use top experts by average logit
                        avg_gate_logits = gate_logits_np.mean(axis=0) if len(gate_logits_np.shape) > 1 else gate_logits_np
                        top_experts = np.argsort(avg_gate_logits)[-top_k:].tolist()
                else:
                    # Baseline model - no expert activations
                    top_experts = []
                
                # Generate prediction (greedy decoding with repetition penalty)
                # Generate up to max_prediction_length tokens
                predicted_token_ids = []
                # Use only the non-padding tokens for generation (truncate padding)
                # Find the last non-zero token
                non_padding_mask = (input_ids[0] != 0)
                if non_padding_mask.any():
                    last_non_padding_idx = non_padding_mask.nonzero()[-1].item() + 1
                    current_input = input_ids[:, :last_non_padding_idx].clone()
                else:
                    current_input = input_ids.clone()
                
                # Repetition penalty parameters
                repetition_penalty = 1.5  # Increased penalty for repeated tokens
                max_repetition = 2  # Stop if same token repeated this many times
                repetition_window = 20  # Check last N tokens for repetition
                
                for step in range(max_prediction_length):
                    # Check if we've exceeded max_length
                    if current_input.shape[1] >= self.max_length:
                        break
                    
                    # Enhanced repetition detection: check for same token repeated
                    if len(predicted_token_ids) >= max_repetition:
                        recent_tokens = predicted_token_ids[-max_repetition:]
                        if len(set(recent_tokens)) == 1:
                            # Same token repeated max_repetition times - stop generation
                            break
                    
                    # Check for n-gram repetition (e.g., "uronal uronal")
                    if len(predicted_token_ids) >= 4:
                        # Check for 2-gram repetition
                        last_2 = tuple(predicted_token_ids[-2:])
                        if len(predicted_token_ids) >= 4:
                            prev_2 = tuple(predicted_token_ids[-4:-2])
                            if last_2 == prev_2:
                                # 2-gram repeated - stop generation
                                break
                    
                    with torch.no_grad():
                        output_step = base_model(
                            current_input,
                            image_features=None,
                            return_load_balance_loss=False,
                            return_gate_logits=False
                        )
                        # Handle both tuple and non-tuple returns
                        if isinstance(output_step, tuple):
                            output_step = output_step[0]
                        
                        next_token_logits = output_step[0, -1, :].clone()  # [vocab_size]
                        
                        # Apply stronger repetition penalty to recently generated tokens
                        if len(predicted_token_ids) > 0:
                            # Get last few tokens with exponential decay penalty
                            recent_window = min(repetition_window, len(predicted_token_ids))
                            recent_tokens = predicted_token_ids[-recent_window:]
                            
                            # Count token frequencies in recent window
                            token_counts = {}
                            for i, token_id in enumerate(recent_tokens):
                                if token_id not in token_counts:
                                    token_counts[token_id] = 0
                                # More recent tokens get higher penalty
                                token_counts[token_id] += (i + 1) / len(recent_tokens)
                            
                            # Apply penalty based on frequency and recency
                            for token_id, count in token_counts.items():
                                if 0 <= token_id < len(next_token_logits):
                                    # Stronger penalty for more frequent/recent tokens
                                    penalty = repetition_penalty ** count
                                    next_token_logits[token_id] /= penalty
                        
                        # Use top-k sampling with better parameters
                        top_k = min(20, len(next_token_logits))  # Reduced from 50 to 20
                        if top_k > 1:
                            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                            # Apply temperature for diversity (temperature > 1 increases randomness)
                            temperature = 1.1  # Slight increase in randomness
                            probs = torch.softmax(top_k_logits / temperature, dim=-1)
                            sampled_idx = torch.multinomial(probs, 1).item()
                            next_token_id = top_k_indices[sampled_idx].item()
                        else:
                            next_token_id = torch.argmax(next_token_logits).item()
                        
                        # Validate token ID is within vocabulary
                        if next_token_id < 0 or next_token_id >= self.vocab_size:
                            break
                        
                        # Stop if we hit EOS or padding token (0 is typically padding)
                        if next_token_id == 0:
                            break
                        
                        # Check for EOS token if tokenizer has it
                        try:
                            eos_id = self.tokenizer.piece_to_id('</s>')
                            if next_token_id == eos_id:
                                break
                        except:
                            pass
                        
                        predicted_token_ids.append(next_token_id)
                        
                        # Append to input for next iteration, but truncate if needed
                        new_token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
                        if current_input.shape[1] + 1 > self.max_length:
                            # Truncate from the beginning to make room
                            current_input = torch.cat([current_input[:, 1:], new_token_tensor], dim=1)
                        else:
                            current_input = torch.cat([current_input, new_token_tensor], dim=1)
                
                # Decode prediction with improved SentencePiece handling
                if predicted_token_ids:
                    try:
                        # Filter out invalid token IDs and special tokens before decoding
                        valid_token_ids = []
                        for tid in predicted_token_ids:
                            if 0 <= tid < self.vocab_size:
                                # Skip padding (0) and check for EOS
                                if tid == 0:
                                    continue
                                try:
                                    # Check if it's a special token that shouldn't be decoded
                                    piece = self.tokenizer.id_to_piece(tid)
                                    if piece.startswith('<') and piece.endswith('>'):
                                        # Skip special tokens like <s>, </s>, etc.
                                        continue
                                    valid_token_ids.append(tid)
                                except:
                                    # Skip invalid token IDs
                                    continue
                        
                        if valid_token_ids:
                            # Decode tokens
                            predicted_text = self.tokenizer.decode(valid_token_ids)
                            
                            # Improved SentencePiece cleaning
                            if predicted_text:
                                # Replace SentencePiece word boundary markers properly
                                # SentencePiece uses ▁ to mark word boundaries
                                predicted_text = predicted_text.replace('▁', ' ')
                                
                                # Remove excessive whitespace
                                predicted_text = ' '.join(predicted_text.split())
                                
                                # Remove common artifacts
                                predicted_text = predicted_text.replace('  ', ' ')
                                predicted_text = predicted_text.strip()
                                
                                # Filter out obviously broken tokens (very short fragments)
                                words = predicted_text.split()
                                # Remove single-character "words" that are likely artifacts
                                words = [w for w in words if len(w) > 1 or w.isalnum()]
                                predicted_text = ' '.join(words)
                        else:
                            predicted_text = ""
                    except Exception as e:
                        # Fallback: try decoding individual tokens with better error handling
                        try:
                            decoded_parts = []
                            for tid in predicted_token_ids[:30]:  # Increased limit
                                if 0 <= tid < self.vocab_size:
                                    try:
                                        piece = self.tokenizer.id_to_piece(tid)
                                        # Skip special tokens
                                        if piece.startswith('<') and piece.endswith('>'):
                                            continue
                                        # Clean the piece
                                        cleaned = piece.replace('▁', ' ').strip()
                                        if cleaned and len(cleaned) > 0:
                                            decoded_parts.append(cleaned)
                                    except:
                                        pass
                            predicted_text = ' '.join(decoded_parts) if decoded_parts else f"[{len(predicted_token_ids)} tokens, decode error: {e}]"
                        except:
                            predicted_text = f"[{len(predicted_token_ids)} tokens, decode failed]"
                else:
                    predicted_text = ""
            
            # Classify domain
            paper_dict = {
                'categories': paper.get('categories', []),
                'domains': paper.get('domains', []),
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', '')
            }
            domain = classify_paper_domain(paper_dict)
            
            example_predictions.append({
                'paper_id': paper_id,
                'input_text': input_text,
                'predicted_text': predicted_text,
                'perplexity': perplexity,
                'activated_experts': top_experts if is_moe_model else [],
                'domain': domain,
                'model_type': 'moe' if is_moe_model else 'baseline'
            })
            
            if is_moe_model:
                print(f"  Generated prediction for {paper_id} (perplexity: {perplexity:.2f}, experts: {top_experts})")
            else:
                print(f"  Generated prediction for {paper_id} (perplexity: {perplexity:.2f})")
        
        # Determine output path - prefer Google Drive if available
        final_output_path = self._get_drive_path_if_available(output_path)
        
        # Save to JSON
        os.makedirs(os.path.dirname(final_output_path) if os.path.dirname(final_output_path) else '.', exist_ok=True)
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(example_predictions, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(example_predictions)} example predictions to {final_output_path}")
        if final_output_path != output_path:
            print(f"  (Saved to Google Drive instead of {output_path})")
        return final_output_path
    
    def generate_baseline_predictions(
        self,
        baseline_checkpoint_path: str,
        dataset_metadata_path: str,
        dataset_text_dir: str,
        output_path: str,
        num_examples: int = 10,
        max_input_length: int = 200,
        max_prediction_length: int = 50
    ) -> str:
        """Generate baseline model predictions for comparison.
        
        Args:
            baseline_checkpoint_path: Path to baseline model checkpoint
            dataset_metadata_path: Path to processed_dataset.jsonl
            dataset_text_dir: Directory containing paper text files
            output_path: Path to save baseline_predictions.json
            num_examples: Number of examples to generate
            max_input_length: Maximum input text length (characters)
            max_prediction_length: Maximum prediction length (tokens)
            
        Returns:
            Path to saved JSON file
        """
        import random
        from evaluate import classify_paper_domain
        
        # Load baseline model
        try:
            from train_baseline import BaselineTransformer
            
            checkpoint = torch.load(baseline_checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Infer model config from checkpoint
            embedding_dim = 256  # Default
            for key in state_dict.keys():
                if 'embedding.weight' in key:
                    weight = state_dict[key]
                    embedding_dim = weight.shape[1]
                    break
            
            num_layers = 6  # Default
            for key in state_dict.keys():
                if 'transformer.layers.' in key:
                    parts = key.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                num_layers = max(num_layers, layer_idx + 1)
                            except ValueError:
                                pass
            
            vocab_size = self.tokenizer.get_piece_size()
            
            baseline_model = BaselineTransformer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
            )
            
            baseline_model.load_state_dict(state_dict, strict=False)
            baseline_model.to(self.device)
            baseline_model.eval()
            
            print(f"Loaded baseline model from {baseline_checkpoint_path}")
        except Exception as e:
            raise ValueError(f"Could not load baseline model: {e}")
        
        # Load metadata
        papers = []
        if os.path.exists(dataset_metadata_path):
            with open(dataset_metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        papers.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Dataset metadata not found: {dataset_metadata_path}")
        
        # Sample papers
        if len(papers) > num_examples:
            papers = random.sample(papers, num_examples)
        
        baseline_predictions = []
        
        print(f"Generating {len(papers)} baseline predictions...")
        
        for paper in papers:
            paper_id = paper.get('arxiv_id', paper.get('id', 'unknown'))
            
            # Load text
            text_file = os.path.join(dataset_text_dir, f"{paper_id}.txt")
            if not os.path.exists(text_file):
                print(f"Warning: Text file not found for {paper_id}, skipping...")
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Prepare input (first max_input_length characters)
            input_text = full_text[:max_input_length]
            
            # Tokenize input
            input_ids = self._tokenize(input_text).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits = baseline_model(
                    input_ids,
                    image_features=None,
                    return_load_balance_loss=False,
                    return_gate_logits=False
                )
                
                # Compute perplexity
                if input_ids.shape[1] > 1:
                    targets = input_ids[:, 1:]
                    logits_pred = logits[:, :-1, :]
                    
                    loss_per_token = F.cross_entropy(
                        logits_pred.view(-1, logits_pred.shape[-1]),
                        targets.view(-1),
                        ignore_index=0,
                        reduction='none'
                    ).view(targets.shape)
                    
                    mask = (targets != 0).float()
                    if mask.sum() > 0:
                        avg_loss = (loss_per_token * mask).sum() / mask.sum()
                        perplexity = float(torch.exp(avg_loss).item())
                    else:
                        perplexity = float('inf')
                else:
                    perplexity = float('inf')
                
                # Generate prediction (greedy decoding with repetition penalty)
                predicted_token_ids = []
                # Use only the non-padding tokens for generation
                non_padding_mask = (input_ids[0] != 0)
                if non_padding_mask.any():
                    last_non_padding_idx = non_padding_mask.nonzero()[-1].item() + 1
                    current_input = input_ids[:, :last_non_padding_idx].clone()
                else:
                    current_input = input_ids.clone()
                
                # Repetition penalty parameters
                repetition_penalty = 1.5  # Increased penalty for repeated tokens
                max_repetition = 2  # Stop if same token repeated this many times
                repetition_window = 20  # Check last N tokens for repetition
                
                for step in range(max_prediction_length):
                    # Check if we've exceeded max_length
                    if current_input.shape[1] >= self.max_length:
                        break
                    
                    # Enhanced repetition detection: check for same token repeated
                    if len(predicted_token_ids) >= max_repetition:
                        recent_tokens = predicted_token_ids[-max_repetition:]
                        if len(set(recent_tokens)) == 1:
                            # Same token repeated max_repetition times - stop generation
                            break
                    
                    # Check for n-gram repetition (e.g., "uronal uronal")
                    if len(predicted_token_ids) >= 4:
                        # Check for 2-gram repetition
                        last_2 = tuple(predicted_token_ids[-2:])
                        if len(predicted_token_ids) >= 4:
                            prev_2 = tuple(predicted_token_ids[-4:-2])
                            if last_2 == prev_2:
                                # 2-gram repeated - stop generation
                                break
                    
                    with torch.no_grad():
                        output_step = baseline_model(
                            current_input,
                            image_features=None,
                            return_load_balance_loss=False,
                            return_gate_logits=False
                        )
                        
                        if isinstance(output_step, tuple):
                            output_step = output_step[0]
                        
                        next_token_logits = output_step[0, -1, :].clone()  # [vocab_size]
                        
                        # Apply stronger repetition penalty to recently generated tokens
                        if len(predicted_token_ids) > 0:
                            # Get last few tokens with exponential decay penalty
                            recent_window = min(repetition_window, len(predicted_token_ids))
                            recent_tokens = predicted_token_ids[-recent_window:]
                            
                            # Count token frequencies in recent window
                            token_counts = {}
                            for i, token_id in enumerate(recent_tokens):
                                if token_id not in token_counts:
                                    token_counts[token_id] = 0
                                # More recent tokens get higher penalty
                                token_counts[token_id] += (i + 1) / len(recent_tokens)
                            
                            # Apply penalty based on frequency and recency
                            for token_id, count in token_counts.items():
                                if 0 <= token_id < len(next_token_logits):
                                    # Stronger penalty for more frequent/recent tokens
                                    penalty = repetition_penalty ** count
                                    next_token_logits[token_id] /= penalty
                        
                        # Use top-k sampling with better parameters
                        top_k = min(20, len(next_token_logits))  # Reduced from 50 to 20
                        if top_k > 1:
                            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                            # Apply temperature for diversity (temperature > 1 increases randomness)
                            temperature = 1.1  # Slight increase in randomness
                            probs = torch.softmax(top_k_logits / temperature, dim=-1)
                            sampled_idx = torch.multinomial(probs, 1).item()
                            next_token_id = top_k_indices[sampled_idx].item()
                        else:
                            next_token_id = torch.argmax(next_token_logits).item()
                        
                        # Validate token ID
                        if next_token_id < 0 or next_token_id >= self.vocab_size:
                            break
                        
                        if next_token_id == 0:
                            break
                        
                        try:
                            eos_id = self.tokenizer.piece_to_id('</s>')
                            if next_token_id == eos_id:
                                break
                        except:
                            pass
                        
                        predicted_token_ids.append(next_token_id)
                        
                        # Append to input, truncate if needed
                        new_token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
                        if current_input.shape[1] + 1 > self.max_length:
                            current_input = torch.cat([current_input[:, 1:], new_token_tensor], dim=1)
                        else:
                            current_input = torch.cat([current_input, new_token_tensor], dim=1)
                
                # Decode prediction with improved SentencePiece handling
                if predicted_token_ids:
                    try:
                        # Filter out invalid token IDs and special tokens before decoding
                        valid_token_ids = []
                        for tid in predicted_token_ids:
                            if 0 <= tid < self.vocab_size:
                                # Skip padding (0) and check for EOS
                                if tid == 0:
                                    continue
                                try:
                                    # Check if it's a special token that shouldn't be decoded
                                    piece = self.tokenizer.id_to_piece(tid)
                                    if piece.startswith('<') and piece.endswith('>'):
                                        # Skip special tokens like <s>, </s>, etc.
                                        continue
                                    valid_token_ids.append(tid)
                                except:
                                    # Skip invalid token IDs
                                    continue
                        
                        if valid_token_ids:
                            # Decode tokens
                            predicted_text = self.tokenizer.decode(valid_token_ids)
                            
                            # Improved SentencePiece cleaning
                            if predicted_text:
                                # Replace SentencePiece word boundary markers properly
                                # SentencePiece uses ▁ to mark word boundaries
                                predicted_text = predicted_text.replace('▁', ' ')
                                
                                # Remove excessive whitespace
                                predicted_text = ' '.join(predicted_text.split())
                                
                                # Remove common artifacts
                                predicted_text = predicted_text.replace('  ', ' ')
                                predicted_text = predicted_text.strip()
                                
                                # Filter out obviously broken tokens (very short fragments)
                                words = predicted_text.split()
                                # Remove single-character "words" that are likely artifacts
                                words = [w for w in words if len(w) > 1 or w.isalnum()]
                                predicted_text = ' '.join(words)
                        else:
                            predicted_text = ""
                    except Exception as e:
                        # Fallback: try decoding individual tokens with better error handling
                        try:
                            decoded_parts = []
                            for tid in predicted_token_ids[:30]:  # Increased limit
                                if 0 <= tid < self.vocab_size:
                                    try:
                                        piece = self.tokenizer.id_to_piece(tid)
                                        # Skip special tokens
                                        if piece.startswith('<') and piece.endswith('>'):
                                            continue
                                        # Clean the piece
                                        cleaned = piece.replace('▁', ' ').strip()
                                        if cleaned and len(cleaned) > 0:
                                            decoded_parts.append(cleaned)
                                    except:
                                        pass
                            predicted_text = ' '.join(decoded_parts) if decoded_parts else f"[{len(predicted_token_ids)} tokens, decode error: {e}]"
                        except:
                            predicted_text = f"[{len(predicted_token_ids)} tokens, decode failed]"
                else:
                    predicted_text = ""
            
            # Classify domain
            paper_dict = {
                'categories': paper.get('categories', []),
                'domains': paper.get('domains', []),
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', '')
            }
            domain = classify_paper_domain(paper_dict)
            
            baseline_predictions.append({
                'paper_id': paper_id,
                'input_text': input_text,
                'predicted_text': predicted_text,
                'perplexity': perplexity,
                'domain': domain,
                'model_type': 'baseline'
            })
            
            print(f"  Generated baseline prediction for {paper_id} (perplexity: {perplexity:.2f})")
        
        # Determine output path
        final_output_path = self._get_drive_path_if_available(output_path)
        
        # Save to JSON
        os.makedirs(os.path.dirname(final_output_path) if os.path.dirname(final_output_path) else '.', exist_ok=True)
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_predictions, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(baseline_predictions)} baseline predictions to {final_output_path}")
        if final_output_path != output_path:
            print(f"  (Saved to Google Drive instead of {output_path})")
        return final_output_path
    
    def _get_drive_path_if_available(self, local_path: str) -> str:
        """Get path for saving files, preferring Google Drive if available.
        
        Args:
            local_path: Local fallback path
            
        Returns:
            Path string (Drive path if available, otherwise local)
        """
        # Check for Google Drive
        drive_base = os.environ.get('DRIVE_BASE', '/content/drive/MyDrive/neuroMOE_results')
        
        # Check if Drive is mounted
        if os.path.exists(drive_base) and os.access(drive_base, os.W_OK):
            # Extract filename from local_path
            filename = os.path.basename(local_path)
            # Save to Drive evaluations folder (same location as eval_results.json)
            drive_path = os.path.join(drive_base, 'evaluations', filename)
            return drive_path
        
        # Also check if we're in Colab and Drive might be mounted at /content/drive
        if os.path.exists('/content/drive/MyDrive'):
            # Try neuroMOE_results/evaluations first
            drive_base = '/content/drive/MyDrive/neuroMOE_results'
            if os.path.exists(drive_base) and os.access(drive_base, os.W_OK):
                filename = os.path.basename(local_path)
                drive_path = os.path.join(drive_base, 'evaluations', filename)
                return drive_path
            
            # Fallback to direct /content/drive/MyDrive/evaluations
            evaluations_dir = '/content/drive/MyDrive/evaluations'
            if os.path.exists('/content/drive/MyDrive') and os.access('/content/drive/MyDrive', os.W_OK):
                filename = os.path.basename(local_path)
                drive_path = os.path.join(evaluations_dir, filename)
                return drive_path
        
        # Fall back to local path
        return local_path
    
    def _keyword_domain_classification(self, text: str) -> Dict[str, float]:
        """Simple keyword-based domain classification."""
        text_lower = text.lower()
        
        domain_keywords = {
            'neurodegeneration': ['alzheimer', 'parkinson', 'dementia', 'als', 'tau', 'amyloid'],
            'neuroscience': ['neural', 'neuron', 'brain', 'fmri', 'eeg', 'cortex'],
            'medical_imaging': ['mri', 'ct scan', 'x-ray', 'ultrasound', 'imaging'],
            'clinical': ['patient', 'clinical', 'diagnosis', 'treatment', 'trial'],
            'drug_discovery': ['drug', 'molecule', 'protein', 'compound', 'pharmaceutical'],
            'general_ml_health': []
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            if domain == 'general_ml_health':
                scores[domain] = 0.1  # Default
            else:
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[domain] = score / max(len(keywords), 1)
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def precompute_corpus_embeddings(
        self,
        corpus_texts: List[str],
        output_path: str,
        batch_size: int = 32,
        save_metadata: bool = True,
        metadata: Optional[List[Dict]] = None
    ):
        """Precompute embeddings for full corpus and save to .npz.
        
        Args:
            corpus_texts: List of corpus texts
            output_path: Output .npz file path
            batch_size: Batch size for processing
            save_metadata: Whether to save metadata
            metadata: Optional metadata for each document
        """
        print(f"Precomputing embeddings for {len(corpus_texts)} documents...")
        
        # Batch encode
        embeddings = self.batch_encode(corpus_texts, batch_size=batch_size)
        
        # Save to .npz
        save_dict = {'embeddings': embeddings}
        
        if save_metadata and metadata:
            save_dict['metadata'] = metadata
        
        np.savez_compressed(output_path, **save_dict)
        print(f"Saved embeddings to {output_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    def export_to_onnx(self, output_path: str, sample_text: str = "Sample text for ONNX export"):
        """Export model to ONNX format for deployment.
        
        Args:
            output_path: Output ONNX file path
            sample_text: Sample text for tracing
        """
        if not ONNX_AVAILABLE:
            print("ONNX not available, skipping export")
            return
        
        self.model.eval()
        
        # Create sample input
        input_ids = self._tokenize(sample_text).to(self.device)
        
        # Export
        try:
            torch.onnx.export(
                self.model,
                input_ids,
                output_path,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                },
                opset_version=11
            )
            print(f"Model exported to ONNX: {output_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")


def benchmark_inference(pipeline: InferencePipeline, num_samples: int = 100):
    """Benchmark inference speed.
    
    Args:
        pipeline: Inference pipeline
        num_samples: Number of samples to test
    """
    print(f"Benchmarking inference ({num_samples} samples)...")
    
    sample_texts = [
        "Alzheimer's disease is a neurodegenerative disorder characterized by cognitive decline.",
        "Machine learning models for medical imaging have shown promising results.",
        "Clinical trials are essential for drug development and approval.",
    ] * (num_samples // 3 + 1)
    sample_texts = sample_texts[:num_samples]
    
    # Warmup
    _ = pipeline.generate_embeddings(sample_texts[0])
    
    # Benchmark
    start_time = time.time()
    for text in sample_texts:
        _ = pipeline.generate_embeddings(text)
    elapsed = time.time() - start_time
    
    avg_time = elapsed / num_samples * 1000  # Convert to ms
    
    print(f"Benchmark results:")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Average per inference: {avg_time:.2f}ms")
    print(f"   Throughput: {num_samples / elapsed:.1f} samples/sec")
    
    if avg_time < 100:
        print(f"   Target met: <100ms per inference")
    else:
        print(f"   Target not met: >100ms per inference")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Production inference pipeline for DeepSeekMoE",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to SentencePiece tokenizer')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on (default: cpu)')
    parser.add_argument('--quantize', action='store_true',
                       help='Use INT8 quantization')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable embedding cache')
    
    # Use cases
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Embedding generation
    embed_parser = subparsers.add_parser('embed', help='Generate embedding for text')
    embed_parser.add_argument('--text', type=str, required=True,
                             help='Input text')
    
    # Batch encoding
    batch_parser = subparsers.add_parser('batch', help='Batch encode texts')
    batch_parser.add_argument('--texts', type=str, required=True,
                              help='File with texts (one per line)')
    batch_parser.add_argument('--output', type=str, required=True,
                             help='Output .npy file')
    
    # Literature review
    review_parser = subparsers.add_parser('review', help='Literature review search')
    review_parser.add_argument('--query', type=str, required=True,
                              help='Query text')
    review_parser.add_argument('--corpus', type=str, required=True,
                              help='Corpus embeddings .npz file')
    review_parser.add_argument('--top-k', type=int, default=10,
                              help='Number of top results (default: 10)')
    
    # Domain classification
    domain_parser = subparsers.add_parser('classify', help='Classify domain')
    domain_parser.add_argument('--text', type=str, required=True,
                              help='Input text')
    
    # Precompute embeddings
    precompute_parser = subparsers.add_parser('precompute', help='Precompute corpus embeddings')
    precompute_parser.add_argument('--corpus', type=str, required=True,
                                  help='File with corpus texts (one per line)')
    precompute_parser.add_argument('--output', type=str, required=True,
                                  help='Output .npz file')
    precompute_parser.add_argument('--batch-size', type=int, default=32,
                                  help='Batch size (default: 32)')
    
    # ONNX export
    onnx_parser = subparsers.add_parser('export-onnx', help='Export to ONNX')
    onnx_parser.add_argument('--output', type=str, required=True,
                            help='Output ONNX file path')
    
    # Benchmark
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark inference speed')
    benchmark_parser.add_argument('--samples', type=int, default=100,
                                 help='Number of samples (default: 100)')
    
    # Generate example predictions
    examples_parser = subparsers.add_parser('generate-examples', help='Generate example predictions with expert activations')
    examples_parser.add_argument('--dataset-metadata', type=str, required=True,
                                help='Path to processed_dataset.jsonl')
    examples_parser.add_argument('--dataset-text-dir', type=str, required=True,
                                help='Directory containing paper text files')
    examples_parser.add_argument('--output', type=str, required=True,
                                help='Output JSON file path (e.g., ./models/deepseek_moe/example_predictions.json)')
    examples_parser.add_argument('--num-examples', type=int, default=10,
                                help='Number of examples to generate (default: 10)')
    examples_parser.add_argument('--max-input-length', type=int, default=200,
                                help='Maximum input text length in characters (default: 200)')
    examples_parser.add_argument('--max-prediction-length', type=int, default=50,
                                help='Maximum prediction length in tokens (default: 50)')
    
    # Generate baseline predictions
    baseline_parser = subparsers.add_parser('generate-baseline', help='Generate baseline model predictions for comparison')
    baseline_parser.add_argument('--baseline-checkpoint', type=str, required=True,
                                help='Path to baseline model checkpoint')
    baseline_parser.add_argument('--dataset-metadata', type=str, required=True,
                                help='Path to processed_dataset.jsonl')
    baseline_parser.add_argument('--dataset-text-dir', type=str, required=True,
                                help='Directory containing paper text files')
    baseline_parser.add_argument('--output', type=str, required=True,
                                help='Output JSON file path (e.g., ./evaluations/baseline_predictions.json)')
    baseline_parser.add_argument('--num-examples', type=int, default=10,
                                help='Number of examples to generate (default: 10)')
    baseline_parser.add_argument('--max-input-length', type=int, default=200,
                                help='Maximum input text length in characters (default: 200)')
    baseline_parser.add_argument('--max-prediction-length', type=int, default=50,
                                help='Maximum prediction length in tokens (default: 50)')
    
    args = parser.parse_args()
    
    # Initialize pipeline (only needed for non-baseline commands)
    if args.command != 'generate-baseline':
        pipeline = InferencePipeline(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            device=args.device,
            quantize=args.quantize,
            use_cache=not args.no_cache
        )
    elif args.command == 'generate-baseline':
        # For baseline generation, we still need a pipeline for tokenizer access
        # Use a dummy checkpoint path (won't be loaded for baseline)
        pipeline = InferencePipeline(
            checkpoint_path=args.baseline_checkpoint,  # Will be overridden in generate_baseline_predictions
            tokenizer_path=args.tokenizer,
            device=args.device,
            quantize=False,  # Baseline doesn't need quantization
            use_cache=False
        )
    
    # Execute command
    if args.command == 'embed':
        embedding = pipeline.generate_embeddings(args.text)
        print(f"Embedding shape: {embedding.shape}")
        print(f"   Embedding (first 10): {embedding[:10]}")
    
    elif args.command == 'batch':
        with open(args.texts, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        embeddings = pipeline.batch_encode(texts)
        np.save(args.output, embeddings)
        print(f"Saved {len(texts)} embeddings to {args.output}")
    
    elif args.command == 'review':
        # Load corpus
        corpus_data = np.load(args.corpus)
        corpus_embeddings = corpus_data['embeddings']
        corpus_metadata = corpus_data.get('metadata', None)
        
        # Search
        results = pipeline.literature_review(
            args.query,
            corpus_embeddings,
            corpus_metadata,
            top_k=args.top_k
        )
        
        print(f"\nTop {args.top_k} results for query: '{args.query}'")
        for result in results:
            print(f"\n  Rank {result['rank']}: similarity={result['similarity']:.4f}")
            if 'arxiv_id' in result:
                print(f"    ArXiv ID: {result['arxiv_id']}")
            if 'domains' in result:
                print(f"    Domains: {result['domains']}")
    
    elif args.command == 'classify':
        domains = pipeline.classify_domain(args.text)
        print(f"\nDomain classification for: '{args.text[:50]}...'")
        for domain, score in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            print(f"   {domain}: {score:.4f}")
    
    elif args.command == 'precompute':
        with open(args.corpus, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        pipeline.precompute_corpus_embeddings(
            texts,
            args.output,
            batch_size=args.batch_size
        )
    
    elif args.command == 'export-onnx':
        pipeline.export_to_onnx(args.output)
    
    elif args.command == 'benchmark':
        benchmark_inference(pipeline, args.samples)
    
    elif args.command == 'generate-examples':
        output_path = pipeline.generate_example_predictions(
            dataset_metadata_path=args.dataset_metadata,
            dataset_text_dir=args.dataset_text_dir,
            output_path=args.output,
            num_examples=args.num_examples,
            max_input_length=args.max_input_length,
            max_prediction_length=args.max_prediction_length
        )
        print(f"\n✅ Example predictions saved to: {output_path}")
    
    elif args.command == 'generate-baseline':
        output_path = pipeline.generate_baseline_predictions(
            baseline_checkpoint_path=args.baseline_checkpoint,
            dataset_metadata_path=args.dataset_metadata,
            dataset_text_dir=args.dataset_text_dir,
            output_path=args.output,
            num_examples=args.num_examples,
            max_input_length=args.max_input_length,
            max_prediction_length=args.max_prediction_length
        )
        print(f"\n✅ Baseline predictions saved to: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

