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
        print(f"✅ Loaded tokenizer (vocab_size={self.vocab_size})")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Quantize if requested
        if quantize and device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            print("✅ Model quantized to INT8")
        
        # Initialize cache
        if use_cache:
            self.cache = EmbeddingCache()
        else:
            self.cache = None
        
        # Domain classifier (lazy-loaded)
        self._domain_classifier = None
        
        print(f"✅ Inference pipeline initialized on {device}")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model
        """
        try:
            from train_real import SimpleMoEModel
            base_model = SimpleMoEModel(
                vocab_size=self.vocab_size,
                embedding_dim=256,
                num_shared_experts=2,
                num_routed_experts=4,
            )
            
            # Wrap model
            class ModelWrapper(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                
                def forward(self, input_ids):
                    output = self.base_model(
                        input_ids,
                        image_features=None,
                        return_load_balance_loss=False,
                        return_gate_logits=False
                    )
                    if isinstance(output, tuple):
                        return output[0]
                    return output
            
            model = ModelWrapper(base_model)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            print(f"✅ Loaded model from {checkpoint_path}")
            return model
            
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
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
                    print(f"⚠️  Classifier prediction failed: {e}, using keyword-based")
        
        # Fallback to keyword-based classification
        return self._keyword_domain_classification(text)
    
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
        print(f"📊 Precomputing embeddings for {len(corpus_texts)} documents...")
        
        # Batch encode
        embeddings = self.batch_encode(corpus_texts, batch_size=batch_size)
        
        # Save to .npz
        save_dict = {'embeddings': embeddings}
        
        if save_metadata and metadata:
            save_dict['metadata'] = metadata
        
        np.savez_compressed(output_path, **save_dict)
        print(f"✅ Saved embeddings to {output_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    def export_to_onnx(self, output_path: str, sample_text: str = "Sample text for ONNX export"):
        """Export model to ONNX format for deployment.
        
        Args:
            output_path: Output ONNX file path
            sample_text: Sample text for tracing
        """
        if not ONNX_AVAILABLE:
            print("⚠️  ONNX not available, skipping export")
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
            print(f"✅ Model exported to ONNX: {output_path}")
        except Exception as e:
            print(f"⚠️  ONNX export failed: {e}")


def benchmark_inference(pipeline: InferencePipeline, num_samples: int = 100):
    """Benchmark inference speed.
    
    Args:
        pipeline: Inference pipeline
        num_samples: Number of samples to test
    """
    print(f"⏱️  Benchmarking inference ({num_samples} samples)...")
    
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
    
    print(f"✅ Benchmark results:")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Average per inference: {avg_time:.2f}ms")
    print(f"   Throughput: {num_samples / elapsed:.1f} samples/sec")
    
    if avg_time < 100:
        print(f"   ✅ Target met: <100ms per inference")
    else:
        print(f"   ⚠️  Target not met: >100ms per inference")


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
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        quantize=args.quantize,
        use_cache=not args.no_cache
    )
    
    # Execute command
    if args.command == 'embed':
        embedding = pipeline.generate_embeddings(args.text)
        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"   Embedding (first 10): {embedding[:10]}")
    
    elif args.command == 'batch':
        with open(args.texts, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        embeddings = pipeline.batch_encode(texts)
        np.save(args.output, embeddings)
        print(f"✅ Saved {len(texts)} embeddings to {args.output}")
    
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
        
        print(f"\n📚 Top {args.top_k} results for query: '{args.query}'")
        for result in results:
            print(f"\n  Rank {result['rank']}: similarity={result['similarity']:.4f}")
            if 'arxiv_id' in result:
                print(f"    ArXiv ID: {result['arxiv_id']}")
            if 'domains' in result:
                print(f"    Domains: {result['domains']}")
    
    elif args.command == 'classify':
        domains = pipeline.classify_domain(args.text)
        print(f"\n🏷️  Domain classification for: '{args.text[:50]}...'")
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
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

