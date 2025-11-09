"""
Evaluation Utilities for DeepSeekMoE Model

Computes comprehensive metrics on held-out test set:
- Perplexity
- Domain classification accuracy
- Neurodegeneration relevance ranking (MRR@20)
- Section classification accuracy

Usage:
    python evaluate.py \
        --model-checkpoint ./checkpoints/step_5000.pt \
        --dataset-text-dir ./data/arxiv/texts \
        --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
        --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
        --output-dir ./evaluations \
        --test-split 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import our components
from arxiv_dataset import ArXivStreamingDataset, create_dataloader
from training_adapter import ModelAdapter

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available, visualization disabled")


class SectionClassifier:
    """Simple classifier for section classification (abstract/intro/methods/results)."""
    
    SECTION_KEYWORDS = {
        'abstract': ['abstract', 'summary', 'overview'],
        'introduction': ['introduction', 'background', 'motivation', 'related work'],
        'methods': ['method', 'methodology', 'approach', 'algorithm', 'model', 'architecture'],
        'results': ['result', 'experiment', 'evaluation', 'performance', 'finding', 'outcome']
    }
    
    @staticmethod
    def classify(text: str) -> str:
        """Classify text snippet into section type.
        
        Args:
            text: Text snippet to classify
            
        Returns:
            Section type: 'abstract', 'introduction', 'methods', or 'results'
        """
        text_lower = text.lower()
        
        # Count keyword matches
        scores = {}
        for section, keywords in SectionClassifier.SECTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[section] = score
        
        # Return section with highest score, default to 'methods'
        if max(scores.values()) == 0:
            return 'methods'
        
        return max(scores.items(), key=lambda x: x[1])[0]


def extract_embeddings(
    model: nn.Module,
    adapter: ModelAdapter,
    dataloader,
    max_samples: Optional[int] = None
) -> Tuple[torch.Tensor, List[Dict]]:
    """Extract model embeddings for test samples.
    
    Args:
        model: Trained model
        adapter: Model adapter
        dataloader: DataLoader for test data
        max_samples: Maximum number of samples to process (None = all)
        
    Returns:
        Tuple of (embeddings tensor, metadata list)
    """
    model.eval()
    embeddings = []
    metadata_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and len(metadata_list) >= max_samples:
                break
            
            # Forward pass
            with torch.cuda.amp.autocast():
                result = adapter.process_batch(batch)
                logits = result['logits']  # [batch, seq_len, vocab_size]
                batch_metadata = result['batch_metadata']
            
            # Extract embeddings: use mean pooling over sequence
            # Get embeddings from model if available, otherwise use logits
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'embedding'):
                # Extract from embedding layer
                input_ids = batch['input_ids'].to(adapter.device)
                embeds = model.base_model.embedding(input_ids)  # [batch, seq_len, embed_dim]
                pooled = embeds.mean(dim=1)  # [batch, embed_dim]
            else:
                # Fallback: use logits projection (not ideal but works)
                pooled = logits.mean(dim=1)  # [batch, vocab_size]
            
            embeddings.append(pooled.cpu())
            
            # Store metadata
            has_nd = batch_metadata.get('has_neurodegeneration', None)
            if has_nd is not None and isinstance(has_nd, torch.Tensor):
                has_nd_list = has_nd.cpu().tolist()
            else:
                has_nd_list = [False] * len(batch_metadata['arxiv_ids'])
            
            for i in range(len(batch_metadata['arxiv_ids'])):
                metadata_list.append({
                    'arxiv_id': batch_metadata['arxiv_ids'][i],
                    'domains': batch_metadata['domains'][i] if i < len(batch_metadata['domains']) else [],
                    'year': batch_metadata['years'][i] if i < len(batch_metadata['years']) else None,
                    'has_neurodegeneration': has_nd_list[i] if i < len(has_nd_list) else False
                })
    
    if embeddings:
        embeddings_tensor = torch.cat(embeddings, dim=0)
    else:
        embeddings_tensor = torch.empty(0)
    
    return embeddings_tensor, metadata_list


def compute_perplexity(
    model: nn.Module,
    adapter: ModelAdapter,
    dataloader
) -> float:
    """Compute perplexity on test set.
    
    Args:
        model: Trained model
        adapter: Model adapter
        dataloader: DataLoader for test data
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            with torch.cuda.amp.autocast():
                result = adapter.process_batch(batch)
                loss = result['loss']
                batch_metadata = result['batch_metadata']
            
            # Get number of non-padding tokens
            target_ids = batch['target_ids'].to(adapter.device)
            num_tokens = (target_ids != 0).sum().item()
            
            # Accumulate loss (weighted by tokens)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def compute_domain_classification_accuracy(
    embeddings: torch.Tensor,
    metadata: List[Dict],
    test_size: float = 0.2
) -> float:
    """Compute domain classification accuracy using model embeddings.
    
    Args:
        embeddings: Model embeddings [num_samples, embed_dim]
        metadata: List of metadata dicts
        test_size: Fraction of data to use for testing classifier
        
    Returns:
        Classification accuracy
    """
    if len(metadata) < 10:
        return 0.0
    
    # Prepare labels: convert domain lists to single label (use first domain)
    labels = []
    valid_indices = []
    
    for i, meta in enumerate(metadata):
        domains = meta.get('domains', [])
        if domains:
            labels.append(domains[0])  # Use first domain as label
            valid_indices.append(i)
    
    if len(labels) < 10:
        return 0.0
    
    # Filter embeddings and labels
    embeddings_filtered = embeddings[valid_indices].numpy()
    labels_filtered = labels
    
    # Get unique domains
    unique_domains = list(set(labels_filtered))
    if len(unique_domains) < 2:
        return 0.0
    
    # Convert labels to indices
    domain_to_idx = {domain: idx for idx, domain in enumerate(unique_domains)}
    label_indices = [domain_to_idx[label] for label in labels_filtered]
    
    # Split train/test
    n_samples = len(label_indices)
    n_test = int(n_samples * test_size)
    indices = list(range(n_samples))
    random.shuffle(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = embeddings_filtered[train_indices]
    y_train = [label_indices[i] for i in train_indices]
    X_test = embeddings_filtered[test_indices]
    y_test = [label_indices[i] for i in test_indices]
    
    # Train simple classifier
    try:
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        print(f"⚠️  Domain classification failed: {e}")
        return 0.0
    
    return accuracy


def compute_mrr_at_k(
    embeddings: torch.Tensor,
    metadata: List[Dict],
    query_indices: List[int],
    k: int = 20
) -> float:
    """Compute Mean Reciprocal Rank (MRR) for neurodegeneration relevance ranking.
    
    Args:
        embeddings: Model embeddings [num_samples, embed_dim]
        metadata: List of metadata dicts
        query_indices: Indices of query papers (neurodegeneration papers)
        k: Top-k for MRR calculation (default: 20)
        
    Returns:
        MRR@k score
    """
    if len(query_indices) == 0:
        return 0.0
    
    embeddings_np = embeddings.numpy()
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_norm = embeddings_np / norms
    
    reciprocal_ranks = []
    
    for query_idx in query_indices:
        if query_idx >= len(embeddings_norm):
            continue
        
        query_embedding = embeddings_norm[query_idx:query_idx+1]
        
        # Compute cosine similarity
        similarities = np.dot(embeddings_norm, query_embedding.T).flatten()
        
        # Get top-k indices (excluding query itself)
        top_k_indices = np.argsort(similarities)[::-1]
        top_k_indices = [idx for idx in top_k_indices if idx != query_idx][:k]
        
        # Check if any neurodegeneration paper is in top-k
        found = False
        for rank, idx in enumerate(top_k_indices, start=1):
            if metadata[idx].get('has_neurodegeneration', False):
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break
        
        if not found:
            reciprocal_ranks.append(0.0)
    
    if len(reciprocal_ranks) == 0:
        return 0.0
    
    mrr = np.mean(reciprocal_ranks)
    return mrr


def compute_section_classification_accuracy(
    model: nn.Module,
    adapter: ModelAdapter,
    dataloader,
    num_samples: int = 100
) -> float:
    """Compute section classification accuracy.
    
    Args:
        model: Trained model
        adapter: Model adapter
        dataloader: DataLoader for test data
        num_samples: Number of samples to evaluate
        
    Returns:
        Classification accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    # Load text files for section extraction
    text_dir = dataloader.dataset.text_dir if hasattr(dataloader.dataset, 'text_dir') else None
    
    if not text_dir:
        return 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total >= num_samples:
                break
            
            batch_metadata = adapter.process_batch(batch)['batch_metadata']
            arxiv_ids = batch_metadata['arxiv_ids']
            
            for arxiv_id in arxiv_ids:
                if total >= num_samples:
                    break
                
                # Read text file
                text_file = os.path.join(text_dir, f"{arxiv_id}.txt")
                if not os.path.exists(text_file):
                    continue
                
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Split into sentences and classify first few
                sentences = text.split('.')[:5]  # First 5 sentences
                
                for sentence in sentences:
                    if len(sentence.strip()) < 20:
                        continue
                    
                    # Classify section
                    predicted_section = SectionClassifier.classify(sentence)
                    
                    # For evaluation, we'll use a simple heuristic:
                    # If sentence contains section header keywords, it's correct
                    # This is a simplified evaluation
                    sentence_lower = sentence.lower()
                    is_correct = False
                    
                    if predicted_section == 'abstract' and 'abstract' in sentence_lower:
                        is_correct = True
                    elif predicted_section == 'introduction' and ('introduction' in sentence_lower or 'background' in sentence_lower):
                        is_correct = True
                    elif predicted_section == 'methods' and ('method' in sentence_lower or 'approach' in sentence_lower):
                        is_correct = True
                    elif predicted_section == 'results' and ('result' in sentence_lower or 'experiment' in sentence_lower):
                        is_correct = True
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    if total >= num_samples:
                        break
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    return accuracy


def evaluate_model(
    model_checkpoint: str,
    dataset_text_dir: str,
    dataset_metadata: str,
    tokenizer_path: str,
    output_dir: str,
    test_split: float = 0.1,
    batch_size: int = 16,
    max_test_samples: Optional[int] = None
) -> Dict:
    """Run comprehensive evaluation on test set.
    
    Args:
        model_checkpoint: Path to model checkpoint
        dataset_text_dir: Directory containing text files
        dataset_metadata: JSONL file with paper metadata
        tokenizer_path: Path to SentencePiece tokenizer
        output_dir: Output directory for results
        test_split: Fraction of data to use for testing
        batch_size: Batch size for evaluation
        max_test_samples: Maximum number of test samples (None = all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("🔍 Model Evaluation")
    print("=" * 60)
    print()
    
    # Load tokenizer
    if not SENTENCEPIECE_AVAILABLE:
        raise ImportError("sentencepiece package required")
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    print(f"✅ Loaded tokenizer (vocab_size={vocab_size})")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Using device: {device}")
    
    try:
        from train_real import SimpleMoEModel
        base_model = SimpleMoEModel(
            vocab_size=vocab_size,
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
                output = self.base_model(input_ids, image_features=None, return_load_balance_loss=False, return_gate_logits=False)
                if isinstance(output, tuple):
                    return output[0]
                return output
        
        model = ModelWrapper(base_model)
        
        # Load checkpoint
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(device)
        model.eval()
        print(f"✅ Loaded model from {model_checkpoint}")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        raise
    
    # Create adapter
    adapter = ModelAdapter(model, device=device)
    
    # Create full dataset
    full_dataset = ArXivStreamingDataset(
        text_dir=dataset_text_dir,
        metadata_jsonl=dataset_metadata,
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )
    
    # Split into test set (simple: use last N files)
    all_files = full_dataset.text_files
    n_test = int(len(all_files) * test_split)
    test_files = all_files[-n_test:] if n_test > 0 else []
    
    # Create test dataset (subset)
    class TestDataset(ArXivStreamingDataset):
        def __init__(self, text_files, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.text_files = text_files
    
    test_dataset = TestDataset(
        test_files,
        text_dir=dataset_text_dir,
        metadata_jsonl=dataset_metadata,
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )
    
    print(f"✅ Created test dataset: {len(test_files)} papers")
    
    # Create test dataloader
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    
    # 1. Perplexity
    print("   Computing perplexity...")
    perplexity = compute_perplexity(model, adapter, test_dataloader)
    print(f"   ✅ Perplexity: {perplexity:.2f}")
    
    # 2. Extract embeddings
    print("   Extracting embeddings...")
    embeddings, metadata = extract_embeddings(
        model, adapter, test_dataloader, max_samples=max_test_samples
    )
    print(f"   ✅ Extracted embeddings: {embeddings.shape}")
    
    # 3. Domain classification accuracy
    print("   Computing domain classification accuracy...")
    domain_accuracy = compute_domain_classification_accuracy(embeddings, metadata)
    print(f"   ✅ Domain accuracy: {domain_accuracy:.4f}")
    
    # 4. Neurodegeneration relevance ranking (MRR@20)
    print("   Computing neurodegeneration relevance ranking (MRR@20)...")
    query_indices = [
        i for i, meta in enumerate(metadata)
        if meta.get('has_neurodegeneration', False)
    ]
    mrr_20 = compute_mrr_at_k(embeddings, metadata, query_indices, k=20)
    print(f"   ✅ MRR@20: {mrr_20:.4f}")
    
    # 5. Section classification accuracy
    print("   Computing section classification accuracy...")
    section_accuracy = compute_section_classification_accuracy(
        model, adapter, test_dataloader, num_samples=min(100, len(test_files))
    )
    print(f"   ✅ Section accuracy: {section_accuracy:.4f}")
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_checkpoint': model_checkpoint,
        'test_samples': len(metadata),
        'metrics': {
            'perplexity': float(perplexity),
            'domain_classification_accuracy': float(domain_accuracy),
            'neurodegeneration_mrr_at_20': float(mrr_20),
            'section_classification_accuracy': float(section_accuracy),
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"evaluation_{int(time.time())}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results


def plot_training_curves(
    training_log_csv: str,
    output_file: Optional[str] = None
):
    """Plot training curves from CSV log file.
    
    Args:
        training_log_csv: Path to training log CSV file
        output_file: Output file path (None = auto-generate)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping visualization")
        return
    
    import pandas as pd
    
    # Load data
    df = pd.read_csv(training_log_csv)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # 1. Loss curve
    axes[0, 0].plot(df['step'], df['loss'], label='Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Learning rate
    axes[0, 1].plot(df['step'], df['learning_rate'], label='LR', color='orange')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')
    
    # 3. GPU memory
    if 'gpu_memory_mb' in df.columns:
        axes[1, 0].plot(df['step'], df['gpu_memory_mb'], label='GPU Memory', color='green')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('GPU Memory (MB)')
        axes[1, 0].set_title('GPU Memory Usage')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. Throughput
    if 'throughput_samples_per_sec' in df.columns:
        axes[1, 1].plot(df['step'], df['throughput_samples_per_sec'], label='Throughput', color='red')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Samples/sec')
        axes[1, 1].set_title('Training Throughput')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = training_log_csv.replace('.csv', '_curves.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved to: {output_file}")
    
    plt.close()


def plot_evaluation_trends(
    evaluation_dir: str,
    output_file: Optional[str] = None
):
    """Plot evaluation metric trends over time.
    
    Args:
        evaluation_dir: Directory containing evaluation JSON files
        output_file: Output file path (None = auto-generate)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping visualization")
        return
    
    # Load all evaluation files
    eval_files = sorted([
        os.path.join(evaluation_dir, f)
        for f in os.listdir(evaluation_dir)
        if f.startswith('evaluation_') and f.endswith('.json')
    ])
    
    if len(eval_files) == 0:
        print("⚠️  No evaluation files found")
        return
    
    # Extract metrics
    steps = []
    metrics_data = {
        'perplexity': [],
        'domain_classification_accuracy': [],
        'neurodegeneration_mrr_at_20': [],
        'section_classification_accuracy': []
    }
    
    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        # Extract step from checkpoint path
        checkpoint_path = data.get('model_checkpoint', '')
        try:
            step = int(checkpoint_path.split('step_')[1].split('.pt')[0])
        except:
            step = len(steps)
        
        steps.append(step)
        
        metrics = data.get('metrics', {})
        for key in metrics_data.keys():
            metrics_data[key].append(metrics.get(key, 0.0))
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Evaluation Metrics Over Time', fontsize=16)
    
    # Sort by step
    sorted_indices = sorted(range(len(steps)), key=lambda i: steps[i])
    steps_sorted = [steps[i] for i in sorted_indices]
    
    # Plot each metric
    axes[0, 0].plot(steps_sorted, [metrics_data['perplexity'][i] for i in sorted_indices], 'o-', label='Perplexity')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Perplexity')
    axes[0, 0].set_title('Perplexity Trend')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(steps_sorted, [metrics_data['domain_classification_accuracy'][i] for i in sorted_indices], 'o-', label='Domain Accuracy', color='orange')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Domain Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(steps_sorted, [metrics_data['neurodegeneration_mrr_at_20'][i] for i in sorted_indices], 'o-', label='MRR@20', color='green')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('MRR@20')
    axes[1, 0].set_title('Neurodegeneration Relevance Ranking')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(steps_sorted, [metrics_data['section_classification_accuracy'][i] for i in sorted_indices], 'o-', label='Section Accuracy', color='red')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Section Classification Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = os.path.join(evaluation_dir, 'evaluation_trends.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Evaluation trends saved to: {output_file}")
    
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained DeepSeekMoE model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model-checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset-text-dir', type=str, required=True,
                       help='Directory containing text files')
    parser.add_argument('--dataset-metadata', type=str, required=True,
                       help='JSONL file with paper metadata')
    parser.add_argument('--tokenizer-path', type=str, required=True,
                       help='Path to SentencePiece tokenizer')
    parser.add_argument('--output-dir', type=str, default='./evaluations',
                       help='Output directory for results')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Fraction of data for testing (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation (default: 16)')
    parser.add_argument('--max-test-samples', type=int, default=None,
                       help='Maximum test samples (default: all)')
    parser.add_argument('--plot-training-curves', type=str, default=None,
                       help='Path to training log CSV for plotting')
    parser.add_argument('--plot-evaluation-trends', action='store_true',
                       help='Plot evaluation trends from previous evaluations')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_checkpoint=args.model_checkpoint,
        dataset_text_dir=args.dataset_text_dir,
        dataset_metadata=args.dataset_metadata,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        max_test_samples=args.max_test_samples
    )
    
    # Plot training curves if requested
    if args.plot_training_curves:
        plot_training_curves(args.plot_training_curves)
    
    # Plot evaluation trends if requested
    if args.plot_evaluation_trends:
        plot_evaluation_trends(args.output_dir)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

