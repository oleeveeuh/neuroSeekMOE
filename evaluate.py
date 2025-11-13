"""
Evaluation Utilities for DeepSeekMoE Model

Computes comprehensive metrics on held-out test set:
- Perplexity
- Domain classification accuracy
- Neurodegeneration relevance ranking (MRR@20)
- Section classification accuracy
- Expert activation patterns (automatically captured and saved)

The evaluation script automatically captures expert routing decisions during inference
and saves them to expert_activations.npz for analysis in the model_analysis.ipynb notebook.

Usage:
    python evaluate.py \
        --model-checkpoint ./checkpoints/step_5000.pt \
        --dataset-text-dir ./data/arxiv/texts \
        --dataset-metadata ./data/arxiv/processed_dataset.jsonl \
        --tokenizer-path ./data/arxiv/healthcare_tokenizer.model \
        --output-dir ./evaluations \
        --test-split 0.1

Outputs:
    - evaluations/eval_results.json: Evaluation metrics
    - models/deepseek_moe/expert_activations.npz: Expert activation patterns (auto-generated)
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
from tqdm import tqdm

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
    print("matplotlib not available, visualization disabled")


def classify_paper_domain(paper: Dict) -> str:
    """Classify paper as ML, Healthcare, Both, or Other.
    
    Args:
        paper: Dictionary with 'categories'/'domains', 'title', 'abstract' fields
        
    Returns:
        Domain label: 'ML', 'Healthcare', 'Both', or 'Other'
    """
    # Handle both 'categories' and 'domains' field names
    categories = paper.get('categories', paper.get('domains', []))
    if not isinstance(categories, list):
        categories = [categories] if categories else []
    
    title = paper.get('title', '').lower() if paper.get('title') else ''
    abstract = paper.get('abstract', '').lower() if paper.get('abstract') else ''
    text = title + ' ' + abstract
    
    # Check categories (ArXiv format: 'cs.CV', 'q-bio.NC', etc.)
    # Also check for processed domain labels like 'medical_imaging', 'neuroscience', etc.
    has_cs = any(
        (isinstance(cat, str) and (cat.startswith('cs.') or 'stat.' in cat.lower() or 'cs.' in cat.lower())) 
        for cat in categories
    )
    has_bio = any(
        (isinstance(cat, str) and ('q-bio' in cat or 'bio' in cat.lower() or cat.startswith('q-bio')))
        for cat in categories
    )
    
    # Check for processed domain labels (from NeMo Curator)
    # These are the domain labels that NeMo Curator assigns
    healthcare_domain_labels = ['medical_imaging', 'neuroscience', 'clinical', 
                               'drug_discovery', 'neurodegeneration', 'general_ml_health']
    has_healthcare_domain = any(
        (isinstance(cat, str) and cat in healthcare_domain_labels)
        for cat in categories
    )
    
    # Also check if any category string contains healthcare-related terms
    healthcare_terms_in_cats = any(
        (isinstance(cat, str) and any(term in cat.lower() for term in ['medical', 'health', 'clinical', 'neuro', 'imaging', 'disease']))
        for cat in categories
    )
    
    # Check keywords in text (lower threshold if we have domain labels)
    ml_keywords = ['neural network', 'deep learning', 'machine learning', 
                   'convolutional', 'transformer', 'gradient', 'backpropagation',
                   'optimization', 'algorithm', 'model training', 'neural', 'cnn', 'rnn']
    healthcare_keywords = ['patient', 'clinical', 'medical', 'diagnosis',
                          'disease', 'treatment', 'brain', 'imaging', 'mri',
                          'alzheimer', 'parkinson', 'neurodegeneration', 'therapy',
                          'hospital', 'physician', 'symptom', 'pathology']
    
    # Count keyword matches
    ml_keyword_count = sum(1 for kw in ml_keywords if kw in text)
    healthcare_keyword_count = sum(1 for kw in healthcare_keywords if kw in text)
    
    # Lower threshold if we have some domain information
    ml_threshold = 1 if categories else 2
    healthcare_threshold = 1 if (categories or healthcare_terms_in_cats) else 2
    
    has_ml = has_cs or ml_keyword_count >= ml_threshold
    has_healthcare = has_bio or has_healthcare_domain or healthcare_terms_in_cats or healthcare_keyword_count >= healthcare_threshold
    
    if has_ml and has_healthcare:
        return 'Both'
    elif has_ml:
        return 'ML'
    elif has_healthcare:
        return 'Healthcare'
    else:
        return 'Other'


class ExpertActivationHook:
    """Lightweight hook to capture expert routing during evaluation.
    
    Works with SimpleMoEModel which has a single MoE layer.
    Captures routing decisions by intercepting model forward passes.
    """
    
    def __init__(self, model: nn.Module, text_dir: Optional[str] = None):
        """Initialize hook.
        
        Args:
            model: The model to hook into (should be SimpleMoEModel or wrapper)
            text_dir: Optional path to text files directory (for fallback classification)
        """
        self.expert_selections = []  # List of (batch_size, top_k) arrays
        self.expert_probs = []  # List of (batch_size, num_experts) arrays
        self.paper_ids = []  # List of paper IDs
        self.paper_domains = []  # List of domain labels
        self.model = model
        self.text_dir = text_dir  # Store text directory for fallback
        
        # Store original forward method
        if hasattr(model, 'base_model'):
            self.base_model = model.base_model
        else:
            self.base_model = model
    
    def capture_batch(
        self,
        gate_logits: torch.Tensor,
        routing_metrics: Optional[Dict],
        batch_metadata: Dict,
        top_k: int = 2
    ):
        """Capture routing information from a batch.
        
        Args:
            gate_logits: Router logits [batch*seq_len, num_experts] or [batch, seq_len, num_experts]
            routing_metrics: Dictionary with routing metrics (optional)
            batch_metadata: Batch metadata with paper IDs and domains
            top_k: Number of top experts selected
        """
        # Move to CPU and detach
        if isinstance(gate_logits, torch.Tensor):
            gate_logits_np = gate_logits.detach().cpu()
        else:
            gate_logits_np = np.array(gate_logits)
        
        # Handle different gate_logits shapes
        if len(gate_logits_np.shape) == 2:  # [batch*seq_len, num_experts]
            # Reshape: need to know batch_size and seq_len
            # For now, average over sequence dimension
            batch_size = len(batch_metadata.get('arxiv_ids', []))
            if batch_size > 0:
                seq_len = gate_logits_np.shape[0] // batch_size
                gate_logits_reshaped = gate_logits_np.reshape(batch_size, seq_len, -1)
                # Average over sequence: [batch_size, num_experts]
                gate_logits_avg = gate_logits_reshaped.mean(axis=1)
            else:
                gate_logits_avg = gate_logits_np.mean(axis=0, keepdims=True)
        elif len(gate_logits_np.shape) == 3:  # [batch, seq_len, num_experts]
            # Average over sequence: [batch, num_experts]
            gate_logits_avg = gate_logits_np.mean(axis=1)
        else:
            # Already [batch, num_experts] or similar
            gate_logits_avg = gate_logits_np
        
        # Get probabilities
        if isinstance(gate_logits_avg, torch.Tensor):
            probs = F.softmax(gate_logits_avg, dim=-1).numpy()
        else:
            probs = F.softmax(torch.tensor(gate_logits_avg), dim=-1).numpy()
        
        # Get top-k expert selections
        num_experts = probs.shape[-1]
        top_k = min(top_k, num_experts)
        
        # For each sample, get top-k experts
        top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:]  # [batch, top_k]
        
        # Create binary activation matrix (which experts activated)
        batch_size = probs.shape[0]
        activations = np.zeros((batch_size, num_experts), dtype=bool)
        for i in range(batch_size):
            activations[i, top_k_indices[i]] = True
        
        # Store
        self.expert_selections.append(top_k_indices)
        self.expert_probs.append(probs)
        
        # Store paper metadata
        arxiv_ids = batch_metadata.get('arxiv_ids', [])
        domains = batch_metadata.get('domains', [])
        
        for i in range(batch_size):
            if i < len(arxiv_ids):
                self.paper_ids.append(arxiv_ids[i])
            else:
                self.paper_ids.append(f'paper_{len(self.paper_ids)}')
            
            # Classify domain using the same logic as classify_paper_domain
            # Get domains (NeMo Curator labels) and categories (ArXiv categories) from batch metadata
            domain_list = domains[i] if (i < len(domains) and domains[i] is not None) else []
            if not isinstance(domain_list, list):
                # Handle case where domain_list might be a string, None, or other type
                if domain_list is None:
                    domain_list = []
                elif isinstance(domain_list, str):
                    domain_list = [domain_list] if domain_list.strip() else []
                else:
                    domain_list = [domain_list] if domain_list else []
            
            # Get original ArXiv categories (for ML detection)
            categories_list = []
            if 'categories' in batch_metadata:
                categories_list = batch_metadata['categories'][i] if (i < len(batch_metadata['categories']) and batch_metadata['categories'][i] is not None) else []
                if not isinstance(categories_list, list):
                    if categories_list is None:
                        categories_list = []
                    elif isinstance(categories_list, str):
                        categories_list = [categories_list] if categories_list.strip() else []
                    else:
                        categories_list = [categories_list] if categories_list else []
            
            # Try to get title/abstract from batch metadata if available
            title = ''
            abstract = ''
            if 'titles' in batch_metadata and i < len(batch_metadata.get('titles', [])):
                title = batch_metadata['titles'][i] or ''
            if 'abstracts' in batch_metadata and i < len(batch_metadata.get('abstracts', [])):
                abstract = batch_metadata['abstracts'][i] or ''
            
            # If domains are empty and we have text_dir, try to read text file for classification
            if not domain_list and not categories_list and self.text_dir and i < len(arxiv_ids):
                arxiv_id = arxiv_ids[i]
                text_file_path = os.path.join(self.text_dir, f"{arxiv_id}.txt")
                if os.path.exists(text_file_path):
                    try:
                        with open(text_file_path, 'r', encoding='utf-8') as f:
                            text_content = f.read()[:2000]  # Read first 2000 chars for classification
                            # Use text content for keyword-based classification
                            paper_dict = {
                                'categories': categories_list,  # Use ArXiv categories if available
                                'domains': domain_list,  # Use NeMo Curator domain labels
                                'title': title,
                                'abstract': abstract if abstract else text_content[:500]  # Use text as abstract fallback
                            }
                            domain = classify_paper_domain(paper_dict)
                        self.paper_domains.append(domain)
                        continue
                    except Exception:
                        pass  # Fall through to normal classification
            
            # Use classify_paper_domain for consistent classification
            # Combine both categories (ArXiv) and domains (NeMo Curator) for classification
            all_categories = categories_list + domain_list  # Combine both for classification
            paper_dict = {
                'categories': all_categories,  # Use combined list
                'domains': domain_list,  # Also set 'domains' field explicitly
                'title': title,
                'abstract': abstract
            }
            domain = classify_paper_domain(paper_dict)
            self.paper_domains.append(domain)
    
    def get_activations(self) -> Dict:
        """Get all stored activations in format expected by analysis notebook.
        
        Returns:
            Dictionary with:
            - expert_activations: (N_samples, N_experts) binary matrix
            - expert_probs: (N_samples, N_experts) probability matrix
            - paper_ids: List of paper IDs
            - paper_domains: List of domain labels
        """
        if not self.expert_probs:
            return {}
        
        # Concatenate all batches
        expert_probs_all = np.concatenate(self.expert_probs, axis=0)  # (N_samples, N_experts)
        
        # Create binary activation matrix from probabilities
        # Experts with probability > threshold are considered "activated"
        # Or use top-k per sample
        num_experts = expert_probs_all.shape[1]
        expert_activations = np.zeros_like(expert_probs_all, dtype=bool)
        
        # For each sample, mark top-k experts as activated
        # Use top_k from model if available, otherwise use 2
        top_k = getattr(self.base_model, 'top_k', 2) if hasattr(self.base_model, 'top_k') else 2
        top_k = min(top_k, num_experts)
        
        for i in range(expert_probs_all.shape[0]):
            top_indices = np.argsort(expert_probs_all[i])[-top_k:]
            expert_activations[i, top_indices] = True
        
        return {
            'expert_activations': expert_activations,
            'expert_probs': expert_probs_all,
            'paper_ids': np.array(self.paper_ids),
            'paper_domains': np.array(self.paper_domains)
        }
    
    def clear(self):
        """Clear stored activations."""
        self.expert_selections = []
        self.expert_probs = []
        self.paper_ids = []
        self.paper_domains = []


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
    max_samples: Optional[int] = None,
    activation_hook: Optional[ExpertActivationHook] = None
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
    dataloader,
    activation_hook: Optional[ExpertActivationHook] = None
) -> Tuple[float, Dict]:
    """Compute perplexity on test set with domain-specific metrics.
    
    Args:
        model: Trained model
        adapter: Model adapter
        dataloader: DataLoader for test data
        activation_hook: Optional hook to capture expert activations
        
    Returns:
        Tuple of (overall_perplexity, domain_metrics_dict)
        - overall_perplexity: Overall perplexity score
        - domain_metrics_dict: Dictionary with per-domain metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Track per-domain metrics
    domain_losses = defaultdict(float)
    domain_tokens = defaultdict(int)
    domain_paper_counts = defaultdict(int)
    
    # Get base model for routing capture
    base_model = model.base_model if hasattr(model, 'base_model') else model
    top_k = getattr(base_model, 'top_k', 2) if hasattr(base_model, 'top_k') else 2
    
    with torch.no_grad():
        for batch in dataloader:
            with torch.cuda.amp.autocast():
                # Forward pass
                if activation_hook is not None:
                    # Single forward pass: get both loss and routing info
                    input_ids = batch['input_ids'].to(adapter.device)
                    target_ids = batch['target_ids'].to(adapter.device)
                    
                    # Call base model with routing info
                    output, routing_info = base_model(
                        input_ids,
                        image_features=None,
                        return_load_balance_loss=True,
                        return_gate_logits=True
                    )
                    
                    # Compute loss manually (to avoid double forward pass)
                    logits = output  # [batch, seq_len, vocab_size]
                    # Reshape for cross-entropy
                    logits_flat = logits.view(-1, logits.shape[-1])
                    targets_flat = target_ids.view(-1)
                    loss = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        ignore_index=adapter.ignore_index,
                        reduction='mean'
                    )
                    
                    # Compute per-sample losses for domain tracking
                    # Get attention mask (non-padding tokens)
                    attention_mask = (target_ids != adapter.ignore_index).float()
                    
                    # Per-sample loss
                    loss_per_token = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        ignore_index=adapter.ignore_index,
                        reduction='none'
                    ).view(target_ids.shape[0], -1)  # [batch, seq_len]
                    
                    # Mask out padding tokens
                    loss_per_sample = (loss_per_token * attention_mask).sum(dim=1)  # [batch]
                    tokens_per_sample = attention_mask.sum(dim=1)  # [batch]
                    
                    # Extract routing information
                    if routing_info is not None:
                        gate_logits, _, _, _, _, routing_metrics = routing_info
                        # Get batch metadata
                        device_batch = adapter._move_to_device(batch)
                        batch_metadata = {
                            'arxiv_ids': device_batch.get('arxiv_ids', []),
                            'domains': device_batch.get('domains', []),
                            'years': device_batch.get('years', []),
                            'titles': batch.get('titles', []),  # May not be in batch
                            'abstracts': batch.get('abstracts', [])  # May not be in batch
                        }
                        
                        # Classify domains and track per-domain metrics
                        arxiv_ids = batch_metadata.get('arxiv_ids', [])
                        domains_list = batch_metadata.get('domains', [])
                        titles = batch_metadata.get('titles', [])
                        abstracts = batch_metadata.get('abstracts', [])
                        
                        # Determine batch size from input_ids
                        batch_size = input_ids.shape[0] if input_ids is not None else len(arxiv_ids)
                        
                        for i in range(batch_size):
                            # Classify domain (use categories primarily, titles/abstracts if available)
                            paper_dict = {
                                'categories': domains_list[i] if i < len(domains_list) else [],
                                'title': titles[i] if i < len(titles) else '',
                                'abstract': abstracts[i] if i < len(abstracts) else ''
                            }
                            domain = classify_paper_domain(paper_dict)
                            
                            # Track domain metrics
                            if i < len(loss_per_sample):
                                domain_losses[domain] += loss_per_sample[i].item()
                                domain_tokens[domain] += tokens_per_sample[i].item()
                                domain_paper_counts[domain] += 1
                        
                        activation_hook.capture_batch(
                            gate_logits=gate_logits,
                            routing_metrics=routing_metrics,
                            batch_metadata=batch_metadata,
                            top_k=top_k
                        )
                else:
                    result = adapter.process_batch(batch)
                    loss = result['loss']
                    batch_metadata = result['batch_metadata']
                    
                    # Still track domains even without hook
                    device_batch = adapter._move_to_device(batch)
                    domains_list = device_batch.get('domains', [])
                    titles = batch.get('titles', [])  # May not be in batch
                    abstracts = batch.get('abstracts', [])  # May not be in batch
                    arxiv_ids = device_batch.get('arxiv_ids', [])
                    
                    # Compute per-sample loss for domain tracking
                    logits = result['logits']
                    target_ids_batch = batch['target_ids'].to(adapter.device)
                    attention_mask = (target_ids_batch != adapter.ignore_index).float()
                    
                    loss_per_token = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        target_ids_batch.view(-1),
                        ignore_index=adapter.ignore_index,
                        reduction='none'
                    ).view(target_ids_batch.shape[0], -1)
                    
                    loss_per_sample = (loss_per_token * attention_mask).sum(dim=1)
                    tokens_per_sample = attention_mask.sum(dim=1)
                    
                    batch_size = target_ids_batch.shape[0]
                    for i in range(batch_size):
                        paper_dict = {
                            'categories': domains_list[i] if i < len(domains_list) else [],
                            'title': titles[i] if i < len(titles) else '',
                            'abstract': abstracts[i] if i < len(abstracts) else ''
                        }
                        domain = classify_paper_domain(paper_dict)
                        
                        if i < len(loss_per_sample):
                            domain_losses[domain] += loss_per_sample[i].item()
                            domain_tokens[domain] += tokens_per_sample[i].item()
                            domain_paper_counts[domain] += 1
            
            # Get number of non-padding tokens
            target_ids = batch['target_ids'].to(adapter.device)
            num_tokens = (target_ids != 0).sum().item()
            
            # Accumulate loss (weighted by tokens)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf'), {}
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    # Compute per-domain metrics
    domain_metrics = {}
    for domain in domain_losses:
        if domain_tokens[domain] > 0:
            domain_avg_loss = domain_losses[domain] / domain_tokens[domain]
            domain_metrics[domain] = {
                'loss': float(domain_avg_loss),
                'perplexity': float(np.exp(domain_avg_loss)),
                'num_papers': int(domain_paper_counts[domain]),
                'num_tokens': int(domain_tokens[domain])
            }
    
    return perplexity, domain_metrics


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
        print(f"Domain classification failed: {e}")
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


def get_drive_results_path(local_path: str = "./evaluations") -> str:
    """Get path for saving results, preferring Google Drive if available.
    
    Args:
        local_path: Local fallback path
        
    Returns:
        Path string (Drive path if available, otherwise local)
    """
    # Check for Google Drive
    drive_base = os.environ.get('DRIVE_BASE', '/content/drive/MyDrive/neuroMOE_results')
    
    # Check if Drive is mounted
    if os.path.exists(drive_base) and os.access(drive_base, os.W_OK):
        # Use Drive results folder
        drive_results = os.path.join(drive_base, 'evaluations')
        return drive_results
    
    # Also check if we're in Colab and Drive might be mounted at /content/drive
    if os.path.exists('/content/drive/MyDrive'):
        drive_base = '/content/drive/MyDrive/neuroMOE_results'
        if os.path.exists(drive_base) and os.access(drive_base, os.W_OK):
            drive_results = os.path.join(drive_base, 'evaluations')
            return drive_results
    
    # Fall back to local path
    return local_path


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
    print("Model Evaluation")
    print("=" * 60)
    print()
    
    # Load tokenizer
    if not SENTENCEPIECE_AVAILABLE:
        raise ImportError("sentencepiece package required")
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    print(f"Loaded tokenizer (vocab_size={vocab_size})")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
        print(f"Loaded model from {model_checkpoint}")
    except Exception as e:
        print(f"Could not load model: {e}")
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
    
    # Split into test set with stratified sampling to preserve domain distribution
    all_files = full_dataset.text_files
    metadata = full_dataset.metadata
    
    # Classify all papers to get domain distribution
    print("Classifying papers for stratified split...")
    file_domains = []
    for arxiv_id, _ in all_files:
        meta = metadata.get(arxiv_id, {})
        domains = meta.get('domains', [])
        categories = meta.get('categories', [])  # Get ArXiv categories
        title = meta.get('title', '')
        abstract = meta.get('abstract', '')
        
        # Combine categories and domains for classification
        all_cats = []
        if categories:
            if isinstance(categories, list):
                all_cats.extend(categories)
            else:
                all_cats.append(categories)
        if domains:
            if isinstance(domains, list):
                all_cats.extend(domains)
            else:
                all_cats.append(domains)
        
        paper_dict = {
            'categories': all_cats,  # Use combined list
            'domains': domains if isinstance(domains, list) else [domains] if domains else [],
            'title': title,
            'abstract': abstract
        }
        domain = classify_paper_domain(paper_dict)
        file_domains.append((arxiv_id, domain))
    
    # Group files by domain
    from collections import defaultdict
    domain_groups = defaultdict(list)
    for arxiv_id, domain in file_domains:
        domain_groups[domain].append(arxiv_id)
    
    # Print domain distribution
    print(f"\nFull dataset domain distribution:")
    for domain, files in sorted(domain_groups.items()):
        print(f"  {domain}: {len(files)} papers ({len(files)/len(all_files)*100:.1f}%)")
    
    # Stratified sampling: sample proportionally from each domain
    import random
    random.seed(42)  # For reproducibility
    
    test_files = []
    train_files = []
    
    for domain, files in domain_groups.items():
        n_domain_test = max(1, int(len(files) * test_split))  # At least 1 per domain
        random.shuffle(files)
        domain_test = files[:n_domain_test]
        domain_train = files[n_domain_test:]
        
        # Convert back to (arxiv_id, file_path) tuples
        for arxiv_id in domain_test:
            file_path = os.path.join(dataset_text_dir, f"{arxiv_id}.txt")
            if os.path.exists(file_path):
                test_files.append((arxiv_id, file_path))
        
        for arxiv_id in domain_train:
            file_path = os.path.join(dataset_text_dir, f"{arxiv_id}.txt")
            if os.path.exists(file_path):
                train_files.append((arxiv_id, file_path))
    
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
        categories = meta.get('categories', [])  # Get ArXiv categories
        title = meta.get('title', '')
        abstract = meta.get('abstract', '')
        
        # Combine categories and domains for classification
        all_cats = []
        if categories:
            if isinstance(categories, list):
                all_cats.extend(categories)
            else:
                all_cats.append(categories)
        if domains:
            if isinstance(domains, list):
                all_cats.extend(domains)
            else:
                all_cats.append(domains)
        
        paper_dict = {
            'categories': all_cats,  # Use combined list
            'domains': domains if isinstance(domains, list) else [domains] if domains else [],
            'title': title,
            'abstract': abstract
        }
        domain = classify_paper_domain(paper_dict)
        test_domains[domain] += 1
    
    print(f"\nTest set domain distribution:")
    for domain, count in sorted(test_domains.items()):
        print(f"  {domain}: {count} papers ({count/len(test_files)*100:.1f}%)")
    
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
    
    print(f"Created test dataset: {len(test_files)} papers")
    
    # Create test dataloader
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize activation hook for capturing expert routing
    print("\nInitializing expert activation capture...")
    activation_hook = ExpertActivationHook(model, text_dir=dataset_text_dir)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # 1. Perplexity (with activation capture and domain metrics)
    print("   Computing perplexity...")
    perplexity, domain_metrics = compute_perplexity(model, adapter, test_dataloader, activation_hook=activation_hook)
    print(f"   Perplexity: {perplexity:.2f}")
    
    # Print domain-specific results
    if domain_metrics:
        print("\n   Domain-Specific Perplexity:")
        for domain in sorted(domain_metrics.keys()):
            metrics = domain_metrics[domain]
            print(f"     {domain}: {metrics['perplexity']:.2f} ({metrics['num_papers']} papers)")
    
    # 2. Extract embeddings (activations already captured during perplexity, no need to capture again)
    print("   Extracting embeddings...")
    embeddings, metadata = extract_embeddings(
        model, adapter, test_dataloader, max_samples=max_test_samples, activation_hook=None
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
        model, adapter, test_dataloader, num_samples=min(100, len(test_files))
    )
    print(f"   Section accuracy: {section_accuracy:.4f}")
    
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
        },
        'domain_metrics': domain_metrics  # NEW: Per-domain metrics
    }
    
    # Determine results directory (prefer Drive if available)
    results_dir = get_drive_results_path(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📁 Saving results to: {results_dir}")
    
    # Save timestamped file (for tracking multiple evaluations)
    timestamped_file = os.path.join(results_dir, f"evaluation_{int(time.time())}.json")
    with open(timestamped_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Timestamped results saved to: {timestamped_file}")
    
    # Also save standard eval_results.json (for easy access by analysis notebook)
    standard_file = os.path.join(results_dir, "eval_results.json")
    with open(standard_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Standard results saved to: {standard_file}")
    
    # Update results dict with actual paths
    results['eval_results_path'] = standard_file
    
    # Print domain breakdown summary
    if domain_metrics:
        print("\n" + "="*60)
        print("DOMAIN-SPECIFIC RESULTS")
        print("="*60)
        for domain in sorted(domain_metrics.keys()):
            metrics = domain_metrics[domain]
            print(f"\n{domain}:")
            print(f"  Papers: {metrics['num_papers']}")
            print(f"  Tokens: {metrics['num_tokens']:,}")
            print(f"  Perplexity: {metrics['perplexity']:.2f}")
            print(f"  Loss: {metrics['loss']:.4f}")
        print("="*60)
    
    # Debug: Print domain distribution from activation hook
    if activation_hook and len(activation_hook.paper_domains) > 0:
        from collections import Counter
        domain_dist = Counter(activation_hook.paper_domains)
        print("\n" + "="*60)
        print("DOMAIN CLASSIFICATION DISTRIBUTION (from activation hook)")
        print("="*60)
        for domain, count in domain_dist.most_common():
            print(f"  {domain}: {count} papers ({count/len(activation_hook.paper_domains)*100:.1f}%)")
        print("="*60)
        
        # Debug: Show sample of domains from metadata vs classification
        if len(activation_hook.paper_ids) > 0:
            print(f"\nDebug: Sample domain classifications (first 5 papers):")
            sample_count = 0
            for i in range(min(5, len(activation_hook.paper_ids))):
                arxiv_id = activation_hook.paper_ids[i]
                # Try to get domains from metadata
                metadata_entry = test_dataset.metadata.get(arxiv_id, {})
                metadata_domains = metadata_entry.get('domains', [])
                classified_domain = activation_hook.paper_domains[i]
                print(f"  {arxiv_id}:")
                print(f"    Metadata domains: {metadata_domains} (type: {type(metadata_domains)})")
                print(f"    Classified as: {classified_domain}")
                sample_count += 1
                if sample_count >= 5:
                    break
    
    # Save expert activations if captured
    if activation_hook is not None:
        activations = activation_hook.get_activations()
        if activations:
            # Determine output path - save to Drive results folder if available
            drive_base = os.environ.get('DRIVE_BASE', '/content/drive/MyDrive/neuroMOE_results')
            
            # Check if Drive is available
            if os.path.exists(drive_base) and os.access(drive_base, os.W_OK):
                # Save to Drive results folder
                activations_dir = os.path.join(drive_base, 'evaluations')
            elif os.path.exists('/content/drive/MyDrive'):
                # Alternative Drive path
                drive_base = '/content/drive/MyDrive/neuroMOE_results'
                if os.path.exists(drive_base) and os.access(drive_base, os.W_OK):
                    activations_dir = os.path.join(drive_base, 'evaluations')
                else:
                    # Fall back to local results directory
                    activations_dir = results_dir
            else:
                # Fall back to local results directory
                activations_dir = results_dir
            
            os.makedirs(activations_dir, exist_ok=True)
            activations_path = os.path.join(activations_dir, 'expert_activations.npz')
            
            save_expert_activations(activations, activations_path)
            print(f"\n✅ Expert activations saved to: {activations_path}")
            results['expert_activations_path'] = activations_path
    
    return results


def save_expert_activations(activations: Dict, output_path: str):
    """Save expert activations to compressed numpy file.
    
    Args:
        activations: Dictionary with activation data from ExpertActivationHook
        output_path: Path to save .npz file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save in format expected by analysis notebook
    np.savez_compressed(
        output_path,
        expert_activations=activations['expert_activations'],
        expert_probs=activations['expert_probs'],
        paper_ids=activations['paper_ids'],
        paper_domains=activations['paper_domains']
    )
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Samples: {len(activations['paper_ids'])}")
    print(f"   Experts: {activations['expert_activations'].shape[1]}")


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
        print("matplotlib not available, skipping visualization")
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
    print(f"Training curves saved to: {output_file}")
    
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
        print("matplotlib not available, skipping visualization")
        return
    
    # Load all evaluation files
    eval_files = sorted([
        os.path.join(evaluation_dir, f)
        for f in os.listdir(evaluation_dir)
        if f.startswith('evaluation_') and f.endswith('.json')
    ])
    
    if len(eval_files) == 0:
        print("No evaluation files found")
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
    print(f"Evaluation trends saved to: {output_file}")
    
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
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

