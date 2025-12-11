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
    from tokenizer_wrapper import TokenizerWrapper, load_medical_tokenizer, DEFAULT_MEDICAL_TOKENIZER
    TOKENIZER_WRAPPER_AVAILABLE = True
except ImportError:
    TOKENIZER_WRAPPER_AVAILABLE = False
    print("tokenizer_wrapper not available, falling back to SentencePiece only")

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
    # Also check full text if title/abstract are missing or very short
    full_text = paper.get('text', '').lower() if paper.get('text') else ''
    # Use full text if title+abstract is too short (< 50 chars) or missing
    if len(title + ' ' + abstract) < 50 and full_text:
        text = full_text[:2000]  # Use first 2000 chars of full text for keyword matching
    else:
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
    
    # Special case: If we have healthcare domains but no CS categories detected,
    # but we have ML keywords in text, assume it's "Both" (papers from queries like "cat:cs.LG AND healthcare")
    # This handles cases where categories weren't loaded from fallback
    if has_healthcare_domain and not has_cs and ml_keyword_count >= 1:
        has_ml = True
    
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
    
    def __init__(self, model: nn.Module, text_dir: Optional[str] = None, metadata_dict: Optional[Dict] = None):
        """Initialize hook.
        
        Args:
            model: The model to hook into (should be SimpleMoEModel or wrapper)
            text_dir: Optional path to text files directory (for fallback classification)
            metadata_dict: Optional metadata dictionary for fallback category lookup
        """
        self.expert_selections = []  # List of (batch_size, top_k) arrays
        self.expert_probs = []  # List of (batch_size, num_experts) arrays
        self.expert_activations_list = []  # List of (batch_size, num_experts) activation matrices
        self.paper_ids = []  # List of paper IDs
        self.paper_domains = []  # List of domain labels
        self.model = model
        self.text_dir = text_dir  # Store text directory for fallback
        self._metadata_dict = metadata_dict  # Store metadata dict for fallback category lookup
        
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
        # For Expert Choice routing, we need to track which tokens each expert selected
        # Don't average - keep per-token routing decisions
        batch_size = len(batch_metadata.get('arxiv_ids', []))
        
        if len(gate_logits_np.shape) == 2:  # [batch*seq_len, num_experts]
            if batch_size > 0:
                total_tokens = gate_logits_np.shape[0]
                seq_len = total_tokens // batch_size
                # Handle case where division isn't exact (shouldn't happen, but be safe)
                if total_tokens % batch_size != 0:
                    print(f"WARNING: total_tokens ({total_tokens}) not divisible by batch_size ({batch_size})")
                    seq_len = (total_tokens + batch_size - 1) // batch_size  # Ceiling division
                gate_logits_reshaped = gate_logits_np.reshape(batch_size, seq_len, -1)
                num_experts = gate_logits_reshaped.shape[-1]
                
                # For Expert Choice: each expert selects top_k tokens
                # Compute per-token probabilities, then determine which experts selected which tokens
                # Reshape to [batch*seq_len, num_experts] for processing
                gate_logits_flat = gate_logits_reshaped.reshape(-1, num_experts)
                
                # For Expert Choice routing: transpose to [num_experts, batch*seq_len]
                # Each expert sees scores for all tokens
                if isinstance(gate_logits_flat, torch.Tensor):
                    expert_logits = gate_logits_flat.t()  # [num_experts, batch*seq_len]
                else:
                    expert_logits = torch.tensor(gate_logits_flat, dtype=torch.float32).t()  # [num_experts, batch*seq_len]
                
                # For each expert, compute softmax over all tokens to get selection probabilities
                # This gives each expert a probability distribution over all tokens
                # Ensure we use torch tensor for softmax
                if not isinstance(expert_logits, torch.Tensor):
                    expert_logits = torch.tensor(expert_logits, dtype=torch.float32)
                expert_probs_all = F.softmax(expert_logits, dim=-1).detach().cpu().numpy()  # [num_experts, batch*seq_len]
                
                # Each expert selects top_k tokens with highest probabilities
                top_k = min(top_k, num_experts)
                expert_token_selections = np.argsort(expert_probs_all, axis=-1)[:, -top_k:]  # [num_experts, top_k]
                
                # Create activation matrix: [batch_size, num_experts]
                # Track which experts processed tokens from each paper
                activations = np.zeros((batch_size, num_experts), dtype=bool)
                probs = np.zeros((batch_size, num_experts), dtype=float)
                token_counts = np.zeros((batch_size, num_experts), dtype=int)  # Track token counts per expert per paper
                
                for expert_idx in range(num_experts):
                    # Get tokens selected by this expert
                    selected_tokens = expert_token_selections[expert_idx]  # [top_k]
                    
                    # Map tokens back to papers
                    for token_idx in selected_tokens:
                        # Ensure token_idx is within bounds
                        if token_idx >= expert_probs_all.shape[1]:
                            continue
                        # Map flattened token index back to (paper_idx, token_pos_in_paper)
                        # token_idx is in range [0, batch_size * seq_len)
                        paper_idx = token_idx // seq_len
                        token_pos = token_idx % seq_len
                        
                        if paper_idx < batch_size:
                            activations[paper_idx, expert_idx] = True
                            # Use the probability that this expert selected this token
                            # expert_probs_all[expert_idx, token_idx] is the probability expert_idx selected token_idx
                            prob_value = float(expert_probs_all[expert_idx, token_idx])
                            if not np.isnan(prob_value) and prob_value > 0:
                                probs[paper_idx, expert_idx] += prob_value
                                token_counts[paper_idx, expert_idx] += 1
                        else:
                            if len(self.expert_probs) == 0:
                                print(f"  WARNING: token_idx={token_idx} maps to paper_idx={paper_idx} >= batch_size={batch_size}")
                
                # Normalize probabilities (average over tokens that activated each expert)
                for paper_idx in range(batch_size):
                    for expert_idx in range(num_experts):
                        if token_counts[paper_idx, expert_idx] > 0:
                            probs[paper_idx, expert_idx] /= token_counts[paper_idx, expert_idx]
                        else:
                            # If expert didn't select any tokens from this paper, probability is 0
                            probs[paper_idx, expert_idx] = 0.0
                
                # Find papers that actually have tokens
                    papers_with_tokens = np.where(token_counts.sum(axis=1) > 0)[0]
                    if len(papers_with_tokens) > 0:
                        paper_idx = papers_with_tokens[0]
                        print(f"  Papers with tokens: {papers_with_tokens[:5]}...")
                        print(f"  probs for paper {paper_idx}: {probs[paper_idx]}")
                        print(f"  token_counts for paper {paper_idx}: {token_counts[paper_idx]}")
                    else:
                        print(f"  WARNING: No papers have tokens mapped!")
                    print(f"  expert_probs_all shape: {expert_probs_all.shape}")
                    print(f"  expert_probs_all sample (expert 0, first 5 tokens): {expert_probs_all[0, :5] if expert_probs_all.shape[1] >= 5 else expert_probs_all[0]}")
                    
                    # Show sample token selections
                    print(f"  Sample token selections:")
                    for expert_idx in range(min(4, num_experts)):
                        selected = expert_token_selections[expert_idx]
                        print(f"    Expert {expert_idx}: selected tokens {selected[:5]}..." if len(selected) > 5 else f"    Expert {expert_idx}: selected tokens {selected}")
                    
                    # Check if Expert 2 has lower probabilities in expert_probs_all
                    if expert_probs_all.shape[0] >= 3:
                        expert_2_probs = expert_probs_all[2, :]  # All tokens for Expert 2
                        other_experts_probs = np.concatenate([expert_probs_all[:2, :], expert_probs_all[3:, :]], axis=0)
                        print(f"  Expert 2 avg prob across all tokens: {np.mean(expert_2_probs):.6f}")
                        print(f"  Other experts avg prob: {np.mean(other_experts_probs):.6f}")
                        print(f"  Expert 2 max prob: {np.max(expert_2_probs):.6f}")
                        print(f"  Other experts max prob: {np.max(other_experts_probs):.6f}")
                        
                        # Check how many tokens Expert 2 selected
                        expert_2_selected = len(expert_token_selections[2])
                        other_selected = [len(expert_token_selections[i]) for i in range(len(expert_token_selections)) if i != 2]
                        print(f"  Expert 2 selected {expert_2_selected} tokens")
                        print(f"  Other experts selected: {other_selected}")
            else:
                # Fallback: average over sequence (when batch_size == 0)
                gate_logits_avg = gate_logits_np.mean(axis=0, keepdims=True)
                num_experts = gate_logits_avg.shape[-1]
                if isinstance(gate_logits_avg, torch.Tensor):
                    probs = F.softmax(gate_logits_avg, dim=-1).numpy()
                else:
                    probs = F.softmax(torch.tensor(gate_logits_avg), dim=-1).numpy()
                top_k = min(top_k, num_experts)
                top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:]
                batch_size = 1  # Set to 1 for the fallback case
                activations = np.zeros((batch_size, num_experts), dtype=bool)
                activations[0, top_k_indices[0]] = True
        elif len(gate_logits_np.shape) == 3:  # [batch, seq_len, num_experts]
            # Similar processing for 3D case
            batch_size, seq_len, num_experts = gate_logits_np.shape
            gate_logits_flat = gate_logits_np.reshape(-1, num_experts)
            
            # For Expert Choice routing: transpose to [num_experts, batch*seq_len]
            if isinstance(gate_logits_flat, torch.Tensor):
                expert_logits = gate_logits_flat.t()  # [num_experts, batch*seq_len]
            else:
                expert_logits = torch.tensor(gate_logits_flat, dtype=torch.float32).t()  # [num_experts, batch*seq_len]
            
            # For each expert, compute softmax over all tokens
            # Ensure we use torch tensor for softmax
            if not isinstance(expert_logits, torch.Tensor):
                expert_logits = torch.tensor(expert_logits, dtype=torch.float32)
            expert_probs_all = F.softmax(expert_logits, dim=-1).detach().cpu().numpy()  # [num_experts, batch*seq_len]
            
            top_k = min(top_k, num_experts)
            expert_token_selections = np.argsort(expert_probs_all, axis=-1)[:, -top_k:]  # [num_experts, top_k]
            
            activations = np.zeros((batch_size, num_experts), dtype=bool)
            probs = np.zeros((batch_size, num_experts), dtype=float)
            token_counts = np.zeros((batch_size, num_experts), dtype=int)
            
            for expert_idx in range(num_experts):
                selected_tokens = expert_token_selections[expert_idx]
                for token_idx in selected_tokens:
                    # Ensure token_idx is within bounds
                    if token_idx >= expert_probs_all.shape[1]:
                        continue
                    paper_idx = token_idx // seq_len
                    if paper_idx < batch_size:
                        activations[paper_idx, expert_idx] = True
                        # Use the probability that this expert selected this token
                        prob_value = expert_probs_all[expert_idx, token_idx]
                        if not np.isnan(prob_value) and prob_value > 0:
                            probs[paper_idx, expert_idx] += prob_value
                            token_counts[paper_idx, expert_idx] += 1
            
            for paper_idx in range(batch_size):
                for expert_idx in range(num_experts):
                    if token_counts[paper_idx, expert_idx] > 0:
                        probs[paper_idx, expert_idx] /= token_counts[paper_idx, expert_idx]
                    else:
                        probs[paper_idx, expert_idx] = 0.0
        else:
            # Already [batch, num_experts] or similar - use old logic
            gate_logits_avg = gate_logits_np
            num_experts = gate_logits_avg.shape[-1]
            if isinstance(gate_logits_avg, torch.Tensor):
                probs = F.softmax(gate_logits_avg, dim=-1).numpy()
            else:
                probs = F.softmax(torch.tensor(gate_logits_avg), dim=-1).numpy()
            top_k = min(top_k, num_experts)
            top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:]
            batch_size = probs.shape[0]
            activations = np.zeros((batch_size, num_experts), dtype=bool)
            for i in range(batch_size):
                activations[i, top_k_indices[i]] = True
        
        # Store
        # expert_selections: which experts were selected (for compatibility)
        # For Expert Choice, we track which experts activated per paper
        expert_selections = []
        for paper_idx in range(batch_size):
            activated_experts = np.where(activations[paper_idx])[0]
            expert_selections.append(activated_experts)
        self.expert_selections.append(expert_selections)
        self.expert_probs.append(probs)
        self.expert_activations_list.append(activations)  # Store the actual activation matrix
        
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
                if i < len(batch_metadata['categories']):
                    categories_list = batch_metadata['categories'][i]
                    if categories_list is None:
                        categories_list = []
                    elif not isinstance(categories_list, list):
                        if isinstance(categories_list, str):
                            categories_list = [categories_list] if categories_list.strip() else []
                        else:
                            categories_list = [categories_list] if categories_list else []
                else:
                    categories_list = []
            
            # Fallback: If categories are empty, try to look them up from metadata file
            # This handles cases where categories weren't passed through the batch
            if not categories_list and i < len(arxiv_ids):
                arxiv_id = arxiv_ids[i]
                # Try to get categories from the dataset's metadata if available
                # We'll need to store a reference to the dataset or metadata dict
                if hasattr(self, '_metadata_dict') and self._metadata_dict:
                    paper_meta = self._metadata_dict.get(arxiv_id, {})
                    categories_list = paper_meta.get('categories', [])
                    if categories_list and not isinstance(categories_list, list):
                        categories_list = [categories_list] if categories_list else []
            
            if not categories_list:
                # Try fallback lookup if categories are missing
                pass
            
            # Try to get title/abstract from batch metadata if available
            title = ''
            abstract = ''
            if 'titles' in batch_metadata and i < len(batch_metadata.get('titles', [])):
                title = batch_metadata['titles'][i] or ''
            if 'abstracts' in batch_metadata and i < len(batch_metadata.get('abstracts', [])):
                abstract = batch_metadata['abstracts'][i] or ''
            
            # Fallback: If title/abstract are empty, try to get them from metadata dict
            if (not title or not abstract) and i < len(arxiv_ids):
                arxiv_id = arxiv_ids[i]
                if hasattr(self, '_metadata_dict') and self._metadata_dict:
                    paper_meta = self._metadata_dict.get(arxiv_id, {})
                    if not title:
                        title = paper_meta.get('title', '')
                    if not abstract:
                        abstract = paper_meta.get('abstract', '')
            
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
            if len(self.paper_domains) < 5:
                print(f"    Classified as: {domain}")
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
        
        # Use the actual activation matrices computed in capture_batch()
        # These correctly reflect which experts processed tokens from which papers
        if self.expert_activations_list:
            expert_activations = np.concatenate(self.expert_activations_list, axis=0)  # (N_samples, N_experts)
        else:
            # Fallback: create from probabilities (shouldn't happen if capture_batch worked)
            expert_activations = expert_probs_all > 0
        
        num_experts = expert_probs_all.shape[1]
        
        # Debug: Check if probabilities are identical across experts
        # Find a sample that actually has non-zero probabilities
        if expert_probs_all.shape[0] > 0:
            # Find first sample with non-zero probabilities
            sample_idx = None
            for i in range(expert_probs_all.shape[0]):
                sample_probs = expert_probs_all[i]
                if np.any(sample_probs > 0):
                    sample_idx = i
                    break
            
            if sample_idx is not None:
                sample_probs = expert_probs_all[sample_idx]
                if len(sample_probs) > 1:
                    probs_std = np.std(sample_probs)
                    probs_mean = np.mean(sample_probs)
                    probs_min = np.min(sample_probs)
                    probs_max = np.max(sample_probs)
                    if probs_std < 1e-6:
                        print(f"WARNING: Expert probabilities are nearly identical (std={probs_std:.2e}). ")
                        print(f"  Sample {sample_idx} probabilities: {sample_probs}")
                        print(f"  Mean: {probs_mean:.4f}, Min: {probs_min:.4f}, Max: {probs_max:.4f}")
                        print(f"  This suggests the model's gate weights are uniform or haven't learned differentiation.")
                    else:
                        print(f"Expert probability diversity (sample {sample_idx}): std={probs_std:.4f} (good if > 0.01)")
                        print(f"  Sample probabilities: {sample_probs}")
                        print(f"  Mean: {probs_mean:.4f}, Min: {probs_min:.4f}, Max: {probs_max:.4f}")
            else:
                # All samples have zero probabilities - this is the real bug
                print(f"WARNING: All samples have zero probabilities!")
                print(f"  This indicates a bug in probability computation/storage.")
                print(f"  Checking first few samples: {expert_probs_all[:min(5, expert_probs_all.shape[0])]}")
        
        # Debug: Check activation diversity
        activation_counts = expert_activations.sum(axis=0)  # Count activations per expert
        if len(activation_counts) > 1:
            activation_std = np.std(activation_counts)
            activation_mean = np.mean(activation_counts)
            print(f"Expert activation statistics: mean={activation_mean:.1f}, std={activation_std:.1f}, "
                  f"min={activation_counts.min()}, max={activation_counts.max()}")
            if activation_std < 1.0:
                print(f"WARNING: Experts have very similar activation counts. This suggests they're routing identically.")
            else:
                print(f"✓ Experts show different activation patterns (std={activation_std:.1f})")
        
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
        self.expert_activations_list = []
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
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
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

    # Debug: Check dataloader
    print(f"   🔍 Debugging dataloader...")
    print(f"      Dataloader length: {len(dataloader)}")
    print(f"      Batch size: {dataloader.batch_size}")
    print(f"      Dataset length: {len(dataloader.dataset)}")

    # Debug: Check dataset items
    if hasattr(dataloader.dataset, 'papers'):
        print(f"      Dataset papers: {len(dataloader.dataset.papers)}")
        if dataloader.dataset.papers:
            first_paper_id, first_paper_path = dataloader.dataset.papers[0]
            print(f"      First paper: {first_paper_id} -> {first_paper_path}")

    # Debug: Try to process one batch manually
    print(f"   🔍 Testing first batch...")
    batch_count = 0
    try:
        for i, batch in enumerate(dataloader):
            batch_count += 1
            print(f"      Batch {batch_count}: {type(batch)}")
            if isinstance(batch, dict):
                print(f"      Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"         {key}: {value.shape} (dtype: {value.dtype})")
                        # Check for invalid tensors
                        if value.numel() == 0:
                            print(f"         ❌ EMPTY TENSOR for {key}")
                        if torch.isnan(value).any():
                            print(f"         ❌ NAN VALUES in {key}")
                    else:
                        print(f"         {key}: {type(value)}")

                # Check if input_ids are reasonable
                if 'input_ids' in batch:
                    input_ids = batch['input_ids']
                    print(f"      Input ID range: {input_ids.min().item()} to {input_ids.max().item()}")
                    if input_ids.max() >= adapter.vocab_size:
                        print(f"      ❌ VOCAB SIZE MISMATCH: max token {input_ids.max().item()} >= vocab_size {adapter.vocab_size}")

            else:
                print(f"      Batch type: {type(batch)}")

            if batch_count >= 1:  # Only check first batch
                break
    except Exception as e:
        print(f"   ❌ Error processing batch: {e}")
        import traceback
        print(f"      Traceback: {traceback.format_exc()}")

    print(f"   🔍 Actual batches available: {batch_count}")

    if batch_count == 0:
        print(f"   ❌ DATALOADER ISSUE: No batches yielded!")
        print(f"   🔧 Possible causes:")
        print(f"      1. Dataset is empty")
        print(f"      2. Tokenizer is failing to tokenize text")
        print(f"      3. All sequences are being filtered out")
        print(f"      4. Memory issues preventing batch creation")
        return float('inf'), {}
    
    # Track per-domain metrics
    domain_losses = defaultdict(float)
    domain_tokens = defaultdict(int)
    domain_paper_counts = defaultdict(int)
    
    # Get base model for routing capture
    base_model = model.base_model if hasattr(model, 'base_model') else model
    top_k = getattr(base_model, 'top_k', 2) if hasattr(base_model, 'top_k') else 2
    
    # Check if this is a baseline model (no routing/gate)
    is_baseline_model = not (hasattr(base_model, 'gate') or hasattr(base_model, 'routed_experts'))
    
    with torch.no_grad():
        for batch in dataloader:
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                # Forward pass
                if activation_hook is not None and not is_baseline_model:
                    # Single forward pass: get both loss and routing info (MoE models only)
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
                            'categories': device_batch.get('categories', []),  # Include categories!
                            'titles': device_batch.get('title', batch.get('title', [])),  # May not be in batch
                            'abstracts': device_batch.get('abstract', batch.get('abstract', []))  # May not be in batch
                        }
                        
                        # Classify domains and track per-domain metrics
                        arxiv_ids = batch_metadata.get('arxiv_ids', [])
                        domains_list = batch_metadata.get('domains', [])
                        categories_list = batch_metadata.get('categories', [])
                        titles = batch_metadata.get('titles', [])
                        abstracts = batch_metadata.get('abstracts', [])
                        
                        # Determine batch size from input_ids
                        batch_size = input_ids.shape[0] if input_ids is not None else len(arxiv_ids)
                        
                        for i in range(batch_size):
                            # Classify domain (use both categories and domains, titles/abstracts if available)
                            # Get categories (ArXiv categories for ML detection)
                            categories = categories_list[i] if i < len(categories_list) else []
                            if not isinstance(categories, list):
                                categories = [categories] if categories else []
                            
                            # Get domains (NeMo Curator labels for healthcare detection)
                            domains = domains_list[i] if i < len(domains_list) else []
                            if not isinstance(domains, list):
                                domains = [domains] if domains else []
                            
                            paper_dict = {
                                'categories': categories,  # ArXiv categories
                                'domains': domains,  # NeMo Curator domain labels
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
                elif activation_hook is not None and is_baseline_model:
                    # Baseline model: no routing info, just compute loss normally
                    result = adapter.process_batch(batch)
                    loss = result['loss']
                    batch_metadata = result['batch_metadata']
                    
                    # Still track domains even without hook
                    device_batch = adapter._move_to_device(batch)
                    domains_list = device_batch.get('domains', [])
                    categories_list = device_batch.get('categories', [])
                    titles = device_batch.get('title', batch.get('title', []))
                    abstracts = device_batch.get('abstract', batch.get('abstract', []))
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
                        # Get categories (ArXiv categories for ML detection)
                        categories = categories_list[i] if i < len(categories_list) else []
                        if not isinstance(categories, list):
                            categories = [categories] if categories else []
                        
                        # Get domains (NeMo Curator labels for healthcare detection)
                        domains = domains_list[i] if i < len(domains_list) else []
                        if not isinstance(domains, list):
                            domains = [domains] if domains else []
                        
                        paper_dict = {
                            'categories': categories,  # ArXiv categories
                            'domains': domains,  # NeMo Curator domain labels
                            'title': titles[i] if i < len(titles) else '',
                            'abstract': abstracts[i] if i < len(abstracts) else ''
                        }
                        domain = classify_paper_domain(paper_dict)
                        
                        if i < len(loss_per_sample):
                            domain_losses[domain] += loss_per_sample[i].item()
                            domain_tokens[domain] += tokens_per_sample[i].item()
                            domain_paper_counts[domain] += 1
                else:
                    result = adapter.process_batch(batch)
                    loss = result['loss']
                    batch_metadata = result['batch_metadata']
                    
                    # Still track domains even without hook
                    device_batch = adapter._move_to_device(batch)
                    domains_list = device_batch.get('domains', [])
                    categories_list = device_batch.get('categories', [])
                    titles = device_batch.get('title', batch.get('title', []))
                    abstracts = device_batch.get('abstract', batch.get('abstract', []))
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
                        # Get categories (ArXiv categories for ML detection)
                        categories = categories_list[i] if i < len(categories_list) else []
                        if not isinstance(categories, list):
                            categories = [categories] if categories else []
                        
                        # Get domains (NeMo Curator labels for healthcare detection)
                        domains = domains_list[i] if i < len(domains_list) else []
                        if not isinstance(domains, list):
                            domains = [domains] if domains else []
                        
                        paper_dict = {
                            'categories': categories,  # ArXiv categories
                            'domains': domains,  # NeMo Curator domain labels
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
            num_tokens = (target_ids != adapter.ignore_index).sum().item()
            
            # Accumulate loss (weighted by tokens)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        print("\n🚨 CRITICAL ERROR: No tokens processed!")
        print("   This means:")
        print("   1. No test samples were loaded OR")
        print("   2. All sequences were empty after tokenization OR")
        print("   3. Tokenizer vocabulary mismatch causing all tokens to be <unk>")
        print(f"   Total batches processed: {len([None for _ in dataloader])}")
        return float('inf'), {}

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    # Debug: Warn if perplexity is suspiciously low (likely a calculation error)
    if perplexity < 2.0:
        print(f"\n⚠️  WARNING: Computed perplexity ({perplexity:.2f}) is suspiciously low!")
        print(f"   This may indicate a calculation error or data leakage.")
        print(f"   Total loss: {total_loss:.2f}, Total tokens: {total_tokens}, Avg loss: {avg_loss:.6f}")
        print(f"   For reference, typical perplexities: 10-50 (good), 50-200 (moderate), 200+ (poor)")
    
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
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        from train_real import SimpleMoEModel
        from train_baseline import BaselineTransformer
        
        # Load checkpoint first to infer model configuration
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect if this is a baseline model (check for BaselineTransformer-specific keys)
        # BaselineTransformer has 'transformer' but no 'gate' or 'routed_experts'
        is_baseline = False
        has_gate = any('gate' in key for key in state_dict.keys())
        has_routed_experts = any('routed_experts' in key for key in state_dict.keys())
        has_transformer = any('transformer' in key and 'transformer_layers' not in key for key in state_dict.keys())
        
        if has_transformer and not (has_gate or has_routed_experts):
            is_baseline = True
            print("Detected BaselineTransformer model")
        else:
            print("Detected SimpleMoEModel (MoE) model")
        
        if is_baseline:
            # Handle baseline model (encoder or decoder)
            # Infer embedding_dim from embedding layer
            embedding_dim = 256  # Default
            for key in state_dict.keys():
                if 'embedding.weight' in key:
                    weight = state_dict[key]
                    embedding_dim = weight.shape[1]
                    break
            
            # Infer num_layers from transformer encoder/decoder
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
            
            # Detect model type from checkpoint path (encoder or decoder)
            # Both models have same state_dict structure, so we detect from filename/path
            from train_baseline import BaselineTransformer, DecoderOnlyTransformer
            
            is_decoder = 'decoder' in model_checkpoint.lower() 
            
            if is_decoder:
                print(f"Inferred baseline config: embedding_dim={embedding_dim}, num_layers={num_layers}, type=decoder")
                base_model = DecoderOnlyTransformer(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                )
                print("Using DecoderOnlyTransformer (GPT-style, causal attention)")
            else:
                print(f"Inferred baseline config: embedding_dim={embedding_dim}, num_layers={num_layers}, type=encoder")
                base_model = BaselineTransformer(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                )
                print("Using BaselineTransformer (BERT-style, bidirectional attention)")
            
            # Wrap model (baseline doesn't need special handling)
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
            
            # Load checkpoint weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model.to(device)
            model.eval()
            print(f"Loaded baseline model from {model_checkpoint}")
        else:
            # Handle MoE model (existing code)
            # Infer model configuration from checkpoint state_dict
            # The gate layer has shape [embedding_dim, num_routed_experts]
            gate_key = None
            for key in state_dict.keys():
                # Handle different checkpoint formats: 'gate.weight', 'base_model.gate.weight', etc.
                if key.endswith('gate.weight'):
                    gate_key = key
                    break
            
            if gate_key:
                gate_weight = state_dict[gate_key]
                # PyTorch Linear layers store weights as [out_features, in_features]
                # gate is nn.Linear(embedding_dim, num_routed_experts)
                # So gate.weight shape is [num_routed_experts, embedding_dim]
                num_routed_experts = gate_weight.shape[0]
                embedding_dim = gate_weight.shape[1]
                print(f"Inferred from checkpoint: embedding_dim={embedding_dim}, num_routed_experts={num_routed_experts}")
            else:
                # Fallback: try to infer from routed_experts
                num_routed_experts = 4  # Default fallback
                embedding_dim = 256  # Default fallback
                for key in state_dict.keys():
                    if 'routed_experts.0.0.weight' in key or 'base_model.routed_experts.0.0.weight' in key:
                        # First linear layer in first routed expert: [4*embedding_dim, embedding_dim]
                        weight = state_dict[key]
                        embedding_dim = weight.shape[1]
                        break
                    elif 'shared_experts.0.0.weight' in key or 'base_model.shared_experts.0.0.weight' in key:
                        # First linear layer in first shared expert: [4*embedding_dim, embedding_dim]
                        weight = state_dict[key]
                        embedding_dim = weight.shape[1]
                        break
                
                # Count routed experts by counting expert modules
                routed_expert_count = 0
                for key in state_dict.keys():
                    if 'routed_experts.' in key or 'base_model.routed_experts.' in key:
                        # Extract expert index: routed_experts.{idx}.{layer}.weight
                        parts = key.split('.')
                        for i, part in enumerate(parts):
                            if part == 'routed_experts' and i + 1 < len(parts):
                                try:
                                    expert_idx = int(parts[i + 1])
                                    routed_expert_count = max(routed_expert_count, expert_idx + 1)
                                except ValueError:
                                    pass
                
                if routed_expert_count > 0:
                    num_routed_experts = routed_expert_count
                    print(f"Inferred from checkpoint: embedding_dim={embedding_dim}, num_routed_experts={num_routed_experts}")
                else:
                    print(f"Warning: Could not infer model config from checkpoint, using defaults: embedding_dim={embedding_dim}, num_routed_experts={num_routed_experts}")
            
            # Count shared experts
            num_shared_experts = 2  # Default
            shared_expert_count = 0
            for key in state_dict.keys():
                if 'shared_experts.' in key or 'base_model.shared_experts.' in key:
                    parts = key.split('.')
                    for i, part in enumerate(parts):
                        if part == 'shared_experts' and i + 1 < len(parts):
                            try:
                                expert_idx = int(parts[i + 1])
                                shared_expert_count = max(shared_expert_count, expert_idx + 1)
                            except ValueError:
                                pass
            
            if shared_expert_count > 0:
                num_shared_experts = shared_expert_count
            
            print(f"Creating model with: vocab_size={vocab_size}, embedding_dim={embedding_dim}, num_shared_experts={num_shared_experts}, num_routed_experts={num_routed_experts}")
            
            # Try to load routing parameters from config.yaml (for consistency)
            # Note: These don't affect inference routing, but help with consistency
            moe_params = {
                'noise_scale': 0.5,  # Default
                'load_balance_loss_weight': 0.1,  # Default
                'z_loss_weight': 0.001,  # Default
                'temperature_schedule': 'linear',
                'temperature_start': 2.0,
                'temperature_end': 0.1,
                'temperature_steps': 1000,
                'top_k': 2,
            }
            
            # Try to load from config.yaml
            try:
                from pathlib import Path
                import yaml
                config_path = Path('config.yaml')
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        if 'training' in config:
                            training_config = config['training']
                            if 'noise_scale' in training_config:
                                moe_params['noise_scale'] = training_config['noise_scale']
                            if 'load_balance_loss_weight' in training_config:
                                moe_params['load_balance_loss_weight'] = training_config['load_balance_loss_weight']
                            if 'z_loss_weight' in training_config:
                                moe_params['z_loss_weight'] = training_config['z_loss_weight']
                            if 'temperature_schedule' in training_config:
                                moe_params['temperature_schedule'] = training_config['temperature_schedule']
                            if 'temperature_start' in training_config:
                                moe_params['temperature_start'] = training_config['temperature_start']
                            if 'temperature_end' in training_config:
                                moe_params['temperature_end'] = training_config['temperature_end']
                            if 'temperature_steps' in training_config:
                                moe_params['temperature_steps'] = training_config['temperature_steps']
                            if 'top_k' in training_config:
                                moe_params['top_k'] = training_config['top_k']
            except Exception as e:
                # If config.yaml not found or error, use defaults
                pass
            
            base_model = SimpleMoEModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_shared_experts=num_shared_experts,
                num_routed_experts=num_routed_experts,
                top_k=moe_params['top_k'],
                noise_scale=moe_params['noise_scale'],
                load_balance_loss_weight=moe_params['load_balance_loss_weight'],
                z_loss_weight=moe_params['z_loss_weight'],
                temperature_schedule=moe_params['temperature_schedule'],
                temperature_start=moe_params['temperature_start'],
                temperature_end=moe_params['temperature_end'],
                temperature_steps=moe_params['temperature_steps'],
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
            
            # Load checkpoint weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model.to(device)
            model.eval()
            
            # Debug: Check gate weights to see if they're uniform
            if hasattr(base_model, 'gate') and hasattr(base_model.gate, 'weight'):
                gate_weights = base_model.gate.weight.detach().cpu().numpy()
                gate_std = np.std(gate_weights)
                gate_mean = np.mean(gate_weights)
                print(f"\nGate weight statistics:")
                print(f"  Mean: {gate_mean:.4f}, Std: {gate_std:.4f}")
                print(f"  Shape: {gate_weights.shape}")
                
                # Per-expert analysis
                expert_means = np.mean(gate_weights, axis=1)  # Mean weight per expert
                expert_stds = np.std(gate_weights, axis=1)   # Std weight per expert
                expert_norms = np.linalg.norm(gate_weights, axis=1)  # L2 norm per expert
                
                print(f"\n  Per-expert analysis:")
                for expert_idx in range(len(expert_means)):
                    print(f"    Expert {expert_idx}: mean={expert_means[expert_idx]:.6f}, std={expert_stds[expert_idx]:.4f}, norm={expert_norms[expert_idx]:.4f}")
                
                # Check if any expert has significantly smaller weights
                mean_of_means = np.mean(expert_means)
                std_of_means = np.std(expert_means)
                min_expert = np.argmin(expert_means)
                max_expert = np.argmax(expert_means)
                
                print(f"\n  Expert comparison:")
                print(f"    Mean of expert means: {mean_of_means:.6f}, Std: {std_of_means:.6f}")
                print(f"    Min expert (Expert {min_expert}): {expert_means[min_expert]:.6f}")
                print(f"    Max expert (Expert {max_expert}): {expert_means[max_expert]:.6f}")
                print(f"    Ratio (max/min): {expert_means[max_expert] / abs(expert_means[min_expert]) if expert_means[min_expert] != 0 else 'inf':.2f}")
                
                # Check if Expert 2 (or any expert) is significantly underutilized
                if len(expert_means) >= 3:
                    expert_2_mean = expert_means[2]
                    other_means = np.concatenate([expert_means[:2], expert_means[3:]])
                    other_mean = np.mean(other_means)
                    if abs(expert_2_mean) < abs(other_mean) * 0.5:  # Expert 2 is < 50% of others
                        print(f"\n  ⚠️  WARNING: Expert 2 has significantly smaller weights!")
                        print(f"     Expert 2 mean: {expert_2_mean:.6f}")
                        print(f"     Other experts mean: {other_mean:.6f}")
                        print(f"     This explains why Expert 2 is underutilized in routing.")
                
                if gate_std < 0.01:
                    print(f"\n  ⚠️  WARNING: Gate weights are nearly uniform (std={gate_std:.4f})")
                    print(f"     This explains why expert probabilities are identical.")
                    print(f"     The model may need more training or more aggressive specialization parameters.")
                else:
                    print(f"\n  ✓ Gate weights show diversity (std={gate_std:.4f})")
                    
                # Check if weights are too small (could indicate poor initialization or training)
                if np.max(np.abs(gate_weights)) < 0.1:
                    print(f"\n  ⚠️  WARNING: Gate weights are very small (max abs: {np.max(np.abs(gate_weights)):.4f})")
                    print(f"     This could indicate the model hasn't learned strong routing preferences.")
            
            print(f"Loaded MoE model from {model_checkpoint}")
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

    # If no text files found (directory missing), create paper list from metadata
    if not all_files:
        print("No text files found. Creating paper list from metadata with embedded text...")
        all_files = []
        paper_count = 0

        # Debug: Check metadata structure
        print(f"Total metadata entries: {len(metadata)}")
        first_few_keys = list(metadata.keys())[:3]
        print(f"First few paper IDs: {first_few_keys}")

        for arxiv_id, meta in metadata.items():
            # Debug: Show structure of first few papers
            if paper_count < 3:
                print(f"\nPaper {arxiv_id}:")
                print(f"  Available keys: {list(meta.keys())}")
                print(f"  Has text field: {'text' in meta}")
                if 'text' in meta:
                    text_len = len(meta['text']) if meta['text'] else 0
                    print(f"  Text length: {text_len}")

            # Check if paper has text content in metadata
            if meta.get('text') and meta.get('text').strip():
                # Use embedded text - mark with special prefix
                all_files.append((arxiv_id, f"embedded_text:{arxiv_id}"))
                paper_count += 1
                # Limit to reasonable number for evaluation
                if paper_count >= 100:  # Reduced to 100 for debugging
                    break

        print(f"Created {len(all_files)} papers from embedded metadata text")

        if not all_files:
            raise ValueError("No papers with text content found in metadata. Cannot proceed with evaluation.")
    
    # Classify all papers to get domain distribution
    print("Classifying papers for stratified split...")
    file_domains = []
    sample_count = 0
    for arxiv_id, _ in all_files:
        meta = metadata.get(arxiv_id, {})
        domains = meta.get('domains', [])
        categories = meta.get('categories', [])  # Get ArXiv categories
        title = meta.get('title', '')
        abstract = meta.get('abstract', '')
        
        # Debug: Show first 5 papers
        if sample_count < 5:
            print(f"\n  {arxiv_id}:")
            print(f"    Metadata domains: {domains} (type: {type(domains)})")
            print(f"    Metadata categories: {categories} (type: {type(categories)})")
        
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
        if sample_count < 5:
            print(f"    Combined categories for classification: {all_cats}")
            print(f"    Classified as: {domain}")
        file_domains.append((arxiv_id, domain))
        sample_count += 1
    
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
        # Use embedded text from metadata instead of requiring separate text files
        for arxiv_id in domain_test:
            # Check if paper has text content in metadata
            meta = metadata.get(arxiv_id, {})
            if meta.get('text') and meta.get('text').strip():
                # Use arxiv_id as "file path" since we're using embedded text
                test_files.append((arxiv_id, f"embedded_text:{arxiv_id}"))

        for arxiv_id in domain_train:
            # Check if paper has text content in metadata
            meta = metadata.get(arxiv_id, {})
            if meta.get('text') and meta.get('text').strip():
                # Use arxiv_id as "file path" since we're using embedded text
                train_files.append((arxiv_id, f"embedded_text:{arxiv_id}"))
    
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
    # Reuse metadata from full_dataset to avoid reloading and losing categories
    class TestDataset(ArXivStreamingDataset):
        def __init__(self, text_files, metadata_dict, text_dir, metadata_jsonl, tokenizer, max_length=512, min_length=64, shuffle_buffer=100, seed=None):
            # Don't call super().__init__() which would reload metadata
            # Instead, manually set up with existing metadata that already has categories
            self.text_dir = text_dir
            self.metadata_jsonl = metadata_jsonl
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.min_length = min_length
            self.shuffle_buffer = shuffle_buffer
            self.seed = seed

            # Use the metadata dictionary that already has categories loaded
            self.metadata = metadata_dict
            self.text_files = text_files
            self._estimated_length = None

        def _get_text_files(self) -> List[Tuple[str, str]]:
            """Override to use the provided text_files instead of scanning directory."""
            return self.text_files

        def _load_text(self, arxiv_id: str, file_path: str) -> str:
            """Load text from metadata instead of separate files."""
            if file_path.startswith("embedded_text:"):
                # Use embedded text from metadata
                meta = self.metadata.get(arxiv_id, {})
                text = meta.get('text', '')
                if not text:
                    print(f"DEBUG: No text found in metadata for {arxiv_id}")
                    raise ValueError(f"No text found in metadata for {arxiv_id}")

                # Debug: Show text sample
                text_sample = text[:100] + "..." if len(text) > 100 else text
                print(f"DEBUG: Loaded text for {arxiv_id} (length: {len(text)}): {text_sample}")
                return text
            else:
                # Fallback to original file-based loading
                print(f"DEBUG: Using file-based loading for {arxiv_id}")
                return super()._load_text(arxiv_id, file_path)
    
    test_dataset = TestDataset(
        test_files,
        full_dataset.metadata,  # Reuse metadata with categories already loaded
        dataset_text_dir,
        dataset_metadata,
        tokenizer,
        max_length=512,
        min_length=64
    )
    
    print(f"Created test dataset: {len(test_files)} papers")

    # Debug: Check if we actually have test data
    if len(test_files) == 0:
        print("\n🚨 CRITICAL ERROR: No test files found!")
        print("   This means the test/train split failed to create any test samples.")
        print("   Possible causes:")
        print("   1. No papers with embedded text in metadata")
        print("   2. Test split parameter removed all papers")
        print("   3. All papers were filtered out during processing")
        return None

    # Debug: Sample first test paper
    first_arxiv_id, first_path = test_files[0]
    print(f"   Sample test paper: {first_arxiv_id}")
    print(f"   Sample path: {first_path}")

    # Create test dataloader
    # Use num_workers=0 to avoid worker serialization issues with metadata dict
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,  # Single process to avoid metadata sharing issues
        pin_memory=True
    )
    
    # Initialize activation hook for capturing expert routing
    print("\nInitializing expert activation capture...")
    activation_hook = ExpertActivationHook(
        model, 
        text_dir=dataset_text_dir,
        metadata_dict=full_dataset.metadata  # Pass metadata for fallback category lookup
    )
    
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
    # Use user-specified output_dir, but check if it's in Drive and use Drive path if available
    # Only override if output_dir is the default "./evaluations"
    if output_dir == "./evaluations" or output_dir == "evaluations":
        results_dir = get_drive_results_path(output_dir)
    else:
        # User specified a custom directory - respect it
        results_dir = output_dir
    
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
        
        if len(activation_hook.paper_ids) > 0:
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
            # Debug: Print expert activation statistics
            expert_activations = activations.get('expert_activations')
            expert_probs = activations.get('expert_probs')
            if expert_activations is not None and expert_probs is not None:
                num_samples, num_experts = expert_activations.shape
                print(f"\n{'='*60}")
                print("EXPERT ACTIVATION ANALYSIS")
                print(f"{'='*60}")
                print(f"Total samples: {num_samples}, Total experts: {num_experts}")
                
                # Check activation diversity
                activation_counts = expert_activations.sum(axis=0)  # Count activations per expert
                activation_std = np.std(activation_counts)
                activation_mean = np.mean(activation_counts)
                print(f"\nActivation counts per expert:")
                print(f"  Mean: {activation_mean:.1f}, Std: {activation_std:.1f}")
                print(f"  Min: {activation_counts.min()}, Max: {activation_counts.max()}")
                for expert_id in range(num_experts):
                    print(f"  Expert {expert_id}: {activation_counts[expert_id]} activations")
                
                # Check probability diversity - find a sample with non-zero probabilities
                if expert_probs.shape[0] > 0:
                    # Find first sample with non-zero probabilities
                    sample_idx = None
                    for i in range(expert_probs.shape[0]):
                        if np.any(expert_probs[i] > 0):
                            sample_idx = i
                            break
                    
                    if sample_idx is not None:
                        sample_probs = expert_probs[sample_idx]
                        probs_std = np.std(sample_probs)
                        probs_mean = np.mean(sample_probs)
                        print(f"\nProbability diversity (sample {sample_idx}, first with non-zero probs):")
                        print(f"  Mean: {probs_mean:.4f}, Std: {probs_std:.4f}")
                        print(f"  Probabilities: {sample_probs}")
                        if probs_std < 1e-6:
                            print(f"  ⚠️  WARNING: Expert probabilities are nearly identical!")
                            print(f"     This suggests the model hasn't learned to differentiate experts.")
                        else:
                            print(f"  ✓ Expert probabilities show diversity")
                else:
                    # All samples have zero probabilities
                    print(f"\nProbability diversity:")
                    print(f"  ⚠️  WARNING: All samples have zero probabilities!")
                    print(f"     This indicates a bug in probability computation/storage.")
                    # Show statistics across all samples
                    non_zero_count = np.sum(expert_probs > 0)
                    total_count = expert_probs.size
                    print(f"     Non-zero probabilities: {non_zero_count}/{total_count} ({100*non_zero_count/total_count:.1f}%)")
                
                # Check if all experts activate on same papers
                if activation_std < 1.0:
                    print(f"\n⚠️  WARNING: Experts have very similar activation counts (std={activation_std:.1f})")
                    print(f"   This suggests they're routing identically.")
                else:
                    print(f"\n✓ Experts show different activation patterns (std={activation_std:.1f})")
                print(f"{'='*60}\n")
            
            # Save expert activations to the same directory as other evaluation results
            # Use results_dir (which respects the user-specified output_dir, including subdirectories)
            activations_dir = results_dir
            
            os.makedirs(activations_dir, exist_ok=True)
            activations_path = os.path.join(activations_dir, 'expert_activations.npz')
            
            save_expert_activations(activations, activations_path)
            print(f"\n✅ Expert activations saved to: {activations_path}")
            results['expert_activations_path'] = activations_path
    
    # Save embeddings for cluster analysis in notebook
    if embeddings is not None and len(embeddings) > 0:
        embeddings_path = os.path.join(results_dir, "embeddings.npz")
        
        # Convert embeddings to numpy
        embeddings_np = embeddings.numpy() if hasattr(embeddings, 'numpy') else np.array(embeddings)
        
        # Convert metadata to format expected by notebook
        # Handle variable-length lists by converting to object arrays or strings
        arxiv_ids = [m.get('arxiv_id', '') for m in metadata]
        domains_list = [m.get('domains', []) for m in metadata]
        years = [m.get('year', None) for m in metadata]
        has_nd = [m.get('has_neurodegeneration', False) for m in metadata]
        
        # Convert domains to strings (join lists) for numpy compatibility
        domains_str = []
        for domain in domains_list:
            if isinstance(domain, list):
                domains_str.append(','.join(str(d) for d in domain) if domain else '')
            else:
                domains_str.append(str(domain) if domain else '')
        
        # Save with proper numpy array types
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings_np,
            arxiv_ids=np.array(arxiv_ids, dtype=object),  # Object array for variable-length strings
            domains=np.array(domains_str, dtype=object),  # Object array for variable-length strings
            years=np.array(years, dtype=object),  # Object array to handle None values
            has_neurodegeneration=np.array(has_nd, dtype=bool)
        )
        print(f"\n✅ Embeddings saved to: {embeddings_path}")
        print(f"   Embeddings shape: {embeddings_np.shape}")
        print(f"   Metadata: {len(arxiv_ids)} papers")
        results['embeddings_path'] = embeddings_path
    
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

