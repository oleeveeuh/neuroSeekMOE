"""
Training Adapter for DeepSeekMoE Model

Connects streaming ArXiv dataset to DeepSeekMoE model for efficient training.
Handles batching, device transfer, forward pass, and loss computation.

Usage:
    from training_adapter import ModelAdapter
    
    adapter = ModelAdapter(
        model=model,
        device='cuda',
        domain_weights={'neurodegeneration': 1.5, 'neuroscience': 1.2}
    )
    
    for batch in dataloader:
        result = adapter.process_batch(batch)
        loss = result['loss']
        # ... backprop, optimizer step, etc.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
import numpy as np


class ModelAdapter:
    """Adapter connecting streaming dataset to DeepSeekMoE model.
    
    Handles:
    - Device transfer
    - Variable-length sequence padding
    - Forward pass
    - Loss computation with domain-aware weighting
    - Batch metadata preservation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device] = 'cuda',
        domain_weights: Optional[Dict[str, float]] = None,
        ignore_index: int = 0,  # Padding token ID
        reduction: str = 'mean'
    ):
        """Initialize model adapter.
        
        Args:
            model: DeepSeekMoE model with signature model(input_ids) -> logits
            device: Device to run on ('cuda', 'cpu', or torch.device)
            domain_weights: Optional dict mapping domain names to loss weights
                Example: {'neurodegeneration': 1.5, 'neuroscience': 1.2}
            ignore_index: Token ID to ignore in loss computation (padding token)
            reduction: Loss reduction method ('mean', 'sum', or 'none')
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.domain_weights = domain_weights or {}
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Will be set to train mode by training loop
        
        print(f"ModelAdapter initialized:")
        print(f"   Device: {self.device}")
        print(f"   Domain weights: {self.domain_weights}")
        print(f"   Ignore index: {self.ignore_index}")
    
    def _get_domain_weight(self, domains: List[str], has_neurodegeneration: bool) -> float:
        """Get loss weight for a sample based on domains.
        
        Args:
            domains: List of domain labels
            has_neurodegeneration: Boolean flag for neurodegeneration
            
        Returns:
            Loss weight multiplier
        """
        if not self.domain_weights:
            return 1.0
        
        weight = 1.0
        
        # Check neurodegeneration (highest priority)
        if has_neurodegeneration and 'neurodegeneration' in self.domain_weights:
            weight *= self.domain_weights['neurodegeneration']
        
        # Check other domains
        for domain in domains:
            if domain in self.domain_weights:
                weight *= self.domain_weights[domain]
        
        return weight
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device efficiently.
        
        Args:
            batch: Batch dictionary from DataLoader
            
        Returns:
            Batch with tensors moved to device
        """
        device_batch = {}
        
        # Move tensor fields
        if 'input_ids' in batch:
            device_batch['input_ids'] = batch['input_ids'].to(self.device, non_blocking=True)
        if 'target_ids' in batch:
            device_batch['target_ids'] = batch['target_ids'].to(self.device, non_blocking=True)
        if 'has_neurodegeneration' in batch:
            device_batch['has_neurodegeneration'] = batch['has_neurodegeneration'].to(
                self.device, non_blocking=True
            )
        
        # Keep metadata on CPU (no need to move)
        device_batch['domains'] = batch.get('domains', [])
        device_batch['years'] = batch.get('years', [])
        device_batch['arxiv_ids'] = batch.get('arxiv_ids', [])
        
        return device_batch
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        domain_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss with optional domain weighting.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            target_ids: Target token IDs [batch, seq_len]
            domain_weights: Optional per-sample weights [batch]
            
        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy: [batch*seq_len, vocab_size] and [batch*seq_len]
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)
        
        # Compute base loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction='none'  # Get per-token losses
        )
        
        # Reshape back to [batch, seq_len]
        loss = loss.view(batch_size, seq_len)
        
        # Apply domain weights if provided
        if domain_weights is not None:
            # Expand domain_weights to match sequence length
            domain_weights = domain_weights.unsqueeze(1).expand(batch_size, seq_len)
            loss = loss * domain_weights
        
        # Reduce to scalar
        if self.reduction == 'mean':
            # Mean over non-ignored tokens
            mask = (targets_flat != self.ignore_index).view(batch_size, seq_len)
            if mask.sum() > 0:
                loss = (loss * mask.float()).sum() / mask.sum()
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:  # 'none'
            pass  # Keep as [batch, seq_len]
        
        return loss
    
    def process_batch(self, batch: Dict) -> Dict:
        """Process a batch through the model.
        
        Args:
            batch: Batch dictionary from DataLoader with keys:
                - input_ids: [batch, seq_len]
                - target_ids: [batch, seq_len]
                - domains: List[List[str]]
                - years: List[int]
                - arxiv_ids: List[str]
                - has_neurodegeneration: [batch] (bool tensor)
        
        Returns:
            Dictionary with:
                - loss: Scalar loss tensor
                - logits: [batch, seq_len, vocab_size]
                - batch_metadata: Dict with domains, years, arxiv_ids
        """
        # Handle empty batch (edge case: last batch may be empty)
        if not batch or 'input_ids' not in batch or batch['input_ids'].numel() == 0:
            return {
                'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'logits': None,
                'batch_metadata': {}
            }
        
        # Move to device (this only moves tensors, not lists like categories)
        device_batch = self._move_to_device(batch)
        
        # Debug: Check batch contents (first batch only)
        if not hasattr(self, '_debug_batch_checked'):
            self._debug_batch_checked = True
            print(f"DEBUG ModelAdapter.process_batch: batch keys: {list(batch.keys())}")
            print(f"DEBUG ModelAdapter.process_batch: device_batch keys: {list(device_batch.keys())}")
            if 'categories' in batch:
                print(f"DEBUG ModelAdapter.process_batch: batch['categories'] type: {type(batch['categories'])}, len: {len(batch['categories']) if batch['categories'] else 0}")
                if batch['categories'] and len(batch['categories']) > 0:
                    print(f"DEBUG ModelAdapter.process_batch: batch['categories'][0]: {batch['categories'][0]}")
            else:
                print(f"DEBUG ModelAdapter.process_batch: 'categories' NOT in batch!")
        
        input_ids = device_batch['input_ids']
        target_ids = device_batch['target_ids']
        domains = device_batch['domains']
        has_neurodegeneration = device_batch.get('has_neurodegeneration', None)
        
        # Handle variable sequence lengths (already padded by collate_fn, but verify)
        batch_size, seq_len = input_ids.shape
        
        # Forward pass
        logits = self.model(input_ids)  # [batch, seq_len, vocab_size]
        
        # Compute domain weights if enabled
        domain_weights_tensor = None
        if self.domain_weights and has_neurodegeneration is not None:
            weights_list = []
            for i in range(batch_size):
                weight = self._get_domain_weight(
                    domains[i] if i < len(domains) else [],
                    has_neurodegeneration[i].item() if has_neurodegeneration is not None else False
                )
                weights_list.append(weight)
            domain_weights_tensor = torch.tensor(
                weights_list, device=self.device, dtype=torch.float32
            )
        
        # Compute loss
        loss = self._compute_loss(logits, target_ids, domain_weights_tensor)
        
        # Get categories from batch (categories are lists, not tensors, so get from original batch)
        categories = batch.get('categories', device_batch.get('categories', []))
        
        # Debug: Check if categories are in batch (first batch only)
        if not hasattr(self, '_debug_categories_checked'):
            self._debug_categories_checked = True
            has_categories_in_batch = 'categories' in batch
            has_categories_in_device = 'categories' in device_batch
            categories_len = len(categories) if categories else 0
            print(f"DEBUG ModelAdapter: has_categories_in_batch={has_categories_in_batch}, has_categories_in_device={has_categories_in_device}, categories_len={categories_len}")
            if categories and len(categories) > 0:
                print(f"DEBUG ModelAdapter: sample categories[0]={categories[0]}")
        
        # Prepare batch metadata
        batch_metadata = {
            'domains': domains,
            'categories': categories,  # Include original ArXiv categories
            'years': device_batch.get('years', []),
            'arxiv_ids': device_batch.get('arxiv_ids', []),
            'titles': batch.get('title', []),  # Get from original batch (not device_batch)
            'abstracts': batch.get('abstract', []),  # Get from original batch (not device_batch)
            'batch_size': batch_size,
            'seq_len': seq_len,
        }
        
        return {
            'loss': loss,
            'logits': logits,
            'batch_metadata': batch_metadata
        }
    
    def debug_batch(self, batch: Dict, print_details: bool = True) -> Dict:
        """Debug function to inspect batch shapes and sanity-check tensors.
        
        Args:
            batch: Batch dictionary from DataLoader
            print_details: Whether to print detailed information
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {}
        
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
            debug_info['input_ids'] = {
                'shape': list(input_ids.shape),
                'dtype': str(input_ids.dtype),
                'device': str(input_ids.device),
                'min': input_ids.min().item(),
                'max': input_ids.max().item(),
                'mean': input_ids.float().mean().item(),
            }
        
        if 'target_ids' in batch:
            target_ids = batch['target_ids']
            debug_info['target_ids'] = {
                'shape': list(target_ids.shape),
                'dtype': str(target_ids.dtype),
                'device': str(target_ids.device),
                'min': target_ids.min().item(),
                'max': target_ids.max().item(),
            }
        
        if 'domains' in batch:
            debug_info['domains'] = {
                'count': len(batch['domains']),
                'sample': batch['domains'][:3] if len(batch['domains']) > 0 else []
            }
        
        if 'years' in batch:
            debug_info['years'] = {
                'count': len(batch['years']),
                'sample': batch['years'][:3] if len(batch['years']) > 0 else [],
                'min': min(batch['years']) if batch['years'] else None,
                'max': max(batch['years']) if batch['years'] else None,
            }
        
        if 'arxiv_ids' in batch:
            debug_info['arxiv_ids'] = {
                'count': len(batch['arxiv_ids']),
                'sample': batch['arxiv_ids'][:3] if len(batch['arxiv_ids']) > 0 else []
            }
        
        if 'has_neurodegeneration' in batch:
            has_nd = batch['has_neurodegeneration']
            debug_info['has_neurodegeneration'] = {
                'shape': list(has_nd.shape),
                'dtype': str(has_nd.dtype),
                'sum': has_nd.sum().item() if has_nd.dtype == torch.bool else has_nd.sum().item(),
                'count': len(has_nd),
            }
        
        if print_details:
            print("=" * 60)
            print("Batch Debug Information")
            print("=" * 60)
            
            for key, info in debug_info.items():
                print(f"\n{key}:")
                if isinstance(info, dict):
                    for subkey, value in info.items():
                        print(f"   {subkey}: {value}")
                else:
                    print(f"   {info}")
            
            # Sanity checks
            print("\nSanity Checks:")
            
            if 'input_ids' in debug_info and 'target_ids' in debug_info:
                input_shape = debug_info['input_ids']['shape']
                target_shape = debug_info['target_ids']['shape']
                
                if input_shape == target_shape:
                    print("   OK input_ids and target_ids have matching shapes")
                else:
                    print(f"   FAIL Shape mismatch: input_ids {input_shape} vs target_ids {target_shape}")
                
                # Check sequence length consistency
                if input_shape[0] == target_shape[0]:
                    print(f"   OK Batch size consistent: {input_shape[0]}")
                else:
                    print(f"   FAIL Batch size mismatch: {input_shape[0]} vs {target_shape[0]}")
            
            if 'input_ids' in debug_info:
                input_min = debug_info['input_ids']['min']
                input_max = debug_info['input_ids']['max']
                if input_min >= 0:
                    print(f"   OK input_ids in valid range: [{input_min}, {input_max}]")
                else:
                    print(f"   FAIL input_ids has negative values: min={input_min}")
            
            print("=" * 60)
        
        return debug_info


# Example usage and testing
if __name__ == "__main__":
    """Example usage of ModelAdapter."""
    print("=" * 60)
    print("ModelAdapter Example")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from training_adapter import ModelAdapter")
    print("  import torch.nn as nn")
    print()
    print("  # Create dummy model (replace with your DeepSeekMoE)")
    print("  class DummyModel(nn.Module):")
    print("      def __init__(self, vocab_size=50000):")
    print("          super().__init__()")
    print("          self.embedding = nn.Embedding(vocab_size, 768)")
    print("          self.lm_head = nn.Linear(768, vocab_size)")
    print()
    print("      def forward(self, input_ids):")
    print("          x = self.embedding(input_ids)")
    print("          logits = self.lm_head(x)")
    print("          return logits")
    print()
    print("  model = DummyModel(vocab_size=50000)")
    print()
    print("  # Create adapter")
    print("  adapter = ModelAdapter(")
    print("      model=model,")
    print("      device='cuda' if torch.cuda.is_available() else 'cpu',")
    print("      domain_weights={'neurodegeneration': 1.5, 'neuroscience': 1.2}")
    print("  )")
    print()
    print("  # Process batch")
    print("  for batch in dataloader:")
    print("      # Debug batch")
    print("      adapter.debug_batch(batch)")
    print()
    print("      # Process batch")
    print("      result = adapter.process_batch(batch)")
    print("      loss = result['loss']")
    print("      logits = result['logits']")
    print("      metadata = result['batch_metadata']")
    print()
    print("      # Backward pass")
    print("      loss.backward()")
    print("      # ... optimizer step ...")
    print()
    print("ModelAdapter ready to use!")

