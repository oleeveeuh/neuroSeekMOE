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

# math is already imported, so we can use math.cos and math.pi

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


class ExpertChoiceSTE(torch.autograd.Function):
    """Expert Choice routing with Straight-Through Estimator.
    
    Forward pass: Returns token indices selected by each expert (discrete).
    Backward pass: Gradients flow through soft probabilities (continuous).
    This implements Expert Choice routing where experts choose tokens.
    """
    
    @staticmethod
    def forward(ctx, logits, k):
        """
        Args:
            logits: [batch, num_experts] tensor of router logits (token-to-expert scores)
            k: Number of top tokens each expert should select
            
        Returns:
            token_indices: [num_experts, k] tensor of token indices selected by each expert
            expert_probs: [num_experts, k] tensor of soft probabilities for gradient flow
        """
        batch_size, num_experts = logits.shape
        
        # Transpose: [batch, num_experts] -> [num_experts, batch]
        # Each expert now sees scores for all tokens
        expert_logits = logits.t()  # [num_experts, batch]
        
        # Each expert selects top-k tokens
        topk_values, topk_indices = torch.topk(expert_logits, k, dim=-1)  # [num_experts, k]
        
        # Compute soft probabilities for backward pass
        expert_probs = torch.softmax(topk_values, dim=-1)  # [num_experts, k]
        
        # Store for backward pass
        ctx.save_for_backward(expert_probs, topk_indices, logits)
        ctx.k = k
        ctx.batch_size = batch_size
        ctx.num_experts = num_experts
        
        return topk_indices, expert_probs
    
    @staticmethod
    def backward(ctx, grad_indices, grad_probs):
        """
        Backward pass: Use soft probabilities to allow gradients to flow through routing.
        """
        expert_probs, topk_indices, logits = ctx.saved_tensors
        k = ctx.k
        batch_size = ctx.batch_size
        num_experts = ctx.num_experts
        
        # Initialize gradient for logits
        grad_logits = None
        
        if ctx.needs_input_grad[0]:
            expert_logits = logits.t()  # [num_experts, batch]
            grad_logits = torch.zeros_like(logits)  # [batch, num_experts]
            
            # For each expert, distribute gradients through soft probabilities
            for expert_idx in range(num_experts):
                # Get gradients for this expert's selected tokens
                grad_from_probs = grad_probs[expert_idx]  # [k]
                selected_token_indices = topk_indices[expert_idx]  # [k]
                
                # Distribute gradient proportionally to soft probability
                for i in range(k):
                    token_idx = selected_token_indices[i].item()
                    prob_weight = expert_probs[expert_idx, i]
                    grad_logits[token_idx, expert_idx] += grad_from_probs[i] * prob_weight
        
        return grad_logits, None  # No gradient for k


class TopKGatingSTE(torch.autograd.Function):
    """Straight-Through Estimator for top-k gating (Token Choice - legacy).
    
    Forward pass: Returns hard one-hot assignments for top-k experts (discrete).
    Backward pass: Gradients flow through soft probabilities (continuous).
    This allows better gradient flow through discrete routing decisions.
    
    NOTE: This is kept for backward compatibility. Expert Choice routing is preferred.
    """
    
    @staticmethod
    def forward(ctx, logits, k):
        """
        Args:
            logits: [batch, num_experts] tensor of router logits
            k: Number of top experts to select
            
        Returns:
            hard_assignments: [batch, num_experts] one-hot tensor for top-k experts
            topk_indices: [batch, k] indices of selected experts
        """
        # Get top-k experts
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)  # [batch, k]
        
        # Compute soft probabilities for backward pass
        soft_probs = torch.softmax(topk_values, dim=-1)  # [batch, k]
        
        # Create hard one-hot assignments (discrete for forward pass)
        batch_size, num_experts = logits.shape
        hard_assignments = torch.zeros(batch_size, num_experts, device=logits.device, dtype=logits.dtype)
        
        # Create one-hot encoding: each row has k ones at top-k positions
        batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand(-1, k)  # [batch, k]
        hard_assignments[batch_indices, topk_indices] = 1.0 / k  # Uniform weighting for top-k
        
        # Store soft probabilities for backward pass
        ctx.save_for_backward(soft_probs, topk_indices, logits)
        ctx.k = k
        
        return hard_assignments, topk_indices
    
    @staticmethod
    def backward(ctx, grad_hard, grad_indices):
        """
        Backward pass: Use soft probabilities instead of hard assignments.
        This allows gradients to flow through the routing decision as if it was continuous.
        
        The key insight: gradients flow through soft probabilities (continuous),
        even though forward pass used hard assignments (discrete).
        """
        soft_probs, topk_indices, logits = ctx.saved_tensors
        k = ctx.k
        
        # Initialize gradient for logits
        grad_logits = None
        
        if ctx.needs_input_grad[0]:
            batch_size, num_experts = logits.shape
            grad_logits = torch.zeros_like(logits)
            
            # Get top-k logits and compute softmax for gradient computation
            topk_values = torch.gather(logits, dim=1, index=topk_indices)  # [batch, k]
            
            # Compute soft probabilities for all top-k experts
            # This is what we use for gradient flow (continuous approximation)
            soft_probs_normalized = torch.softmax(topk_values, dim=-1)  # [batch, k]
            
            # Extract gradients at top-k positions from grad_hard
            batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand(-1, k)  # [batch, k]
            
            # For each top-k expert, distribute gradient through soft probability
            # This is the STE: use soft probabilities in backward even though forward used hard
            for i in range(k):
                # Get gradient flowing back through hard assignment at this position
                grad_from_hard = grad_hard[batch_indices, topk_indices[:, i]]  # [batch]
                
                # Distribute gradient proportionally to soft probability
                # This allows the routing decision to learn via gradient flow
                grad_logits[batch_indices, topk_indices[:, i]] += (
                    grad_from_hard.unsqueeze(1) * soft_probs_normalized[:, i:i+1]
                )
        
        return grad_logits, None  # No gradient for k


def expert_choice_routing(gate_logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expert Choice routing: each expert selects top-k tokens to process.
    
    Unlike token-choice routing (where tokens select experts), Expert Choice has each
    expert selecting top-k tokens, enabling better load balancing and capacity control.
    
    Args:
        gate_logits: [batch*seq_len, num_routed_experts] tensor of router logits
            Each row represents a token's scores for all experts.
            Each column represents an expert's scores for all tokens.
        k: Number of tokens each expert should select (top-k selection)
    
    Returns:
        token_indices: [num_routed_experts, k] tensor of token indices selected by each expert
            Values are indices into the flattened batch*seq_len dimension (0 to batch*seq_len-1)
        expert_probs: [num_routed_experts, k] tensor of soft probabilities for selected tokens
            Used for gradient flow through routing decisions
    
    Implementation Details:
    1. Transpose gate_logits to get expert-centric view: [num_experts, batch*seq_len]
    2. For each expert, compute softmax over all tokens to get selection probabilities
    3. Select top-k tokens with highest probabilities using torch.topk
    4. Return token indices and their probabilities
    
    Edge Cases:
    - If k exceeds number of available tokens, k is clamped to num_tokens
    - If num_experts is 0, returns empty tensors
    """
    num_tokens, num_experts = gate_logits.shape
    
    # Handle edge cases
    if num_tokens == 0 or num_experts == 0:
        # Return empty tensors with correct shapes
        empty_indices = torch.empty((num_experts, 0), dtype=torch.long, device=gate_logits.device)
        empty_probs = torch.empty((num_experts, 0), dtype=gate_logits.dtype, device=gate_logits.device)
        return empty_indices, empty_probs
    
    # Clamp k to not exceed available tokens
    k = min(k, num_tokens)
    
    # Transpose: [num_tokens, num_experts] -> [num_experts, num_tokens]
    # Now each row represents an expert seeing scores for all tokens
    expert_logits = gate_logits.t()  # [num_experts, num_tokens]
    
    # For each expert, compute softmax over all tokens to get selection probabilities
    # This gives each expert a probability distribution over all tokens
    expert_probs_all = torch.softmax(expert_logits, dim=-1)  # [num_experts, num_tokens]
    
    # For each expert, select top-k tokens with highest probabilities
    # topk_values: [num_experts, k] - probabilities of selected tokens
    # topk_indices: [num_experts, k] - indices of selected tokens (0 to num_tokens-1)
    topk_values, topk_indices = torch.topk(expert_probs_all, k, dim=-1)  # [num_experts, k]
    
    # Normalize the selected probabilities (softmax over top-k selections)
    # This ensures probabilities sum to 1 for each expert's selected tokens
    expert_probs = torch.softmax(topk_values, dim=-1)  # [num_experts, k]
    
    # Return token indices and their probabilities
    # token_indices: [num_experts, k] - indices into flattened batch*seq_len
    # expert_probs: [num_experts, k] - probabilities for gradient flow
    return topk_indices, expert_probs


def top_k_gating_ste(logits, k=2):
    """Top-k gating with Straight-Through Estimator for better gradient flow.
    
    Args:
        logits: [batch, num_experts] tensor of router logits
        k: Number of top experts to select
        
    Returns:
        hard_assignments: [batch, num_experts] one-hot tensor for top-k experts
        topk_indices: [batch, k] indices of selected experts
    """
    return TopKGatingSTE.apply(logits, k)


def top_k_gating(logits, k=2):
    """Select top-k experts per token and return their indices and normalized weights.
    
    Legacy function for backward compatibility. Consider using top_k_gating_ste for better gradients.
    """
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


def load_balance_loss(gate_logits: torch.Tensor) -> torch.Tensor:
    """Compute auxiliary load balancing loss to encourage balanced expert utilization.
    
    This loss penalizes uneven expert utilization by measuring how evenly probability
    mass is distributed across experts. It encourages the model to use all experts
    rather than concentrating on a few.
    
    Reference: DeepSeek paper uses importance-weighted load balancing where tokens
    the model is more confident about contribute more to the load balancing objective.
    This encourages balance in high-confidence assignments while allowing flexibility
    in low-confidence ones.
    
    Args:
        gate_logits: [batch*seq_len, num_routed_experts] tensor of router logits
            Each row represents a token's scores for all experts
    
    Returns:
        Scalar loss tensor - penalty for imbalanced expert utilization
        Higher values indicate more imbalanced routing
    
    Implementation Details:
    1. Compute softmax probabilities: probs = softmax(gate_logits, dim=1)
    2. Compute importance weights: importance = max(probs, dim=1)[0]
       (importance of each token = max probability it assigns to any expert)
    3. Compute expert assignment rates: expert_fraction = mean(probs, dim=0)
       (fraction of total probability mass going to each expert on average)
    4. Compute load balance loss: penalizes uneven expert utilization
    5. Return loss * num_routed_experts for scaling
    """
    if gate_logits is None or gate_logits.numel() == 0:
        return torch.tensor(0.0, device=gate_logits.device if gate_logits is not None else 'cpu')
    
    num_tokens, num_experts = gate_logits.shape
    
    # Step 1: Compute softmax probabilities: probs = softmax(gate_logits, dim=1)
    # probs: [batch*seq_len, num_routed_experts]
    # Each row sums to 1.0, representing probability distribution over experts for that token
    probs = torch.softmax(gate_logits, dim=1)  # [batch*seq_len, num_routed_experts]
    
    # Step 2: Compute importance weights: importance = max(probs, dim=1)[0]
    # importance: [batch*seq_len]
    # Importance of each token = max probability it assigns to any expert
    # High importance = token is confident about expert selection
    # Low importance = token is uncertain about expert selection
    importance = torch.max(probs, dim=1)[0]  # [batch*seq_len]
    
    # Step 3: Compute expert assignment rates: expert_fraction = mean(probs, dim=0)
    # expert_fraction: [num_routed_experts]
    # Fraction of total probability mass going to each expert on average
    # If balanced, all experts should have similar fractions (close to 1/num_experts)
    expert_fraction = probs.mean(dim=0)  # [num_routed_experts]
    
    # Step 4: Compute load balance loss
    # Option 1: Without importance weighting (simpler)
    # Penalize deviation from uniform distribution
    # Ideal: each expert gets 1/num_experts of the probability mass
    ideal_fraction = 1.0 / num_experts
    deviation = expert_fraction - ideal_fraction  # [num_experts]
    # L2 penalty on deviation encourages uniform distribution
    loss = (deviation ** 2).sum()  # Scalar
    
    # Alternative: With importance weighting (DeepSeek-style)
    # Weight the load balancing by token importance
    # This makes high-confidence tokens contribute more to load balancing
    # importance_weighted_probs = probs * importance.unsqueeze(1)  # [batch*seq_len, num_experts]
    # importance_weighted_expert_fraction = importance_weighted_probs.mean(dim=0)  # [num_experts]
    # importance_weighted_deviation = importance_weighted_expert_fraction - ideal_fraction
    # loss = (importance_weighted_deviation ** 2).sum()
    
    # Step 5: Return loss * num_routed_experts for scaling
    # This scales the loss to be comparable across different numbers of experts
    scaled_loss = loss * num_experts
    
    return scaled_loss


def z_loss(gate_logits: torch.Tensor, z_loss_weight: float = 1.0, target_z: float = 1.0) -> torch.Tensor:
    """Z-loss: log-sum-exp penalty to encourage balanced expert routing.
    
    Z-loss measures the "spread" of routing logits for each token. By penalizing
    extreme values, it encourages moderate routing confidence, preventing both
    overconfident (single expert dominates) and underconfident (mass spread too uniformly)
    routing decisions.
    
    Why Z-loss?
    - High z values mean the model has many competitive routing options
    - Low z values mean routing is dominated by one expert
    - By penalizing z^2, we encourage moderate spread, preventing both:
      a) Overconfident routing (all mass on one expert)
      b) Underconfident routing (mass spread too uniformly)
    - This improves load balancing naturally through the routing dynamics
    
    Args:
        gate_logits: [batch*seq_len, num_routed_experts] tensor of router logits
            Each row represents a token's scores for all experts
        z_loss_weight: Weight for the Z-loss (default 1.0)
        target_z: Target Z value for auxiliary loss (default 1.0)
            If None, only uses mean(z^2) penalty
    
    Returns:
        Scalar loss tensor - penalty for extreme routing confidence
        Lower values indicate more balanced routing with moderate confidence
    
    Implementation Details:
    1. For each token (row in gate_logits):
       - Compute log-sum-exp: z_i = log(sum(exp(logits_i))) for token i
       - This measures the "spread" of logits for that token
    2. Use torch.logsumexp for numerical stability
    3. Return loss = mean(z^2) to penalize extreme values
    4. Optional: Add auxiliary loss = (z.mean() - target_z)^2
    """
    if gate_logits is None or gate_logits.numel() == 0:
        return torch.tensor(0.0, device=gate_logits.device if gate_logits is not None else 'cpu')
    
    # Step 1: For each token (row in gate_logits), compute log-sum-exp
    # z_i = log(sum(exp(logits_i))) for token i
    # This measures the "spread" of logits for that token
    # Higher z = more competitive routing options (logits spread out)
    # Lower z = one expert dominates (logits concentrated)
    
    # Step 2: Implementation using torch.logsumexp for numerical stability
    # logsumexp computes log(sum(exp(x))) numerically stably
    z = torch.logsumexp(gate_logits, dim=1)  # [batch*seq_len]
    # z[i] = log(sum(exp(gate_logits[i, :]))) for each token i
    
    # Step 3: Return loss = mean(z^2) to penalize extreme values
    # Penalizing z^2 encourages moderate spread:
    # - Very high z (many competitive options) -> penalized
    # - Very low z (one expert dominates) -> penalized
    # - Moderate z -> lower penalty
    z_squared_loss = (z ** 2).mean()  # Scalar
    
    # Step 4 (Optional): Add auxiliary loss = z_loss_weight * (z.mean() - target_z)^2
    # This encourages the mean Z value to be close to target_z
    # target_z = 1.0 is a reasonable default (moderate spread)
    if target_z is not None:
        mean_z = z.mean()  # Scalar - average Z across all tokens
        auxiliary_loss = (mean_z - target_z) ** 2  # Penalty for deviation from target
        # Combine both losses
        total_loss = z_squared_loss + z_loss_weight * auxiliary_loss
    else:
        total_loss = z_squared_loss
    
    return total_loss


def compute_capacity_loss_and_overflow_expert_choice(
    token_indices: torch.Tensor,  # [num_experts, top_k]
    batch_size: int,
    seq_len: int,
    num_experts: int,
    capacity_factor: float,
    device: torch.device
) -> Tuple[Optional[torch.Tensor], float, torch.Tensor, float, torch.Tensor]:
    """Compute capacity loss and handle token overflow for Expert Choice routing.
    
    Ensures sparsity by enforcing that experts don't get overloaded, dropping excess
    tokens that get routed to shared experts as fallback.
    
    Args:
        token_indices: [num_experts, top_k] tensor of token indices selected by each expert
            Values are indices into flattened batch*seq_len dimension
        batch_size: Batch size
        seq_len: Sequence length
        num_experts: Number of routed experts
        capacity_factor: Capacity multiplier (default 1.5)
            Higher values allow more tokens per expert, lower values enforce stricter sparsity
        device: Device for tensors
    
    Returns:
        capacity_mask: [num_experts, top_k] bool mask (True = keep token, False = drop/overflow)
            None if no capacity constraints needed
        capacity_loss: float scalar - penalty for exceeding capacity
            L2 penalty on deviation from average load
        expert_load: [num_experts] tensor - number of tokens assigned to each expert
            Counts non-unique tokens (same token can be selected multiple times)
        dropped_token_fraction: float - fraction of tokens dropped due to overflow
            Range: 0.0 (no drops) to 1.0 (all dropped)
        expert_utilization_rate: [num_experts] tensor - utilization percentage per expert
            Values are expert_load / max_tokens_per_expert for each expert
            Range: 0.0 to potentially >1.0 (if exceeding capacity)
    
    Implementation Details:
    1. Calculate max_tokens_per_expert = capacity_factor * (batch_size * seq_len) / num_experts
    2. For each expert, count how many tokens it selected (non-unique from token_indices)
    3. Create capacity_mask where True if within capacity, False if overflow
    4. Compute capacity_loss as L2 penalty: loss = sum((expert_load - avg_load)^2)
    5. Calculate dropped_token_fraction = (overflow tokens) / (total tokens)
    6. Calculate expert_utilization_rate = expert_load / max_tokens_per_expert per expert
    """
    num_experts_actual, top_k = token_indices.shape
    
    # Handle edge case: no experts or no tokens
    if num_experts == 0 or num_experts_actual == 0:
        empty_mask = torch.empty((0, top_k), dtype=torch.bool, device=device)
        empty_load = torch.zeros(num_experts, dtype=torch.long, device=device)
        empty_util = torch.zeros(num_experts, dtype=torch.float, device=device)
        return None, 0.0, empty_load, 0.0, empty_util
    
    # Step 1: Calculate max_tokens_per_expert = capacity_factor * (batch_size * seq_len) / num_experts
    total_tokens = batch_size * seq_len
    max_tokens_per_expert = capacity_factor * (total_tokens / num_experts)
    
    # Step 2: For each expert, count how many tokens it selected (non-unique from token_indices)
    # Count all token selections, including duplicates (same token selected multiple times)
    expert_load = torch.zeros(num_experts, dtype=torch.long, device=device)
    for expert_idx in range(num_experts_actual):
        if expert_idx < num_experts:
            # Count all token selections (non-unique count)
            selected_tokens = token_indices[expert_idx]  # [top_k]
            expert_load[expert_idx] = len(selected_tokens)  # Count all, including duplicates
    
    # Convert to float for calculations
    expert_load_float = expert_load.float()  # [num_experts]
    
    # Step 3: Create capacity_mask where True if within capacity, False if overflow
    capacity_mask = torch.ones(num_experts_actual, top_k, dtype=torch.bool, device=device)
    
    for expert_idx in range(num_experts_actual):
        if expert_idx < num_experts:
            if expert_load[expert_idx] > max_tokens_per_expert:
                # Expert exceeds capacity - mask out excess tokens
                # Keep first max_tokens_per_expert tokens, drop the rest
                excess_count = expert_load[expert_idx] - int(max_tokens_per_expert)
                if excess_count > 0:
                    # Mask out the last excess_count token selections
                    # If excess_count >= top_k, mask all tokens
                    mask_start_idx = max(0, top_k - excess_count)
                    capacity_mask[expert_idx, mask_start_idx:] = False
    
    # Step 4: Compute capacity_loss as L2 penalty on deviation from average load
    # Formula: loss = sum((expert_load - avg_load)^2)
    # This encourages balanced load distribution across experts
    avg_load = expert_load_float.mean()  # Average tokens per expert
    load_deviation = expert_load_float - avg_load  # [num_experts]
    capacity_loss = (load_deviation ** 2).sum().item()  # L2 penalty
    # Normalize by number of experts for scale invariance
    capacity_loss = capacity_loss / (num_experts + 1e-10)
    
    # Step 5: Calculate dropped_token_fraction = (overflow tokens) / (total tokens)
    # Count tokens that are masked out (dropped due to overflow)
    total_token_selections = num_experts_actual * top_k
    dropped_selections = (~capacity_mask).sum().item()
    dropped_token_fraction = dropped_selections / total_token_selections if total_token_selections > 0 else 0.0
    
    # Step 6: Calculate expert_utilization_rate = expert_load / max_tokens_per_expert per expert
    # Returns tensor with utilization rate for each expert
    expert_utilization_rate = expert_load_float / (max_tokens_per_expert + 1e-10)  # [num_experts]
    # Clamp to reasonable range (0 to 2.0) for numerical stability
    expert_utilization_rate = torch.clamp(expert_utilization_rate, min=0.0, max=2.0)
    
    # Return all 5 values for logging and auxiliary loss computation
    return capacity_mask, capacity_loss, expert_load, dropped_token_fraction, expert_utilization_rate


def batch_expert_forward_expert_choice(
    expert_modules: nn.ModuleList,
    inputs: torch.Tensor,  # [batch*seq_len, embedding_dim]
    token_indices: torch.Tensor,  # [num_experts, top_k] - flat indices into batch*seq_len
    expert_probs: torch.Tensor,  # [num_experts, top_k] - gating probabilities
    capacity_mask: Optional[torch.Tensor] = None  # [num_experts, top_k] bool
) -> torch.Tensor:
    """Vectorized Expert Choice forward pass using scatter_add for efficient GPU computation.
    
    In Expert Choice routing, each expert selects which tokens to process.
    This implementation uses torch.scatter_add_ for efficient in-place accumulation
    of expert outputs, enabling GPU acceleration.
    
    Args:
        expert_modules: ModuleList of expert networks
        inputs: [batch*seq_len, embedding_dim] input tensor (flattened batch and sequence)
        token_indices: [num_experts, top_k] tensor of token indices selected by each expert
            Values are indices into the flattened batch*seq_len dimension (0 to batch*seq_len-1)
        expert_probs: [num_experts, top_k] tensor of expert probabilities for selected tokens
        capacity_mask: Optional [num_experts, top_k] boolean mask (True = keep token, False = drop)
            If provided, only tokens where mask is True are processed
    
    Returns:
        outputs: [batch*seq_len, embedding_dim] - expert outputs aggregated back to original shape
            Each token position contains the weighted sum of outputs from all experts that selected it
    
    Implementation Details:
    1. Initialize outputs: [batch*seq_len, embedding_dim] with zeros
    2. For each expert:
       a. Get selected token indices for this expert from token_indices[expert_id]
       b. If capacity_mask exists, mask out overflow tokens
       c. Gather tokens: selected_tokens = inputs[token_indices]
       d. Process through expert: expert_output = expert(selected_tokens)
       e. Weight by probabilities: weighted_output = expert_probs * expert_output
       f. Scatter back to outputs using torch.scatter_add_ for efficient accumulation
    3. Return aggregated outputs
    
    Key Optimization: Uses torch.scatter_add_ for efficient in-place accumulation of expert outputs
    instead of looping through tokens. This enables GPU acceleration.
    """
    num_tokens, embedding_dim = inputs.shape
    num_experts, top_k = token_indices.shape
    
    # Step 1: Initialize outputs: [batch*seq_len, embedding_dim] with zeros
    outputs = torch.zeros(num_tokens, embedding_dim, device=inputs.device, dtype=inputs.dtype)
    
    # Step 2: For each expert, process selected tokens and scatter outputs back
    for expert_id, expert in enumerate(expert_modules):
        if expert_id >= num_experts:
            break
        
        # Step 2a: Get selected token indices for this expert
        token_idx = token_indices[expert_id]  # [top_k] indices
        
        # Step 2b: If capacity_mask exists, mask out overflow tokens
        if capacity_mask is not None:
            mask = capacity_mask[expert_id]  # [top_k] bool
            token_idx = token_idx[mask]  # [selected_k] indices (may be fewer than top_k)
            
            # Get corresponding probabilities for selected tokens
            expert_prob = expert_probs[expert_id][mask]  # [selected_k]
        else:
            expert_prob = expert_probs[expert_id]  # [top_k]
        
        # Skip if no tokens selected after masking
        if len(token_idx) == 0:
            continue
        
        # Step 2c: Gather tokens: selected_tokens = inputs[token_indices[expert_id]]
        # Ensure token indices are within valid range
        valid_mask = (token_idx >= 0) & (token_idx < num_tokens)
        if not valid_mask.all():
            # Filter out invalid indices
            token_idx = token_idx[valid_mask]
            expert_prob = expert_prob[valid_mask]
            if len(token_idx) == 0:
                continue
        
        selected_inputs = inputs[token_idx]  # [selected_k, embedding_dim]
        
        # Step 2d: Process through expert: expert_output = expert(selected_tokens)
        expert_output = expert(selected_inputs)  # [selected_k, embedding_dim]
        
        # Step 2e: Weight by probabilities: weighted_output = expert_probs * expert_output
        # Reshape probabilities to [selected_k, 1] for broadcasting
        expert_prob_expanded = expert_prob.unsqueeze(-1)  # [selected_k, 1]
        weighted_output = expert_output * expert_prob_expanded  # [selected_k, embedding_dim]
        
        # Step 2f: Scatter back to outputs tensor at original token indices
        # Use torch.scatter_add_ for efficient in-place accumulation
        # token_idx: [selected_k] -> expand to [selected_k, embedding_dim] for scatter
        token_idx_expanded = token_idx.unsqueeze(1).expand(-1, embedding_dim)  # [selected_k, embedding_dim]
        
        # Scatter add: accumulate weighted outputs at token positions
        # This efficiently handles cases where multiple experts select the same token
        outputs.scatter_add_(0, token_idx_expanded, weighted_output)
    
    # Return aggregated outputs [batch*seq_len, embedding_dim]
    # Each token position now contains the weighted sum of outputs from all experts that selected it
    return outputs


def batch_expert_forward(expert_modules: nn.ModuleList, inputs: torch.Tensor, 
                         expert_indices: torch.Tensor, gate_probs: torch.Tensor,
                         capacity_mask: torch.Tensor = None) -> torch.Tensor:
    """Vectorized expert forward pass using advanced indexing (Token Choice - legacy).
    
    Computes expert outputs in parallel and applies gating weights efficiently.
    This replaces nested loops with vectorized operations for better performance.
    
    NOTE: This is kept for backward compatibility. Expert Choice routing is preferred.
    
    Args:
        expert_modules: ModuleList of expert networks
        inputs: [batch, embedding_dim] input tensor
        expert_indices: [batch, top_k] tensor of expert indices to select
        gate_probs: [batch, top_k] tensor of gating probabilities (or hard assignments)
        capacity_mask: [batch, top_k] boolean mask for tokens within capacity (None = all allowed)
        
    Returns:
        [batch, embedding_dim] tensor of weighted expert outputs
    """
    batch_size, embedding_dim = inputs.shape
    top_k = expert_indices.shape[1]
    
    # Compute all expert outputs in parallel (vectorized)
    # Stack all expert outputs: [batch, num_experts, embedding_dim]
    all_expert_outputs = torch.stack([
        expert(inputs) for expert in expert_modules
    ], dim=1)  # [batch, num_experts, embedding_dim]
    
    # Use gather to select top-k expert outputs efficiently
    # expert_indices: [batch, top_k] -> expand for gather: [batch, top_k, 1]
    # Gather along expert dimension: [batch, top_k, embedding_dim]
    expert_indices_expanded = expert_indices.unsqueeze(-1).expand(-1, -1, embedding_dim)  # [batch, top_k, embedding_dim]
    selected_outputs = torch.gather(
        all_expert_outputs, 
        dim=1,  # Gather along expert dimension
        index=expert_indices_expanded
    )  # [batch, top_k, embedding_dim]
    
    # Apply capacity mask if provided (drop tokens that exceed capacity)
    if capacity_mask is not None:
        # Apply mask: set gate_probs to 0 for tokens exceeding capacity
        gate_probs = gate_probs * capacity_mask.float()  # [batch, top_k]
        # Renormalize to maintain probability distribution
        gate_probs_sum = gate_probs.sum(dim=1, keepdim=True)  # [batch, 1]
        gate_probs = gate_probs / (gate_probs_sum + 1e-10)  # Normalize, avoid division by zero
    
    # Apply gating probabilities: [batch, top_k, 1] * [batch, top_k, embedding_dim]
    gate_probs_expanded = gate_probs.unsqueeze(-1)  # [batch, top_k, 1]
    weighted_outputs = gate_probs_expanded * selected_outputs  # [batch, top_k, embedding_dim]
    
    # Sum over top_k dimension to get final output
    final_output = weighted_outputs.sum(dim=1)  # [batch, embedding_dim]
    
    return final_output


class SimpleMoEModel(nn.Module):
    """Simplified trainable MoE model with learnable parameters.
    
    Architecture:
    - Shared Experts (always activated): Process all tokens, provide baseline functionality
      - Typically 1-2 experts (num_shared_experts parameter)
      - Always active regardless of routing decisions
      - Stored in self.shared_experts ModuleList
      
    - Routed Experts (selected via Expert Choice routing): Specialized experts that select tokens
      - Typically 60+ experts for large-scale models (num_routed_experts parameter)
      - Each expert selects top_k tokens to process
      - Selected dynamically via Expert Choice routing
      - Stored in self.routed_experts ModuleList
      
    Routing Mechanism:
    - Expert Choice: Each routed expert selects top_k tokens to process
    - Capacity Control: Enforces sparsity via (batch_size * seq_len) / num_routed_experts
    - Fail-safe: Unprocessed tokens fall back to shared experts
    
    Backward Compatibility:
    - Supports legacy num_experts, num_text_experts, num_image_experts, num_multimodal_experts
    - If num_routed_experts is None, infers from legacy parameters
"""
    
    def __init__(
        self,
        vocab_size: int = 10007,
        embedding_dim: int = 128,
        num_experts: int = 2,
        num_text_experts: int = None,
        num_image_experts: int = None,
        num_multimodal_experts: int = None,
        num_shared_experts: int = 1,
        num_routed_experts: int = None,
        top_k: int = 2,
        noise_scale: float = 0.01,
        z_loss_weight: float = 0.001,
        capacity_factor: float = 1.5,
        residual_factor: float = 0.1,
        temperature_schedule: str = "constant",
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        temperature_steps: int = 1000,
    ):
        super().__init__()
        
        # Validate and set shared experts count (typically 1-2)
        if num_shared_experts < 0:
            raise ValueError(f"num_shared_experts must be >= 0, got {num_shared_experts}")
        self.num_shared_experts = num_shared_experts
        
        # Determine routed experts count (typically 60+)
        # Backward compatibility: if num_routed_experts is None, infer from legacy parameters
        if num_routed_experts is None:
            # Legacy mode: if only num_experts is provided, use it as routed experts
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
            
            # Subtract shared experts from total to get routed experts
            num_routed_experts = max(1, total_num_experts - num_shared_experts)
        
        # Validate routed experts count
        if num_routed_experts < 1:
            raise ValueError(f"num_routed_experts must be >= 1, got {num_routed_experts}")
        
        self.num_routed_experts = num_routed_experts
        self.num_experts = num_shared_experts + num_routed_experts  # Total for compatibility
        
        # Validate top_k parameter (controls how many tokens each expert selects)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        # Auto-adjust top_k if it exceeds num_routed_experts (for small expert counts)
        if top_k > num_routed_experts:
            print(f"⚠️  Warning: top_k ({top_k}) exceeds num_routed_experts ({num_routed_experts}). Adjusting top_k to {num_routed_experts}")
            top_k = num_routed_experts
        self.top_k = top_k
        self.noise_scale = noise_scale  # Scale of noise added to router logits during training
        self.z_loss_weight = z_loss_weight  # Weight for Z-loss auxiliary loss
        self.target_z = 1.0  # Target Z value for Z-loss (default 1.0 for moderate spread)
        self.capacity_factor = capacity_factor  # Capacity factor for expert load balancing
        self.residual_factor = residual_factor  # Residual connection strength (default 0.1)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size  # Store vocab_size for validation
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # ============================================================
        # Expert Architecture: Shared vs Routed Experts
        # ============================================================
        
        # Shared experts: always activated regardless of gating
        # These experts process all tokens and provide baseline functionality
        # Typically 1-2 experts that ensure all tokens are processed
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.ReLU(),
                nn.Linear(4 * embedding_dim, embedding_dim),
            ) for _ in range(self.num_shared_experts)
        ])
        
        # Routed experts: selected via Expert Choice routing (top-k tokens per expert)
        # These experts specialize on different patterns and are selected dynamically
        # Typically 60+ experts for large-scale models
        # Each expert selects top_k tokens to process via routing
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.ReLU(),
                nn.Linear(4 * embedding_dim, embedding_dim),
            ) for _ in range(self.num_routed_experts)
        ])
        
        # Router gate: only routes to routed experts (not shared experts)
        # Input: [batch, embedding_dim] -> Output: [batch, num_routed_experts]
        # Used for Expert Choice routing where each expert selects top_k tokens
        self.gate = nn.Linear(embedding_dim, self.num_routed_experts)
        
        # Temperature scheduling for router: enables exploration then exploitation
        # Higher temperature early = soft routing (exploration)
        # Lower temperature later = sparse routing (exploitation)
        self.temperature_schedule = temperature_schedule
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_steps = temperature_steps
        self.gate_temperature = temperature_start  # Initialize to start temperature
        
        # Joint fusion (combines expert outputs)
        # Single expert output dimension
        self.joint_fusion = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.joint_fusion_norm = nn.LayerNorm(embedding_dim)
        
        # Output normalization before projection (applied after combining experts)
        self.output_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout for regularization before residual connection
        self.residual_dropout = nn.Dropout(p=0.1)
        
        # Output decoder (standard 2-layer MLP)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, vocab_size),
        )
    
    def update_temperature(self, step: int) -> None:
        """Update router temperature based on schedule.
        
        Higher temperature early in training enables exploration (soft routing),
        then gradually cool down to enforce sparse, decisive routing decisions.
        
        Args:
            step: Current training step (global step counter)
        
        Schedules:
        - "constant": gate_temperature = temperature_start (no change)
        - "linear": Linear interpolation from start to end
        - "exponential": Exponential decay from start to end
        - "cosine": Cosine annealing from start to end
        """
        if self.temperature_schedule == "constant":
            # No change: keep temperature at start value
            self.gate_temperature = self.temperature_start
        
        elif self.temperature_schedule == "linear":
            # Linear interpolation: temperature_start -> temperature_end over temperature_steps
            progress = min(step / self.temperature_steps, 1.0)
            self.gate_temperature = self.temperature_start - (self.temperature_start - self.temperature_end) * progress
        
        elif self.temperature_schedule == "exponential":
            # Exponential decay: temperature_start * (decay_rate ^ step)
            # decay_rate chosen such that after temperature_steps, we reach temperature_end
            if self.temperature_start > 0 and self.temperature_end > 0:
                decay_rate = (self.temperature_end / self.temperature_start) ** (1.0 / self.temperature_steps)
                progress = min(step / self.temperature_steps, 1.0)
                current_step = int(progress * self.temperature_steps)
                self.gate_temperature = self.temperature_start * (decay_rate ** current_step)
            else:
                # Fallback to linear if invalid values
                progress = min(step / self.temperature_steps, 1.0)
                self.gate_temperature = self.temperature_start - (self.temperature_start - self.temperature_end) * progress
        
        elif self.temperature_schedule == "cosine":
            # Cosine annealing: smooth transition from start to end
            progress = min(step / self.temperature_steps, 1.0)
            # Cosine: 1 -> 0 over [0, pi], so we use 0.5 * (1 + cos(pi * progress))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            self.gate_temperature = self.temperature_end + (self.temperature_start - self.temperature_end) * cosine_factor
        
        else:
            # Invalid schedule: default to constant
            print(f"⚠️  Invalid temperature_schedule '{self.temperature_schedule}', using 'constant'")
            self.gate_temperature = self.temperature_start
        
        # Ensure temperature doesn't go below a minimum (avoid division by zero or negative)
        self.gate_temperature = max(self.gate_temperature, 1e-6)
    
    def add_gating_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """Add noise to router logits during training to improve expert exploration.
        
        Adds small Gumbel or Gaussian noise to router logits only when training.
        This prevents the model from always routing to the same experts early in training.
        
        Args:
            logits: Router logits tensor of shape [batch, num_routed_experts]
            
        Returns:
            Noisy logits of the same shape as input
        """
        if not self.training:
            return logits
        
        # Use Gumbel noise (standard for categorical distributions)
        # Gumbel noise is better for routing decisions as it's used in Gumbel-Softmax
        # Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
        uniform_noise = torch.rand_like(logits)
        uniform_noise = torch.clamp(uniform_noise, min=1e-7, max=1.0 - 1e-7)  # Avoid log(0)
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        
        # Scale the noise and add to logits
        noisy_logits = logits + self.noise_scale * gumbel_noise
        
        return noisy_logits
    
    def compute_routing_metrics(
        self,
        gate_logits: torch.Tensor,  # [batch*seq_len, num_routed_experts]
        token_indices: torch.Tensor,  # [num_routed_experts, top_k]
        expert_load: torch.Tensor,  # [num_routed_experts]
        total_tokens: int,
        max_tokens_per_expert: float
    ) -> Dict[str, torch.Tensor]:
        """Compute detailed routing metrics for monitoring routing health.
        
        Purpose: Enable monitoring of routing health during training to detect issues like
        expert collapse, load imbalance, or underutilization.
        
        Args:
            gate_logits: [batch*seq_len, num_routed_experts] router logits
            token_indices: [num_routed_experts, top_k] token indices selected by each expert
            expert_load: [num_routed_experts] number of tokens assigned to each expert
            total_tokens: Total number of tokens in the batch
            max_tokens_per_expert: Maximum capacity per expert
        
        Returns:
            Dictionary with routing metrics:
            - router_entropy: scalar - entropy of routing distribution
            - load_imbalance: scalar - standard deviation of expert loads / mean
            - top_expert_fraction: scalar - fraction of tokens going to top expert
            - expert_utilization: [num_routed_experts] - utilization per expert (0-1)
            - token_concentration: scalar - Herfindahl index of token distribution
        """
        num_routed_experts = self.num_routed_experts
        device = gate_logits.device
        
        # Metric 1: Router entropy
        # For each token (row), compute softmax to get routing probabilities
        # Then compute entropy: -sum(p * log(p)) for each token, then average
        router_probs = torch.softmax(gate_logits, dim=1)  # [batch*seq_len, num_routed_experts]
        # Avoid log(0) by clamping probabilities
        router_probs_clamped = torch.clamp(router_probs, min=1e-10)
        # Compute entropy per token: -sum(p * log(p)) across experts
        token_entropies = -(router_probs_clamped * torch.log(router_probs_clamped)).sum(dim=1)  # [batch*seq_len]
        router_entropy = token_entropies.mean()  # Scalar - average entropy across tokens
        
        # Metric 2: Load imbalance
        # std(expert_load) / mean(expert_load) - coefficient of variation
        expert_load_float = expert_load.float()  # [num_routed_experts]
        mean_load = expert_load_float.mean()
        if mean_load > 0:
            std_load = expert_load_float.std()
            load_imbalance = std_load / (mean_load + 1e-10)  # Scalar
        else:
            load_imbalance = torch.tensor(0.0, device=device)  # No imbalance if no load
        
        # Metric 3: Top expert fraction
        # max(expert_load) / total_tokens - fraction going to most loaded expert
        if total_tokens > 0:
            max_load = expert_load_float.max()
            top_expert_fraction = max_load / total_tokens  # Scalar
        else:
            top_expert_fraction = torch.tensor(0.0, device=device)
        
        # Metric 4: Expert utilization
        # expert_load / max_tokens_per_expert per expert (0-1)
        # Already computed in capacity computation, but compute here for consistency
        if max_tokens_per_expert > 0:
            expert_utilization = expert_load_float / (max_tokens_per_expert + 1e-10)  # [num_routed_experts]
            expert_utilization = torch.clamp(expert_utilization, min=0.0, max=1.0)  # Clamp to [0, 1]
        else:
            expert_utilization = torch.zeros(num_routed_experts, device=device)
        
        # Metric 5: Token concentration (Herfindahl index)
        # sum((token_count_per_expert / total_tokens)^2)
        # 1.0 = perfect concentration (all tokens to one expert)
        # 1/num_experts = perfect uniform distribution
        if total_tokens > 0:
            expert_fractions = expert_load_float / (total_tokens + 1e-10)  # [num_routed_experts]
            token_concentration = (expert_fractions ** 2).sum()  # Scalar - Herfindahl index
        else:
            token_concentration = torch.tensor(0.0, device=device)
        
        return {
            'router_entropy': router_entropy,
            'load_imbalance': load_imbalance,
            'top_expert_fraction': top_expert_fraction,
            'expert_utilization': expert_utilization,
            'token_concentration': token_concentration,
        }
    
    def forward(self, text_tokens: torch.Tensor, image_features: torch.Tensor = None, return_load_balance_loss: bool = False, return_gate_logits: bool = False):
        # Validate token indices before embedding
        if torch.any((text_tokens < 0) | (text_tokens >= self.vocab_size)):
            raise ValueError(f"Invalid token indices detected: min={text_tokens.min().item()}, max={text_tokens.max().item()}, vocab_size={self.vocab_size}")
        
        # Embed text tokens
        embedded = self.embedding(text_tokens)  # [batch, seq_len, embedding_dim]
        
        # Check for NaN in embedding output
        if torch.isnan(embedded).any():
            raise ValueError("NaN detected in embedding output — check input token indices")
        
        # Keep sequence-level representation (no pooling)
        # embedded: [batch, seq_len, embedding_dim]
        batch_size, seq_len, embedding_dim = embedded.shape
        text_sequence = embedded  # [batch, seq_len, embedding_dim]
        
        # Image input preparation (keep sequence dimension for consistency)
        if image_features is None:
            image_sequence = text_sequence  # Fallback to text features
        else:
            if len(image_features.shape) == 2:
                # [batch, embedding_dim] -> [batch, 1, embedding_dim]
                image_sequence = image_features.unsqueeze(1)
            elif len(image_features.shape) == 3:
                image_sequence = image_features  # [batch, seq_len, embedding_dim]
            else:
                # Flatten spatial dimensions if needed
                image_sequence = image_features.view(batch_size, -1, embedding_dim)
        
        # Flatten batch and seq_len for routing: [batch, seq_len, embedding_dim] -> [batch*seq_len, embedding_dim]
        text_flat = text_sequence.view(-1, embedding_dim)  # [batch*seq_len, embedding_dim]
        
        # Compute gate logits using shared gate (per token)
        gate_logits = self.gate(text_flat)  # [batch*seq_len, num_routed_experts]
        
        # Check for NaN in gate logits
        if torch.isnan(gate_logits).any():
            raise ValueError("NaN detected in gate_logits")
        
        # Add noise to router logits during training to improve expert exploration
        gate_logits = self.add_gating_noise(gate_logits)
        
        # Apply temperature scaling for better routing
        gate_logits = gate_logits / self.gate_temperature
        
        # Expert Choice routing: each expert selects top-k tokens to process
        # gate_logits: [batch*seq_len, num_routed_experts]
        # Returns token indices [num_experts, top_k] and expert probabilities [num_experts, top_k]
        token_indices, expert_probs = expert_choice_routing(gate_logits, k=self.top_k)
        # token_indices: [num_routed_experts, top_k] - each expert selects k flat indices (0 to batch*seq_len-1)
        # expert_probs: [num_routed_experts, top_k] - soft probabilities for gradient flow
        
        # Compute capacity constraints and handle overflow
        total_tokens = batch_size * seq_len
        capacity_mask, capacity_loss, expert_load, dropped_token_fraction, expert_utilization_rate = compute_capacity_loss_and_overflow_expert_choice(
            token_indices=token_indices,
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=self.num_routed_experts,
            capacity_factor=self.capacity_factor,
            device=text_sequence.device
        )
        
        # Store capacity metrics for later access
        self._capacity_metrics = {
            'capacity_loss': capacity_loss,  # Store as float scalar
            'dropped_token_fraction': dropped_token_fraction,
            'expert_utilization_rate': expert_utilization_rate,  # Tensor [num_experts]
            'expert_load': expert_load,  # Tensor [num_experts]
            'capacity_mask': capacity_mask,  # Tensor [num_experts, top_k] or None
        }
        
        # Compute detailed routing metrics for monitoring
        max_tokens_per_expert = self.capacity_factor * (total_tokens / self.num_routed_experts)
        routing_metrics = self.compute_routing_metrics(
            gate_logits=gate_logits,  # [batch*seq_len, num_routed_experts]
            token_indices=token_indices,  # [num_routed_experts, top_k]
            expert_load=expert_load,  # [num_routed_experts]
            total_tokens=total_tokens,
            max_tokens_per_expert=max_tokens_per_expert
        )
        # Store routing metrics for later access
        self._routing_metrics = routing_metrics
        
        # Compute shared expert outputs (always activated) - process each token
        # Process all tokens: [batch, seq_len, embedding_dim] -> [batch, seq_len, embedding_dim]
        if len(self.shared_experts) > 0:
            # Process each token through shared experts
            # Flatten for processing: [batch, seq_len, embedding_dim] -> [batch*seq_len, embedding_dim]
            text_flat_for_shared = text_sequence.view(-1, embedding_dim)  # [batch*seq_len, embedding_dim]
            
            # Compute all shared expert outputs in parallel
            shared_outputs_flat = torch.stack([
                shared_expert(text_flat_for_shared) for shared_expert in self.shared_experts
            ], dim=1)  # [batch*seq_len, num_shared_experts, embedding_dim]
            
            # Average across shared experts dimension
            shared_combined_flat = shared_outputs_flat.mean(dim=1)  # [batch*seq_len, embedding_dim]
            
            # Reshape back to sequence: [batch*seq_len, embedding_dim] -> [batch, seq_len, embedding_dim]
            shared_combined = shared_combined_flat.view(batch_size, seq_len, embedding_dim)
        else:
            shared_combined = text_sequence  # Fallback if no shared experts
        
        # Expert Choice routing: iterate over experts and gather assigned tokens
        # Process flattened tokens: [batch*seq_len, embedding_dim]
        routed_combined_flat = batch_expert_forward_expert_choice(
            expert_modules=self.routed_experts,
            inputs=text_flat,  # [batch*seq_len, embedding_dim]
            token_indices=token_indices,  # [num_experts, top_k] - indices into flattened batch*seq_len
            expert_probs=expert_probs,
            capacity_mask=capacity_mask  # Drop tokens exceeding capacity
        )  # [batch*seq_len, embedding_dim]
        
        # Reshape back to sequence: [batch*seq_len, embedding_dim] -> [batch, seq_len, embedding_dim]
        routed_combined = routed_combined_flat.view(batch_size, seq_len, embedding_dim)
        
        # Handle tokens that were not selected by any expert (fail-safe to shared experts)
        # Track which tokens were processed by routed experts (in flattened space)
        processed_tokens_flat = torch.zeros(batch_size * seq_len, dtype=torch.bool, device=text_sequence.device)
        for expert_idx in range(token_indices.shape[0]):
            selected_tokens = token_indices[expert_idx]  # [top_k] - indices into flattened batch*seq_len
            if capacity_mask is not None:
                selected_tokens = selected_tokens[capacity_mask[expert_idx]]
            processed_tokens_flat[selected_tokens] = True
        
        # Route unprocessed tokens to shared experts (if available)
        unprocessed_tokens_flat = ~processed_tokens_flat
        if unprocessed_tokens_flat.any() and len(self.shared_experts) > 0:
            # Reshape to sequence dimension for masking
            unprocessed_mask = unprocessed_tokens_flat.view(batch_size, seq_len, 1).float()  # [batch, seq_len, 1]
            # Blend: use shared experts for unprocessed tokens, routed for others
            routed_combined = (1.0 - unprocessed_mask) * routed_combined + unprocessed_mask * shared_combined
        
        # Combine shared and routed expert outputs
        combined = shared_combined + routed_combined  # [batch, seq_len, embedding_dim]
        
        # Joint fusion with normalization and residual connection
        # Process each token: [batch, seq_len, embedding_dim] -> [batch, seq_len, embedding_dim]
        # Flatten for processing
        combined_flat = combined.view(-1, embedding_dim)  # [batch*seq_len, embedding_dim]
        fused_output_flat = self.joint_fusion(combined_flat)  # [batch*seq_len, embedding_dim]
        # Apply normalization and residual: x = x + dropout(norm(ffn(x)))
        fused_output_flat = self.joint_fusion_norm(fused_output_flat)
        fused_output_flat = nn.functional.dropout(fused_output_flat, p=0.1, training=self.training)
        # Add residual connection
        fused_output_flat = combined_flat + fused_output_flat  # Residual connection
        
        # Reshape back to sequence
        fused_output = fused_output_flat.view(batch_size, seq_len, embedding_dim)  # [batch, seq_len, embedding_dim]
        
        # Apply normalization and residual connection before output projection
        # Formula: output = output_proj(norm(combined + residual_factor * dropout(embedded)))
        # This helps with gradient flow and regularization
        embedded_dropped = self.residual_dropout(text_sequence)  # Apply dropout before residual: [batch, seq_len, embedding_dim]
        normalized_input = self.output_norm(
            fused_output + self.residual_factor * embedded_dropped
        )  # [batch, seq_len, embedding_dim]
        
        # Flatten for decoder: [batch, seq_len, embedding_dim] -> [batch*seq_len, embedding_dim]
        normalized_input_flat = normalized_input.view(-1, embedding_dim)
        
        # Decode to vocabulary: [batch*seq_len, embedding_dim] -> [batch*seq_len, vocab_size]
        output_flat = self.decoder(normalized_input_flat)  # [batch*seq_len, vocab_size]
        
        # Reshape back to sequence: [batch*seq_len, vocab_size] -> [batch, seq_len, vocab_size]
        output = output_flat.view(batch_size, seq_len, -1)  # [batch, seq_len, vocab_size]
        
        # For compatibility with existing code, return [batch, vocab_size] by taking last token
        # This maintains backward compatibility while enabling sequence-level processing
        output = output[:, -1, :]  # [batch, vocab_size] - use last token for prediction
        
        # Debug: Check output shape (should be [batch, vocab_size])
        # Only print once to reduce noise
        if len(output.shape) != 2:
            if not hasattr(self, '_debug_shape_warned') or not self._debug_shape_warned:
                print(f"⚠️  Model decoder output shape is unexpected: {output.shape}")
                print(f"   fused_output shape: {fused_output.shape}")
                print(f"   text_sequence shape: {text_sequence.shape}")
                print(f"   text_tokens shape: {text_tokens.shape}")
                print(f"   This should not happen - model should output [batch, vocab_size]")
                self._debug_shape_warned = True
        
        if return_load_balance_loss or return_gate_logits:
            if return_gate_logits:
                # Compute auxiliary losses for load balancing
                # Use gate_logits before noise and temperature scaling for loss computation
                # Flatten for gate computation: [batch, seq_len, embedding_dim] -> [batch*seq_len, embedding_dim]
                text_flat_for_loss = text_sequence.view(-1, embedding_dim)
                gate_logits_for_loss = self.gate(text_flat_for_loss)  # [batch*seq_len, num_routed_experts]
                
                # Add error handling for loss computation
                try:
                    # Compute load balance loss (entropy-based)
                    load_bal_loss = load_balance_loss(gate_logits_for_loss)
                    # Ensure it's a tensor on the correct device
                    if not isinstance(load_bal_loss, torch.Tensor):
                        load_bal_loss = torch.tensor(float(load_bal_loss), device=output.device)
                    else:
                        load_bal_loss = load_bal_loss.to(output.device)
                    
                    # Compute Z-loss (log-sum-exp to encourage balanced routing)
                    # Pass z_loss_weight and target_z from model config
                    target_z = getattr(self, 'target_z', 1.0)  # Default target_z = 1.0
                    z_loss_val = z_loss(gate_logits_for_loss, z_loss_weight=1.0, target_z=target_z)
                    # Ensure it's a tensor on the correct device
                    if not isinstance(z_loss_val, torch.Tensor):
                        z_loss_val = torch.tensor(float(z_loss_val), device=output.device)
                    else:
                        z_loss_val = z_loss_val.to(output.device)
                    
                    # Get capacity loss (computed during forward pass)
                    # capacity_loss is returned as float scalar from compute_capacity_loss_and_overflow_expert_choice
                    cap_loss = self._capacity_metrics.get('capacity_loss', 0.0)
                    if isinstance(cap_loss, torch.Tensor):
                        cap_loss = cap_loss.item()
                    # Convert to tensor for loss computation on correct device
                    cap_loss_tensor = torch.tensor(float(cap_loss), device=output.device, dtype=output.dtype)
                    
                    # Combined auxiliary loss: weighted sum of all losses
                    aux_loss = load_bal_loss + self.z_loss_weight * z_loss_val + 0.1 * cap_loss_tensor
                    
                    # Ensure routing_metrics exists and convert all values to tensors on correct device
                    if not hasattr(self, '_routing_metrics') or self._routing_metrics is None:
                        # Create empty routing metrics if not computed
                        routing_metrics = {
                            'router_entropy': torch.tensor(0.0, device=output.device),
                            'load_imbalance': torch.tensor(0.0, device=output.device),
                            'top_expert_fraction': torch.tensor(0.0, device=output.device),
                            'expert_utilization': torch.zeros(self.num_routed_experts, device=output.device),
                            'token_concentration': torch.tensor(0.0, device=output.device),
                        }
                    else:
                        # Convert all routing metrics to tensors on correct device
                        routing_metrics = {}
                        for key, value in self._routing_metrics.items():
                            if isinstance(value, torch.Tensor):
                                routing_metrics[key] = value.to(output.device)
                            else:
                                routing_metrics[key] = torch.tensor(float(value), device=output.device)
                    
                    # Ensure gate_logits is on the correct device
                    gate_logits_for_return = gate_logits.to(output.device)
                    
                    # Return gate_logits, auxiliary loss components, and routing metrics
                    # Include routing metrics if requested (return_load_balance_loss or return_gate_logits)
                    return output, (gate_logits_for_return, load_bal_loss, z_loss_val, cap_loss_tensor, aux_loss, routing_metrics)
                    
                except Exception as e:
                    print(f"⚠️  Error computing auxiliary losses: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return output without losses on error
                    return output
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
        
        # Initialize global step counter for temperature scheduling
        global_step = 0
        
        for epoch in range(start_epoch, epochs):
            print(f"\n{'='*60}")
            print(f"📚 Epoch {epoch + 1}/{epochs}" + (f" (resumed from {start_epoch})" if epoch == start_epoch and start_epoch > 0 else ""))
            print(f"{'='*60}")
            
            model.train()
            total_loss = 0.0
            total_aux_loss = 0.0  # Track combined auxiliary loss
            total_load_bal_loss = 0.0  # Track load balance loss component
            total_z_loss = 0.0  # Track Z-loss component
            batch_count = 0
            bert_scores = []
            diagnostics_run = False  # Flag to ensure diagnostics run once per first epoch
            expert_usage = torch.zeros(model.num_experts, device=device)  # Track expert usage
            
            for batch_idx, batch in enumerate(train_dataloader):
                batch_count += 1
                
                # Update router temperature based on schedule (before forward pass)
                # Access model through model_engine if using DeepSpeed, otherwise use model directly
                if use_deepspeed and model_engine is not None:
                    model_engine.module.update_temperature(global_step)
                elif model is not None:
                    model.update_temperature(global_step)
                
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
                # Unpack: (gate_logits, load_bal_loss, z_loss_val, cap_loss, aux_loss, routing_metrics)
                if len(gate_logits_tuple) == 6:
                    # New format with routing metrics
                    gate_logits, load_bal_loss, z_loss_val, cap_loss, aux_loss, routing_metrics = gate_logits_tuple
                elif len(gate_logits_tuple) == 5:
                    # Backward compatibility: format without routing metrics
                    gate_logits, load_bal_loss, z_loss_val, cap_loss, aux_loss = gate_logits_tuple
                    routing_metrics = {}  # Empty metrics for backward compatibility
                elif len(gate_logits_tuple) == 4:
                    # Backward compatibility: old format without capacity loss
                    gate_logits, load_bal_loss, z_loss_val, aux_loss = gate_logits_tuple
                    cap_loss = torch.tensor(0.0, device=output.device)
                    routing_metrics = {}  # Empty metrics for backward compatibility
                else:
                    # Legacy format: old format
                    gate_logits, _, _ = gate_logits_tuple
                    load_bal_loss = load_balance_loss(gate_logits)
                    # Pass z_loss_weight and target_z from model config
                    target_z = getattr(model, 'target_z', 1.0)  # Default target_z = 1.0
                    z_loss_val = z_loss(gate_logits, z_loss_weight=1.0, target_z=target_z)
                    cap_loss = torch.tensor(0.0, device=output.device)
                    aux_loss = load_bal_loss + model.z_loss_weight * z_loss_val
                    routing_metrics = {}  # Empty metrics for backward compatibility
                
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
                
                # Auxiliary losses are already computed in forward() and returned
                # aux_loss = load_bal_loss + z_loss_weight * z_loss_val
                # where load_bal_loss is entropy-based and z_loss_val is log-sum-exp
                
                # Track expert usage (which experts were selected in top-k)
                with torch.no_grad():
                    # Get top-k indices from gate logits (with temperature)
                    gate_logits_scaled = gate_logits / model_engine.gate_temperature if hasattr(model_engine, 'gate_temperature') else gate_logits
                    _, topk_idx = top_k_gating(gate_logits_scaled, k=model_engine.top_k if hasattr(model_engine, 'top_k') else 2)
                    # Count unique expert indices used in this batch
                    unique_experts = torch.unique(topk_idx)
                    expert_usage[unique_experts] += 1
                
                # Combine main loss with auxiliary loss (already weighted in forward())
                # Weight the auxiliary loss to not dominate (typically 0.01-0.1)
                aux_loss_weight = 0.01
                loss = main_loss + aux_loss_weight * aux_loss
                
                # Accumulate auxiliary loss components
                total_aux_loss += aux_loss.item()
                total_load_bal_loss += load_bal_loss.item()
                total_z_loss += z_loss_val.item()
                total_cap_loss += cap_loss.item() if isinstance(cap_loss, torch.Tensor) else cap_loss
                
                # Track capacity metrics if available
                if hasattr(model_engine, '_capacity_metrics'):
                    total_dropped_fraction += model_engine._capacity_metrics.get('dropped_token_fraction', 0.0)
                    util_rate = model_engine._capacity_metrics.get('expert_utilization_rate', None)
                    if util_rate is not None:
                        # expert_utilization_rate is now a tensor [num_experts], compute mean
                        if isinstance(util_rate, torch.Tensor):
                            total_expert_utilization += util_rate.mean().item()
                        else:
                            total_expert_utilization += float(util_rate)
                
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
                    print(f"     Load balance loss: {load_bal_loss.item():.4f}")
                    z_loss_w = model_engine.z_loss_weight if hasattr(model_engine, 'z_loss_weight') else (model.z_loss_weight if hasattr(model, 'z_loss_weight') else 0.001)
                    print(f"     Z-loss: {z_loss_val.item():.4f} (weight: {z_loss_w})")
                    cap_loss_val = cap_loss.item() if isinstance(cap_loss, torch.Tensor) else cap_loss
                    print(f"     Capacity loss: {cap_loss_val:.4f} (weight: 0.1)")
                    if hasattr(model_engine, '_capacity_metrics'):
                        dropped = model_engine._capacity_metrics.get('dropped_token_fraction', 0.0) * 100
                        util_rate = model_engine._capacity_metrics.get('expert_utilization_rate', None)
                        if util_rate is not None:
                            # expert_utilization_rate is now a tensor [num_experts], compute mean
                            if isinstance(util_rate, torch.Tensor):
                                util = util_rate.mean().item() * 100
                            else:
                                util = float(util_rate) * 100
                        else:
                            util = 0.0
                        print(f"     Capacity: Dropped {dropped:.2f}%, Utilization {util:.2f}%")
                    print(f"     Combined aux loss: {aux_loss.item():.4f} (weighted: {aux_loss_weight * aux_loss.item():.6f})")
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
                
                # Increment global step counter for temperature scheduling
                global_step += 1
                
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
            avg_load_bal_loss = total_load_bal_loss / batch_count if batch_count > 0 else 0.0
            avg_z_loss = total_z_loss / batch_count if batch_count > 0 else 0.0
            avg_cap_loss = total_cap_loss / batch_count if batch_count > 0 else 0.0
            avg_dropped_fraction = total_dropped_fraction / batch_count if batch_count > 0 else 0.0
            avg_expert_utilization = total_expert_utilization / batch_count if batch_count > 0 else 0.0
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
            print(f"🔍 Epoch {epoch + 1}: Aux loss = {avg_aux_loss:.4f} (LoadBal: {avg_load_bal_loss:.4f}, Z-loss: {avg_z_loss:.4f}, Cap: {avg_cap_loss:.4f})")
            print(f"   Expert capacity: Dropped {avg_dropped_fraction*100:.2f}%, Utilization {avg_expert_utilization*100:.2f}%, Active experts = {num_active_experts}/{model.num_experts}")
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
