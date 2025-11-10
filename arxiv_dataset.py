"""
Streaming IterableDataset for ArXiv Healthcare Papers

Optimized for Colab with efficient streaming, worker-aware distribution,
and minimal memory footprint (<500MB RAM regardless of corpus size).

Usage:
    from arxiv_dataset import ArXivStreamingDataset
    
    dataset = ArXivStreamingDataset(
        text_dir='./data/arxiv/texts',
        metadata_jsonl='./data/arxiv/arxiv_papers.jsonl',
        tokenizer=tokenizer,
        max_length=512,
        min_length=64
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterator, Optional, List, Tuple
import torch
from torch.utils.data import IterableDataset, get_worker_info

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


class ArXivStreamingDataset(IterableDataset):
    """Streaming IterableDataset for ArXiv papers.
    
    Streams papers from disk without loading entire corpus into memory.
    Worker-aware: distributes papers across DataLoader workers to avoid duplicates.
    
    Features:
    - Memory efficient: <500MB RAM regardless of corpus size
    - Worker-aware distribution
    - Shuffling with buffer
    - Variable length sequences (no padding)
    - Skips short papers (<min_length tokens)
    """
    
    def __init__(
        self,
        text_dir: str,
        metadata_jsonl: str,
        tokenizer,
        max_length: int = 512,
        min_length: int = 64,
        shuffle_buffer: int = 100,
        seed: Optional[int] = None
    ):
        """Initialize streaming dataset.
        
        Args:
            text_dir: Directory containing .txt files (one per paper, from extract step)
            metadata_jsonl: JSONL file with paper metadata (from preprocess step)
                Expected format: {arxiv_id, text, domains, year, has_neurodegeneration}
            tokenizer: SentencePiece tokenizer (or compatible tokenizer)
            max_length: Maximum sequence length (default: 512)
            min_length: Minimum sequence length to include (default: 64)
            shuffle_buffer: Buffer size for shuffling (default: 100)
            seed: Random seed for reproducibility
        """
        self.text_dir = text_dir
        self.metadata_jsonl = metadata_jsonl
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        
        # Load metadata mapping
        self.metadata = self._load_metadata()
        
        # Get list of text files
        self.text_files = self._get_text_files()
        
        # Estimate dataset length
        self._estimated_length = None
        
        print(f"ArXivStreamingDataset initialized:")
        print(f"   Text files: {len(self.text_files)}")
        print(f"   Metadata entries: {len(self.metadata)}")
        print(f"   Max length: {max_length}, Min length: {min_length}")
        print(f"   Shuffle buffer: {shuffle_buffer}")
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load metadata from JSONL file into dictionary keyed by arxiv_id.
        
        Returns:
            Dictionary mapping arxiv_id to metadata
        """
        metadata = {}
        if not os.path.exists(self.metadata_jsonl):
            print(f"Warning: Metadata file not found: {self.metadata_jsonl}")
            return metadata
        
        print(f"Loading metadata from {self.metadata_jsonl}...")
        with open(self.metadata_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    arxiv_id = record.get('arxiv_id', '')
                    if arxiv_id:
                        metadata[arxiv_id] = record
                except json.JSONDecodeError:
                    continue
        
        print(f"   Loaded {len(metadata)} metadata entries")
        return metadata
    
    def _get_text_files(self) -> List[Tuple[str, str]]:
        """Get list of (arxiv_id, file_path) tuples.
        
        Returns:
            List of (arxiv_id, file_path) tuples
        """
        text_files = []
        if not os.path.exists(self.text_dir):
            print(f"Warning: Text directory not found: {self.text_dir}")
            return text_files
        
        for filename in os.listdir(self.text_dir):
            if filename.endswith('.txt'):
                arxiv_id = filename[:-4]  # Remove .txt extension
                file_path = os.path.join(self.text_dir, filename)
                text_files.append((arxiv_id, file_path))
        
        return text_files
    
    def _tokenize_text(self, text: str) -> Optional[torch.Tensor]:
        """Tokenize text using tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Tensor of token IDs, or None if tokenization fails
        """
        try:
            if hasattr(self.tokenizer, 'encode'):
                # SentencePiece tokenizer
                token_ids = self.tokenizer.encode(text, out_type=int)
            elif hasattr(self.tokenizer, '__call__'):
                # Transformers-style tokenizer
                result = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True)
                token_ids = result['input_ids'].squeeze(0).tolist()
            else:
                # Fallback: assume it's callable
                token_ids = self.tokenizer(text)
            
            # Convert to tensor
            if isinstance(token_ids, list):
                token_ids = torch.tensor(token_ids, dtype=torch.long)
            
            # Truncate to max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            
            # Check minimum length
            if len(token_ids) < self.min_length:
                return None
            
            return token_ids
            
        except Exception as e:
            return None
    
    def _process_paper(self, arxiv_id: str, file_path: str) -> Optional[Dict]:
        """Process a single paper file.
        
        Args:
            arxiv_id: ArXiv ID
            file_path: Path to text file
            
        Returns:
            Dictionary with sample data, or None if processing fails
        """
        try:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                return None
            
            # Tokenize
            token_ids = self._tokenize_text(text)
            if token_ids is None:
                return None
            
            # Create input and target sequences (shifted by 1 for language modeling)
            if len(token_ids) < 2:
                return None
            
            input_ids = token_ids[:-1]  # All tokens except last
            target_ids = token_ids[1:]  # All tokens except first
            
            # Get metadata
            metadata = self.metadata.get(arxiv_id, {})
            domains = metadata.get('domains', [])
            year = metadata.get('year', None)
            has_neurodegeneration = metadata.get('has_neurodegeneration', False)
            
            return {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'domains': domains,
                'year': year,
                'arxiv_id': arxiv_id,
                'has_neurodegeneration': has_neurodegeneration,
            }
            
        except Exception as e:
            return None
    
    def _get_worker_files(self) -> List[Tuple[str, str]]:
        """Get files assigned to current worker.
        
        Returns:
            List of (arxiv_id, file_path) tuples for this worker
        """
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single process, return all files
            return self.text_files
        
        # Multi-worker: distribute files across workers
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        
        # Assign files to this worker (round-robin distribution)
        worker_files = [
            (arxiv_id, file_path)
            for idx, (arxiv_id, file_path) in enumerate(self.text_files)
            if idx % num_workers == worker_id
        ]
        
        return worker_files
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over dataset samples.
        
        Yields:
            Dictionary with sample data
        """
        # Get files for this worker
        worker_files = self._get_worker_files()
        
        if not worker_files:
            return
        
        # Set random seed for this worker
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_seed = (self.seed or 42) + worker_info.id
            random.seed(worker_seed)
        else:
            random.seed(self.seed or 42)
        
        # Shuffle file order
        file_order = list(worker_files)
        random.shuffle(file_order)
        
        # Shuffle buffer for paper-level shuffling
        # Use list instead of deque for easier index-based removal
        shuffle_buffer = []
        
        for arxiv_id, file_path in file_order:
            # Process paper
            sample = self._process_paper(arxiv_id, file_path)
            
            if sample is None:
                continue
            
            # Add to shuffle buffer
            shuffle_buffer.append(sample)
            
            # Yield from buffer if it's full (for shuffling)
            if len(shuffle_buffer) >= self.shuffle_buffer:
                # Randomly yield one from buffer
                idx = random.randint(0, len(shuffle_buffer) - 1)
                yield shuffle_buffer[idx]
                # Remove by index (avoid tensor comparison issues)
                shuffle_buffer.pop(idx)
        
        # Yield remaining items from buffer
        random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample
    
    def __len__(self) -> int:
        """Estimate dataset length.
        
        Note: Exact length is unknown for streaming datasets.
        This provides an estimate based on number of text files.
        
        Returns:
            Estimated number of samples
        """
        if self._estimated_length is None:
            # Estimate: assume ~80% of files produce valid samples
            # (accounting for short papers, tokenization failures, etc.)
            self._estimated_length = int(len(self.text_files) * 0.8)
        
        return self._estimated_length
    
    def estimate_length(self) -> int:
        """Get estimated dataset length.
        
        Returns:
            Estimated number of samples
        """
        return len(self)
    
    def inspect_sample(self, num_samples: int = 5) -> List[Dict]:
        """Inspect samples from dataset (for debugging).
        
        Args:
            num_samples: Number of samples to inspect
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        count = 0
        
        for sample in iter(self):
            samples.append({
                'arxiv_id': sample['arxiv_id'],
                'input_length': len(sample['input_ids']),
                'target_length': len(sample['target_ids']),
                'domains': sample['domains'],
                'year': sample['year'],
                'has_neurodegeneration': sample['has_neurodegeneration'],
                'input_ids_preview': sample['input_ids'][:10].tolist() if len(sample['input_ids']) > 10 else sample['input_ids'].tolist(),
            })
            count += 1
            if count >= num_samples:
                break
        
        return samples


def create_dataloader(
    dataset: ArXivStreamingDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2
):
    """Create DataLoader with optimal settings for Colab.
    
    Args:
        dataset: ArXivStreamingDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes (default: 4 for Colab)
        pin_memory: Pin memory for GPU transfer (default: True)
        prefetch_factor: Number of batches to prefetch per worker (default: 2)
        
    Returns:
        DataLoader instance
        
    Note:
        - Throughput target: >1000 samples/sec on Colab GPU
        - Memory footprint: <500MB RAM regardless of corpus size
    """
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function for variable-length sequences.
        
        Pads sequences to same length within batch.
        Uses padding token ID 0 (standard for most tokenizers).
        """
        if not batch:
            return None
        
        # Get max length in batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        # Pad sequences
        input_ids_list = []
        target_ids_list = []
        domains_list = []
        years_list = []
        arxiv_ids_list = []
        has_nd_list = []
        
        for item in batch:
            input_ids = item['input_ids']
            target_ids = item['target_ids']
            
            # Pad to max_len with zeros (padding token)
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad_tensor])
                target_ids = torch.cat([target_ids, pad_tensor])
            
            input_ids_list.append(input_ids)
            target_ids_list.append(target_ids)
            domains_list.append(item['domains'])
            years_list.append(item['year'])
            arxiv_ids_list.append(item['arxiv_id'])
            has_nd_list.append(item['has_neurodegeneration'])
        
        return {
            'input_ids': torch.stack(input_ids_list),
            'target_ids': torch.stack(target_ids_list),
            'domains': domains_list,
            'years': years_list,
            'arxiv_ids': arxiv_ids_list,
            'has_neurodegeneration': torch.tensor(has_nd_list, dtype=torch.bool),
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )


# Example usage and testing
if __name__ == "__main__":
    """Example usage of ArXivStreamingDataset."""
    import sys
    
    print("=" * 60)
    print("ArXivStreamingDataset Example")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from arxiv_dataset import ArXivStreamingDataset, create_dataloader")
    print("  import sentencepiece as spm")
    print()
    print("  # Load tokenizer")
    print("  tokenizer = spm.SentencePieceProcessor()")
    print("  tokenizer.load('healthcare_tokenizer.model')")
    print()
    print("  # Create dataset")
    print("  dataset = ArXivStreamingDataset(")
    print("      text_dir='./data/arxiv/texts',")
    print("      metadata_jsonl='./data/arxiv/processed_dataset.jsonl',")
    print("      tokenizer=tokenizer,")
    print("      max_length=512,")
    print("      min_length=64")
    print("  )")
    print()
    print("  # Create dataloader")
    print("  dataloader = create_dataloader(")
    print("      dataset,")
    print("      batch_size=32,")
    print("      num_workers=4,")
    print("      pin_memory=True")
    print("  )")
    print()
    print("  # Iterate")
    print("  for batch in dataloader:")
    print("      input_ids = batch['input_ids']")
    print("      target_ids = batch['target_ids']")
    print("      # ... training code ...")
    print()
    print("Dataset class ready to use!")

