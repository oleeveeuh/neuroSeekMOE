"""
Tokenizer Wrapper for Compatibility

Provides a unified interface for both SentencePiece and HuggingFace tokenizers.
This allows easy switching between custom SentencePiece tokenizers and pre-trained
medical tokenizers from HuggingFace.
"""

from typing import List, Union, Optional
import os

# Try to import SentencePiece
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

# Try to import transformers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TokenizerWrapper:
    """Wrapper class that provides a unified interface for tokenizers."""
    
    def __init__(self, tokenizer_path: str, tokenizer_type: str = 'auto'):
        """Initialize tokenizer wrapper.
        
        Args:
            tokenizer_path: Path to tokenizer file (SentencePiece) or model name (HuggingFace)
            tokenizer_type: Type of tokenizer ('sentencepiece', 'huggingface', or 'auto')
        """
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        self._tokenizer = None
        self._is_sentencepiece = False
        self._is_huggingface = False
        
        # Auto-detect tokenizer type
        if tokenizer_type == 'auto':
            if os.path.exists(tokenizer_path) and tokenizer_path.endswith('.model'):
                tokenizer_type = 'sentencepiece'
            else:
                tokenizer_type = 'huggingface'
        
        # Load appropriate tokenizer
        if tokenizer_type == 'sentencepiece':
            self._load_sentencepiece(tokenizer_path)
        elif tokenizer_type == 'huggingface':
            self._load_huggingface(tokenizer_path)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def _load_sentencepiece(self, path: str):
        """Load SentencePiece tokenizer."""
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("sentencepiece package required. Install with: pip install sentencepiece")
        
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(path)
        self._is_sentencepiece = True
        print(f"Loaded SentencePiece tokenizer from: {path}")
    
    def _load_huggingface(self, model_name: str):
        """Load HuggingFace tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required. Install with: pip install transformers")
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._is_huggingface = True
        print(f"Loaded HuggingFace tokenizer: {model_name}")
    
    def encode(self, text: str, out_type: type = int) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            out_type: Output type (int or list)
            
        Returns:
            List of token IDs
        """
        if self._is_sentencepiece:
            return self._tokenizer.encode(text, out_type=out_type)
        elif self._is_huggingface:
            # HuggingFace tokenizers return lists directly
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            if out_type == int:
                return token_ids
            return token_ids
        else:
            raise RuntimeError("Tokenizer not initialized")
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens (HuggingFace only)
            
        Returns:
            Decoded text
        """
        if self._is_sentencepiece:
            return self._tokenizer.decode(token_ids)
        elif self._is_huggingface:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            raise RuntimeError("Tokenizer not initialized")
    
    def piece_to_id(self, piece: str) -> int:
        """Convert token piece to ID.
        
        Args:
            piece: Token string
            
        Returns:
            Token ID
        """
        if self._is_sentencepiece:
            return self._tokenizer.piece_to_id(piece)
        elif self._is_huggingface:
            return self._tokenizer.convert_tokens_to_ids(piece)
        else:
            raise RuntimeError("Tokenizer not initialized")
    
    def id_to_piece(self, token_id: int) -> str:
        """Convert token ID to piece.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string
        """
        if self._is_sentencepiece:
            return self._tokenizer.id_to_piece(token_id)
        elif self._is_huggingface:
            return self._tokenizer.convert_ids_to_tokens(token_id)
        else:
            raise RuntimeError("Tokenizer not initialized")
    
    def get_piece_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        if self._is_sentencepiece:
            return self._tokenizer.get_piece_size()
        elif self._is_huggingface:
            return len(self._tokenizer)
        else:
            raise RuntimeError("Tokenizer not initialized")
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (property)."""
        return self.get_piece_size()
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        if self._is_sentencepiece:
            # SentencePiece typically uses 0 for padding
            return 0
        elif self._is_huggingface:
            pad_id = self._tokenizer.pad_token_id
            if pad_id is None:
                # Use unk_token_id or 0 as fallback
                return self._tokenizer.unk_token_id if self._tokenizer.unk_token_id is not None else 0
            return pad_id
        else:
            return 0
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        if self._is_sentencepiece:
            # Try to get EOS token
            try:
                return self._tokenizer.piece_to_id('</s>')
            except:
                return 2  # Default EOS for SentencePiece
        elif self._is_huggingface:
            eos_id = self._tokenizer.eos_token_id
            if eos_id is None:
                return self._tokenizer.sep_token_id if self._tokenizer.sep_token_id is not None else 2
            return eos_id
        else:
            return 2
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        if self._is_sentencepiece:
            return 0  # SentencePiece typically uses 0 for UNK
        elif self._is_huggingface:
            unk_id = self._tokenizer.unk_token_id
            if unk_id is None:
                return 0
            return unk_id
        else:
            return 0


# Default medical tokenizer to use
DEFAULT_MEDICAL_TOKENIZER = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Alternative medical tokenizers (in order of preference)
MEDICAL_TOKENIZER_OPTIONS = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",  # Best for abstracts
    "dmis-lab/biobert-v1.1",  # BioBERT
    "emilyalsentzer/Bio_ClinicalBERT",  # ClinicalBERT
    "allenai/scibert_scivocab_uncased",  # SciBERT
]


def load_medical_tokenizer(tokenizer_name: str = None) -> TokenizerWrapper:
    """Load a pre-trained medical tokenizer.
    
    Args:
        tokenizer_name: Name of tokenizer to load (default: PubMedBERT)
        
    Returns:
        TokenizerWrapper instance
    """
    if tokenizer_name is None:
        tokenizer_name = DEFAULT_MEDICAL_TOKENIZER
    
    return TokenizerWrapper(tokenizer_name, tokenizer_type='huggingface')

