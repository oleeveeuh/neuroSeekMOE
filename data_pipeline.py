"""
ArXiv Paper Collector for Healthcare+CS+ML Papers

Optimized for Colab with efficient streaming, deduplication, and checkpointing.
Collects 30-40k papers from ArXiv with healthcare, CS, and ML focus.

Usage:
    python data_pipeline.py --output-dir ./data/arxiv --max-papers 40000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import random
import time
import threading
import shutil
import sys
import gc
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration management
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️  yaml package not available. Install with: pip install pyyaml")

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil package not available. Install with: pip install psutil")

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("⚠️  arxiv package not available. Install with: pip install arxiv")

try:
    import PyPDF2
    PDF_AVAILABLE = True
    USE_PDFPLUMBER = False
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
        USE_PDFPLUMBER = True
    except ImportError:
        PDF_AVAILABLE = False
        USE_PDFPLUMBER = False
        print("⚠️  PDF library not available. Install with: pip install PyPDF2 or pip install pdfplumber")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests package not available. Install with: pip install requests")

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("⚠️  sentencepiece package not available. Install with: pip install sentencepiece")

# NeMo Curator imports (optional, Linux only)
# Use CORRECT imports based on official NeMo Curator API:
# - from nemo_curator.pipeline import Pipeline
# - from nemo_curator.stages.text.io.reader import JsonlReader
# - from nemo_curator.stages.text.modules import ScoreFilter
# - from nemo_curator.stages.text.filters import WordCountFilter
try:
    import platform
    if platform.system() == 'Linux':
        # Core Pipeline API (CORRECT path)
        try:
            from nemo_curator.pipeline import Pipeline
            Pipeline_AVAILABLE = True
        except ImportError:
            Pipeline = None
            Pipeline_AVAILABLE = False
        
        # Text processing stages (CORRECT paths)
        JsonlReader = None
        JsonlWriter = None
        ScoreFilter = None
        WordCountFilter = None
        AlphanumericFilter = None
        LanguageFilter = None
        RepeatedLineFilter = None
        
        try:
            from nemo_curator.stages.text.io.reader import JsonlReader
            JsonlReader_AVAILABLE = True
        except ImportError:
            JsonlReader_AVAILABLE = False
        
        try:
            from nemo_curator.stages.text.io.writer import JsonlWriter
            JsonlWriter_AVAILABLE = True
        except ImportError:
            try:
                from nemo_curator.stages.text.io import JsonlWriter
                JsonlWriter_AVAILABLE = True
            except ImportError:
                JsonlWriter_AVAILABLE = False
        
        try:
            from nemo_curator.stages.text.modules import ScoreFilter
            ScoreFilter_AVAILABLE = True
        except ImportError:
            ScoreFilter_AVAILABLE = False
        
        try:
            from nemo_curator.stages.text.filters import (
                WordCountFilter,
                AlphanumericFilter,
                LanguageFilter,
                RepeatedLineFilter
            )
            Filters_AVAILABLE = True
        except ImportError:
            Filters_AVAILABLE = False
        
        # Try to import ProcessingStage for custom stages
        ProcessingStage = None
        Stage = None
        try:
            from nemo_curator.stages.text import ProcessingStage, Stage
            ProcessingStage_AVAILABLE = True
            Stage_AVAILABLE = True
        except ImportError:
            try:
                from nemo_curator.stages import ProcessingStage, Stage
                ProcessingStage_AVAILABLE = True
                Stage_AVAILABLE = True
            except ImportError:
                ProcessingStage_AVAILABLE = False
                Stage_AVAILABLE = False
        
        # Try to import download_arxiv function (FREE, no AWS needed)
        try:
            from nemo_curator.download import download_arxiv
            download_arxiv_AVAILABLE = True
        except ImportError:
            download_arxiv = None
            download_arxiv_AVAILABLE = False
        
        # Try to import Dask client utility
        try:
            from nemo_curator.utils.distributed_utils import get_client
            get_client_AVAILABLE = True
        except ImportError:
            try:
                from dask.distributed import Client
                # Create a wrapper function
                def get_client():
                    try:
                        from dask.distributed import get_client as dask_get_client
                        return dask_get_client()
                    except:
                        return Client(processes=False, threads_per_worker=2)
                get_client_AVAILABLE = True
            except ImportError:
                get_client = None
                get_client_AVAILABLE = False
        
        # Legacy imports (for backward compatibility with old code)
        DocumentDataset = None
        DocumentModifier = None
        DocumentFilter = None
        try:
            from nemo_curator.datasets import DocumentDataset
        except ImportError:
            pass
        
        try:
            from nemo_curator.modifiers import DocumentModifier
        except ImportError:
            pass
        
        try:
            from nemo_curator.filters import DocumentFilter
        except ImportError:
            pass
        
        import dask
        NEMO_CURATOR_AVAILABLE = True
        print("✅ NeMo Curator imported successfully")
        print(f"   Pipeline: {Pipeline_AVAILABLE}")
        print(f"   JsonlReader: {JsonlReader_AVAILABLE}")
        print(f"   ScoreFilter: {ScoreFilter_AVAILABLE}")
        print(f"   Filters: {Filters_AVAILABLE}")
    else:
        NEMO_CURATOR_AVAILABLE = False
        Pipeline_AVAILABLE = False
        JsonlReader_AVAILABLE = False
        ScoreFilter_AVAILABLE = False
        Filters_AVAILABLE = False
        ProcessingStage_AVAILABLE = False
        Stage_AVAILABLE = False
        download_arxiv_AVAILABLE = False
        get_client_AVAILABLE = False
        print("⚠️  NeMo Curator only supports Linux systems (current: {})".format(platform.system()))
except (ImportError, ValueError) as e:
    NEMO_CURATOR_AVAILABLE = False
    Pipeline_AVAILABLE = False
    JsonlReader_AVAILABLE = False
    ScoreFilter_AVAILABLE = False
    Filters_AVAILABLE = False
    ProcessingStage_AVAILABLE = False
    Stage_AVAILABLE = False
    download_arxiv_AVAILABLE = False
    get_client_AVAILABLE = False
    print("⚠️  nemo-curator package not available. Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'")
    print(f"   Error: {e}")


# Rate limiting: 3 requests per second
RATE_LIMIT_DELAY = 1.0 / 3.0  # ~0.33 seconds between requests
CHECKPOINT_INTERVAL = 5000  # Save checkpoint every 5000 papers
LOG_INTERVAL = 500  # Log progress every 500 papers

# Target date range (set to None to disable date filtering)
MIN_YEAR = 2016  # None = no minimum (accept all years)
MAX_YEAR = None  # None = no maximum (accept all years)
# Alternative: MIN_YEAR = 2015, MAX_YEAR = 2024 for date filtering

# ArXiv search queries
ARXIV_QUERIES = [
    "cat:cs.LG AND (healthcare OR medical OR clinical)",
    "cat:cs.AI AND (neurodegeneration OR disease)",
    "cat:q-bio.NC AND (machine learning)",
]

# Output fields (minimal metadata)
OUTPUT_FIELDS = ['id', 'title', 'abstract', 'year', 'categories', 'pdf_url']

# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary with defaults
    """
    defaults = {
        'pipeline': {
            'output_dir': './data/arxiv',
            'max_papers': 30000,
            'skip_stages': [],
            'resume': True
        },
        'collection': {
            'max_papers': 30000,  # For backward compatibility
            'rate_limit': 0.33,  # requests per second
            'retry_max': 5
        },
        'extraction': {
            'workers': 2,  # Colab safe
            'rate_limit': 0.4,
            'max_pages': 6,
            'max_chars': 12000
        },
        'curation': {
            'use_nemo_curator': True,
            'skip_deduplication': False,
            'min_relevance_score': 0.5
        },
        'preprocessing': {
            'workers': 4
        },
        'tokenizer': {
            'vocab_size': 50000,
            'model_type': 'bpe',
            'character_coverage': 0.9995
        }
    }
    
    if not YAML_AVAILABLE:
        print("⚠️  YAML not available, using defaults")
        return defaults
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            # Merge with defaults
            config = defaults.copy()
            for key, value in user_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
            
            # Ensure max_papers is in both pipeline and collection for backward compatibility
            if 'pipeline' in config and 'max_papers' in config['pipeline']:
                if 'collection' not in config:
                    config['collection'] = {}
                config['collection']['max_papers'] = config['pipeline']['max_papers']
            elif 'collection' in config and 'max_papers' in config['collection']:
                if 'pipeline' not in config:
                    config['pipeline'] = {}
                config['pipeline']['max_papers'] = config['collection']['max_papers']
            
            print(f"✅ Loaded config from {config_path}")
            return config
        except Exception as e:
            print(f"⚠️  Error loading config: {e}, using defaults")
            return defaults
    else:
        # Create default config file
        try:
            with open(config_path, 'w') as f:
                yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)
            print(f"📝 Created default config file: {config_path}")
        except Exception as e:
            print(f"⚠️  Could not create config file: {e}")
        return defaults


def save_config(config: Dict, config_path: str = "config.yaml"):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to config.yaml file
    """
    if not YAML_AVAILABLE:
        print("⚠️  YAML not available, cannot save config")
        return
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Saved config to {config_path}")
    except Exception as e:
        print(f"⚠️  Error saving config: {e}")


# ============================================================================
# Memory Monitoring
# ============================================================================

def get_memory_usage() -> Dict:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory stats
    """
    if not PSUTIL_AVAILABLE:
        return {'available': None, 'percent': None, 'total': None}
    
    try:
        mem = psutil.virtual_memory()
        return {
            'available': mem.available,
            'percent': mem.percent,
            'total': mem.total,
            'used': mem.used
        }
    except Exception:
        return {'available': None, 'percent': None, 'total': None}


def check_memory_usage(warning_threshold: float = 80.0) -> bool:
    """Check if memory usage is below warning threshold.
    
    Args:
        warning_threshold: Warning threshold percentage (default: 80%)
        
    Returns:
        True if memory usage is safe, False if warning threshold exceeded
    """
    if not PSUTIL_AVAILABLE:
        return True  # Assume safe if psutil not available
    
    mem_stats = get_memory_usage()
    if mem_stats['percent'] is not None:
        if mem_stats['percent'] > warning_threshold:
            print(f"⚠️  Memory usage: {mem_stats['percent']:.1f}% (threshold: {warning_threshold}%)")
            print(f"   Available: {mem_stats['available'] / (1024**3):.2f} GB")
            return False
    return True


def log_memory_stats(interval: int = 1000):
    """Log memory statistics periodically.
    
    Args:
        interval: Log every N papers processed
    """
    if not PSUTIL_AVAILABLE:
        return
    
    mem_stats = get_memory_usage()
    if mem_stats['percent'] is not None:
        print(f"   💾 Memory: {mem_stats['percent']:.1f}% used, {mem_stats['available'] / (1024**3):.2f} GB available")


# ============================================================================
# Validation & Diagnostics
# ============================================================================

def check_arxiv_connection() -> bool:
    """Check if ArXiv API is accessible.
    
    Returns:
        True if accessible, False otherwise
    """
    if not ARXIV_AVAILABLE:
        print("❌ ArXiv package not available")
        return False
    
    try:
        client = arxiv.Client(page_size=1, delay_seconds=0.5, num_retries=1)
        search = arxiv.Search(query="cat:cs.LG", max_results=1)
        result = next(client.results(search), None)
        if result:
            print("✅ ArXiv API is accessible")
            return True
        else:
            print("⚠️  ArXiv API returned no results (might be temporary)")
            return False
    except Exception as e:
        print(f"❌ ArXiv API connection failed: {e}")
        return False


def check_disk_space(path: str, required_gb: float = 50.0) -> bool:
    """Check if sufficient disk space is available.
    
    Args:
        path: Path to check disk space for
        required_gb: Required space in GB (default: 50GB)
        
    Returns:
        True if sufficient space, False otherwise
    """
    try:
        if PSUTIL_AVAILABLE:
            stat = shutil.disk_usage(path)
            available_gb = stat.free / (1024**3)
            print(f"💾 Disk space: {available_gb:.2f} GB available (required: {required_gb:.2f} GB)")
            if available_gb < required_gb:
                print(f"⚠️  Warning: Insufficient disk space")
                return False
            return True
        else:
            print("⚠️  psutil not available, cannot check disk space")
            return True  # Assume OK
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Assume OK


def check_output_directory(path: str) -> bool:
    """Check if output directory is writable.
    
    Args:
        path: Path to output directory
        
    Returns:
        True if writable, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"✅ Output directory is writable: {path}")
        return True
    except Exception as e:
        print(f"❌ Output directory not writable: {path} - {e}")
        return False


def count_existing_files(directory: str, pattern: str = "*.jsonl") -> int:
    """Count existing files matching pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern (default: *.jsonl)
        
    Returns:
        Number of matching files
    """
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern.replace('*', '')):
                count += 1
    return count


def print_diagnostics(config: Dict):
    """Print diagnostic information at startup.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 60)
    print("🔍 Pipeline Diagnostics")
    print("=" * 60)
    
    # Check ArXiv connection
    print("\n1. ArXiv API Connection:")
    arxiv_ok = check_arxiv_connection()
    
    # Check disk space
    print("\n2. Disk Space:")
    output_dir = config.get('pipeline', {}).get('output_dir', './data/arxiv')
    check_disk_space(output_dir, required_gb=50.0)
    
    # Check output directory
    print("\n3. Output Directory:")
    check_output_directory(output_dir)
    
    # Check existing files
    print("\n4. Existing Files:")
    existing_count = count_existing_files(output_dir)
    print(f"   Found {existing_count} existing files in {output_dir}")
    
    # Memory status
    print("\n5. Memory Status:")
    mem_stats = get_memory_usage()
    if mem_stats['percent'] is not None:
        print(f"   Memory usage: {mem_stats['percent']:.1f}%")
        print(f"   Available: {mem_stats['available'] / (1024**3):.2f} GB")
    else:
        print("   Memory monitoring not available (install psutil)")
    
    # Package availability
    print("\n6. Package Availability:")
    print(f"   ArXiv: {'✅' if ARXIV_AVAILABLE else '❌'}")
    print(f"   PDF: {'✅' if PDF_AVAILABLE else '❌'}")
    print(f"   NeMo Curator: {'✅' if NEMO_CURATOR_AVAILABLE else '❌'}")
    print(f"   SentencePiece: {'✅' if SENTENCEPIECE_AVAILABLE else '❌'}")
    print(f"   YAML: {'✅' if YAML_AVAILABLE else '❌'}")
    print(f"   psutil: {'✅' if PSUTIL_AVAILABLE else '❌'}")
    
    print("\n" + "=" * 60)
    
    return arxiv_ok


def load_existing_ids(cache_file: str) -> Set[str]:
    """Load existing ArXiv IDs from cache file to avoid re-downloading.
    
    Args:
        cache_file: Path to JSONL cache file
        
    Returns:
        Set of ArXiv IDs already in cache
    """
    existing_ids = set()
    if os.path.exists(cache_file):
        print(f"📖 Loading existing cache from {cache_file}...")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            paper = json.loads(line)
                            if 'id' in paper:
                                existing_ids.add(paper['id'])
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
                            continue
            print(f"   ✅ Found {len(existing_ids)} existing papers in cache")
        except Exception as e:
            print(f"⚠️  Warning: Error reading cache file: {e}")
            print("   Starting fresh...")
    else:
        print(f"📝 No existing cache found. Starting fresh collection...")
    
    return existing_ids


def extract_year_from_date(date: Optional[datetime]) -> Optional[int]:
    """Extract year from ArXiv date object.
    
    Args:
        date: ArXiv date object or None
        
    Returns:
        Year as integer, or None if date is invalid
    """
    if date is None:
        return None
    try:
        return date.year
    except (AttributeError, ValueError):
        return None


def format_abstract(abstract: str, max_length: int = 300) -> str:
    """Format abstract to first N characters.
    
    Args:
        abstract: Full abstract text
        max_length: Maximum length to keep
        
    Returns:
        Truncated abstract
    """
    if not abstract:
        return ""
    abstract = abstract.strip()
    if len(abstract) > max_length:
        # Try to truncate at word boundary
        truncated = abstract[:max_length].rsplit(' ', 1)[0]
        return truncated + "..."
    return abstract


def paper_to_dict(paper: arxiv.Result) -> Optional[Dict]:
    """Convert ArXiv paper result to minimal dictionary format.
    
    Args:
        paper: ArXiv result object
        
    Returns:
        Dictionary with minimal fields, or None if paper should be skipped
    """
    # Skip papers without abstracts
    if not paper.summary or not paper.summary.strip():
        return None
    
    # Extract year
    year = extract_year_from_date(paper.published)
    
    # Skip papers outside target date range (if date filtering is enabled)
    if MIN_YEAR is not None and year is not None and year < MIN_YEAR:
        return None
    if MAX_YEAR is not None and year is not None and year > MAX_YEAR:
        return None
    
    # Format categories (handle both string and object formats)
    categories = []
    if paper.categories:
        for cat in paper.categories:
            if isinstance(cat, str):
                categories.append(cat)
            elif hasattr(cat, 'term'):
                categories.append(cat.term)
            else:
                categories.append(str(cat))
    
    # Build minimal metadata dict
    paper_dict = {
        'id': paper.entry_id.split('/')[-1],  # Extract ArXiv ID from URL
        'title': paper.title.strip() if paper.title else "",
        'abstract': format_abstract(paper.summary, max_length=300),
        'year': year,
        'categories': categories,
        'pdf_url': paper.pdf_url if hasattr(paper, 'pdf_url') else f"https://arxiv.org/pdf/{paper.entry_id.split('/')[-1]}.pdf",
    }
    
    return paper_dict


def search_arxiv_query_streaming(
    query: str,
    max_results: int,
    output_file_handle,
    existing_ids: Set[str],
    rate_limit_delay: float = RATE_LIMIT_DELAY,
    per_result_timeout: int = 5,
    query_timeout: int = 1800
) -> int:
    """Search ArXiv and stream results one-at-a-time to disk.
    
    Args:
        query: ArXiv search query
        max_results: Maximum number of results to fetch
        output_file_handle: Open file handle for writing (in append mode)
        existing_ids: Set of IDs to skip (already in cache)
        rate_limit_delay: Delay between requests (seconds)
        per_result_timeout: Timeout per result in seconds (default: 5)
        query_timeout: Total timeout for query in seconds (default: 1800 = 30 min)
        
    Returns:
        Number of papers found and written
    """
    if existing_ids is None:
        existing_ids = set()
    
    # Validate query
    if not query or not query.strip():
        print(f"   ❌ Error: Empty query")
        return 0
    
    # Timeout handling (Unix only - signal.SIGALRM)
    use_signal_timeout = False
    try:
        import signal
        if hasattr(signal, 'SIGALRM'):
            use_signal_timeout = True
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
    except (ImportError, AttributeError):
        print("   ⚠️  Signal-based timeout not available (Windows), using time-based checks only")
    
    print(f"\n🔍 Query: {query}")
    print(f"   Max results: {max_results}")
    print(f"   Per-result timeout: {per_result_timeout}s")
    print(f"   Query timeout: {query_timeout}s ({query_timeout/60:.1f} min)")
    
    papers_found = 0
    papers_skipped = 0
    skipped_no_abstract = 0
    skipped_date_range = 0
    skipped_duplicate = 0
    
    query_start = time.time()
    last_progress_time = query_start
    
    try:
        # Create ArXiv client
        client = arxiv.Client(
            page_size=100,
            delay_seconds=rate_limit_delay,
            num_retries=2
        )
        
        # Search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        print(f"   🔄 Fetching results from ArXiv API...")
        results_iter = client.results(search)
        
        # Set query-level timeout
        if use_signal_timeout:
            signal.alarm(query_timeout)
        
        result_idx = -1
        for result_idx, result in enumerate(results_iter):
            # Check query timeout (time-based, works on all platforms)
            elapsed = time.time() - query_start
            if elapsed > query_timeout:
                print(f"   ⏱️  Query timeout ({query_timeout}s), aborting")
                break
            
            # Per-result timeout
            if use_signal_timeout:
                signal.alarm(per_result_timeout)
            
            result_start = time.time()
            
            try:
                # Process ONE result at a time
                paper_id = result.entry_id.split('/')[-1]
                
                # Skip if already seen
                if paper_id in existing_ids:
                    skipped_duplicate += 1
                    papers_skipped += 1
                    if use_signal_timeout:
                        signal.alarm(0)
                    continue
                
                # Convert to dict (fast operation)
                paper_dict = paper_to_dict(result)
                if paper_dict is None:
                    if not result.summary or not result.summary.strip():
                        skipped_no_abstract += 1
                    else:
                        year = extract_year_from_date(result.published)
                        if MIN_YEAR is not None and year is not None and year < MIN_YEAR:
                            skipped_date_range += 1
                        elif MAX_YEAR is not None and year is not None and year > MAX_YEAR:
                            skipped_date_range += 1
                    papers_skipped += 1
                    if use_signal_timeout:
                        signal.alarm(0)
                    continue
                
                # Write immediately to disk
                output_file_handle.write(json.dumps(paper_dict, ensure_ascii=False) + '\n')
                output_file_handle.flush()  # Force immediate write
                
                papers_found += 1
                existing_ids.add(paper_id)
                
                # Progress update every 10 results
                if papers_found % 10 == 0:
                    elapsed = time.time() - query_start
                    rate = papers_found / elapsed if elapsed > 0 else 0
                    print(f"   📊 {papers_found} papers, {elapsed:.0f}s elapsed, {rate:.2f} papers/sec")
                    last_progress_time = time.time()
                
                # Memory check every 100 results
                if papers_found % 100 == 0:
                    if PSUTIL_AVAILABLE:
                        mem_stats = get_memory_usage()
                        if mem_stats['percent'] is not None and mem_stats['percent'] > 80:
                            print(f"   ⚠️  RAM at {mem_stats['percent']:.0f}%, stopping to prevent OOM")
                            break
                
                # Check if we've hit max_results
                if papers_found >= max_results:
                    break
                
                # Cancel alarm after successful processing
                if use_signal_timeout:
                    signal.alarm(0)
                
                # Rate limiting
                time.sleep(rate_limit_delay)
                
            except TimeoutError:
                elapsed_result = time.time() - result_start
                print(f"   ⚠️  Result #{papers_found + 1} took >{per_result_timeout}s, skipping...")
                papers_skipped += 1
                if use_signal_timeout:
                    signal.alarm(0)
                continue
            except Exception as e:
                print(f"   ⚠️  Error processing result: {e}")
                papers_skipped += 1
                if use_signal_timeout:
                    signal.alarm(0)
                continue
        
        # Cancel final alarm
        if use_signal_timeout:
            signal.alarm(0)
        
        # Final summary
        elapsed = time.time() - query_start
        rate = papers_found / elapsed if elapsed > 0 else 0
        print(f"   ✅ Query complete: {papers_found} papers ({rate:.2f} papers/sec)")
        if skipped_duplicate > 0:
            print(f"   ⏭️  Skipped {skipped_duplicate} duplicates")
        if skipped_no_abstract > 0:
            print(f"   ⏭️  Skipped {skipped_no_abstract} papers without abstracts")
        if skipped_date_range > 0:
            if MIN_YEAR is not None and MAX_YEAR is not None:
                date_range_str = f"{MIN_YEAR}-{MAX_YEAR}"
            elif MIN_YEAR is not None:
                date_range_str = f">={MIN_YEAR}"
            elif MAX_YEAR is not None:
                date_range_str = f"<={MAX_YEAR}"
            else:
                date_range_str = "all years"
            print(f"   ⏭️  Skipped {skipped_date_range} papers outside date range ({date_range_str})")
        
        if papers_found == 0:
            if result_idx == -1:
                print(f"   ⚠️  Warning: No results found for query")
                print(f"      - Query: {query}")
                print(f"      - This might indicate query syntax issue or API problem")
            else:
                print(f"   ⚠️  Warning: Processed {result_idx + 1} results but none matched criteria")
        
    except TimeoutError:
        elapsed = time.time() - query_start
        print(f"   ⏱️  Query timed out after {elapsed:.0f}s")
    except KeyboardInterrupt:
        print(f"\n   ⏸️  Query interrupted by user")
        raise
    except Exception as e:
        print(f"   ❌ Query error: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    return papers_found


# Backward compatibility alias
def search_arxiv_query(
    query: str,
    max_results: int = 10000,
    existing_ids: Set[str] = None,
    rate_limit_delay: float = RATE_LIMIT_DELAY,
    max_retries: int = 5
) -> list[Dict]:
    """Legacy function - accumulates results in memory. Use search_arxiv_query_streaming() instead."""
    print("⚠️  Warning: Using legacy search_arxiv_query() which accumulates results in memory.")
    print("   Consider using search_arxiv_query_streaming() for better memory efficiency.")
    
    if existing_ids is None:
        existing_ids = set()
    
    papers = []
    # Use streaming function with temporary file, then read back
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl', encoding='utf-8') as tmp:
        try:
            count = search_arxiv_query_streaming(
                query=query,
                max_results=max_results,
                output_file_handle=tmp,
                existing_ids=existing_ids,
                rate_limit_delay=rate_limit_delay,
                per_result_timeout=5,
                query_timeout=1800
            )
            # Read back results (for backward compatibility)
            tmp.seek(0)
            for line in tmp:
                if line.strip():
                    papers.append(json.loads(line))
        finally:
            os.unlink(tmp.name)
    
    return papers


def save_papers_to_jsonl(papers: list[Dict], output_file: str, mode: str = 'a'):
    """Save papers to JSONL file.
    
    Args:
        papers: List of paper dictionaries
        output_file: Path to output JSONL file
        mode: File mode ('a' for append, 'w' for write)
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, mode, encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')


class RAMEfficientArxivCollector:
    """Collect ArXiv papers without OOM, with checkpointing."""
    
    def __init__(
        self,
        output_file: str,
        checkpoint_file: str,
        batch_size: int = 10,
        ram_target_percent: float = 50.0,
        rate_limit: float = 0.33
    ):
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.batch_size = batch_size
        self.ram_target_percent = ram_target_percent
        self.rate_limit = rate_limit
        self.collected_ids = set()
        self.total_collected = 0
        
        # Load checkpoint
        self._load_checkpoint()
    
    def _get_ram_percent(self) -> float:
        """Get current RAM usage percentage."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _log_memory(self, prefix: str = ""):
        """Log current RAM usage."""
        if not PSUTIL_AVAILABLE:
            return
        try:
            mem = psutil.virtual_memory()
            percent = mem.percent
            available_gb = mem.available / (1024**3)
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            print(f"   {prefix} 💾 RAM: {used_gb:.1f}GB/{total_gb:.1f}GB ({percent:.0f}%)")
        except:
            pass
    
    def _save_checkpoint(self):
        """Save checkpoint with current progress."""
        checkpoint = {
            'timestamp': time.time(),
            'total_collected': self.total_collected,
            'collected_ids': list(self.collected_ids)[:1000],  # Keep last 1000 for dedup
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        print(f"   💾 Checkpoint saved: {self.total_collected} papers")
    
    def _load_checkpoint(self):
        """Load checkpoint to resume collection.
        
        Always loads ALL IDs from the output file to ensure accurate deduplication.
        The checkpoint file is only used for the count, but we verify against the actual file.
        """
        try:
            # Always load from output file first (most accurate source of truth)
            if os.path.exists(self.output_file):
                existing_ids = load_existing_ids(self.output_file)
                self.collected_ids = existing_ids
                self.total_collected = len(existing_ids)
                if self.total_collected > 0:
                    print(f"📖 Found {self.total_collected} existing papers in output file")
            
            # Also check checkpoint file for count verification
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    checkpoint_count = checkpoint.get('total_collected', 0)
                    if checkpoint_count != self.total_collected:
                        print(f"⚠️  Checkpoint count ({checkpoint_count}) differs from file count ({self.total_collected}), using file count")
                    # Note: We don't use checkpoint's collected_ids since we loaded all from file
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            if self.total_collected == 0:
                print("📝 Starting fresh")
    
    def _clear_memory(self):
        """Aggressively clear memory."""
        gc.collect()
        self._log_memory("After gc.collect():")
    
    def collect_batch(
        self,
        query: str,
        batch_num: int,
        batch_size: int
    ) -> int:
        """
        Collect one batch of papers.
        
        Returns: Number of papers collected in this batch
        """
        print(f"\n   Batch {batch_num}: Collecting up to {batch_size} papers...")
        self._log_memory("Before batch:")
        
        papers_in_batch = 0
        batch_start = time.time()
        
        if not ARXIV_AVAILABLE:
            print("   ❌ ArXiv package not available")
            return 0
        
        try:
            client = arxiv.Client(
                delay_seconds=self.rate_limit,
                num_retries=2
            )
            
            search = arxiv.Search(
                query=query,
                max_results=batch_size,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for result in client.results(search):
                    paper_id = result.entry_id.split('/')[-1]
                    
                    # Skip if already collected
                    if paper_id in self.collected_ids:
                        continue
                    
                    # Convert to dict (handle categories properly)
                    categories = []
                    if result.categories:
                        for cat in result.categories:
                            if isinstance(cat, str):
                                categories.append(cat)
                            elif hasattr(cat, 'term'):
                                categories.append(cat.term)
                            else:
                                categories.append(str(cat))
                    
                    year = None
                    if result.published:
                        try:
                            year = result.published.year
                        except:
                            pass
                    
                    # Skip if date filtering enabled and outside range
                    if MIN_YEAR is not None and year is not None and year < MIN_YEAR:
                        continue
                    if MAX_YEAR is not None and year is not None and year > MAX_YEAR:
                        continue
                    
                    # Skip if no abstract
                    if not result.summary or not result.summary.strip():
                        continue
                    
                    paper = {
                        'id': paper_id,
                        'title': result.title.strip() if result.title else "",
                        'abstract': format_abstract(result.summary, max_length=300),
                        'year': year,
                        'categories': categories,
                        'pdf_url': f"https://arxiv.org/pdf/{paper_id}.pdf",
                    }
                    
                    # Write immediately
                    f.write(json.dumps(paper, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    # Track collected
                    self.collected_ids.add(paper_id)
                    self.total_collected += 1
                    papers_in_batch += 1
                    
                    # Check RAM after every 5 papers
                    if papers_in_batch % 5 == 0:
                        ram_percent = self._get_ram_percent()
                        if ram_percent > self.ram_target_percent:
                            print(f"   ⚠️  RAM at {ram_percent:.0f}%, stopping batch early")
                            break
                    
                    # Rate limiting
                    time.sleep(self.rate_limit)
        
        except Exception as e:
            print(f"   ⚠️  Batch error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        
        elapsed = time.time() - batch_start
        rate = papers_in_batch / elapsed if elapsed > 0 else 0
        
        print(f"   ✅ Batch {batch_num}: {papers_in_batch} papers ({rate:.1f} papers/sec)")
        self._log_memory("After batch:")
        
        return papers_in_batch
    
    def collect_query(
        self,
        query: str,
        max_papers: int,
        query_num: int,
        total_queries: int
    ) -> int:
        """
        Collect papers for one query, using multiple batches.
        
        Uses a single large search and processes results incrementally to avoid
        getting the same papers in each batch.
        
        Returns: Total papers collected for this query
        """
        print(f"\n{'='*70}")
        print(f"Query {query_num}/{total_queries}: {query[:60]}...")
        print(f"{'='*70}")
        
        if not ARXIV_AVAILABLE:
            print("   ❌ ArXiv package not available")
            return 0
        
        papers_in_query = 0
        batch_num = 0
        total_results_checked = 0  # Track across all batches
        consecutive_empty_batches = 0  # Track consecutive batches with 0 papers
        
        # Fetch many more results to account for filtering (no abstract, date range, etc.)
        # ArXiv API allows up to 300,000 results, so we can request a large number
        # Use a much larger window to ensure we can find enough papers
        max_search_results = max(max_papers * 10, 50000)  # Fetch 10x target to account for filtering
        
        try:
            client = arxiv.Client(
                delay_seconds=self.rate_limit,
                num_retries=2
            )
            
            # Single large search for the entire query
            search = arxiv.Search(
                query=query,
                max_results=max_search_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results_iter = client.results(search)
            
            while papers_in_query < max_papers:
                batch_num += 1
                
                # Adjust batch size based on RAM
                ram_percent = self._get_ram_percent()
                if ram_percent > 70:
                    current_batch_size = max(5, self.batch_size // 2)
                    print(f"   ⚠️  High RAM ({ram_percent:.0f}%), reducing batch size to {current_batch_size}")
                elif ram_percent > 50:
                    current_batch_size = max(5, int(self.batch_size * 0.75))
                    print(f"   ⚠️  Moderate RAM ({ram_percent:.0f}%), reducing batch size to {current_batch_size}")
                else:
                    current_batch_size = self.batch_size
                
                # Collect batch from the same iterator
                papers, results_checked = self._collect_batch_from_iterator(
                    results_iter, query, batch_num, current_batch_size
                )
                
                total_results_checked += results_checked
                
                if papers == 0:
                    consecutive_empty_batches += 1
                    print(f"   ⚠️  Batch returned 0 papers (checked {results_checked} results, total checked: {total_results_checked})")
                    
                    # If we've checked many results with no new papers, move to next query
                    if consecutive_empty_batches >= 3:
                        print(f"   ⚠️  {consecutive_empty_batches} consecutive empty batches, moving to next query")
                        break
                    elif total_results_checked > 5000 and papers_in_query == 0:
                        print(f"   ⚠️  Query returned no papers after checking {total_results_checked} results, moving to next query")
                        break
                    elif total_results_checked > 10000:
                        print(f"   ⚠️  Query exhausted after checking {total_results_checked} results, moving to next query")
                        break
                    # Otherwise, try one more batch in case of temporary issues
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    # Reset counter if we got papers
                    consecutive_empty_batches = 0
                
                papers_in_query += papers
                
                # Save checkpoint after each batch
                self._save_checkpoint()
                
                # Clear memory
                self._clear_memory()
                
                # Avoid hitting rate limits
                time.sleep(1)
        
        except StopIteration:
            print("   Reached end of search results")
        except Exception as e:
            print(f"   ⚠️  Query error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        
        print(f"\n✅ Query complete: {papers_in_query} papers")
        return papers_in_query
    
    def _collect_batch_from_iterator(
        self,
        results_iter,
        query: str,
        batch_num: int,
        batch_size: int
    ) -> Tuple[int, int]:
        """
        Collect one batch of papers from an existing results iterator.
        
        This avoids restarting the search and getting duplicate results.
        
        Returns: Tuple of (number of papers collected, number of results checked)
        """
        print(f"\n   Batch {batch_num}: Collecting up to {batch_size} papers...")
        self._log_memory("Before batch:")
        
        papers_in_batch = 0
        batch_start = time.time()
        results_checked = 0
        
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for result in results_iter:
                    results_checked += 1
                    paper_id = result.entry_id.split('/')[-1]
                    
                    # Skip if already collected
                    if paper_id in self.collected_ids:
                        continue
                    
                    # Convert to dict (handle categories properly)
                    categories = []
                    if result.categories:
                        for cat in result.categories:
                            if isinstance(cat, str):
                                categories.append(cat)
                            elif hasattr(cat, 'term'):
                                categories.append(cat.term)
                            else:
                                categories.append(str(cat))
                    
                    year = None
                    if result.published:
                        try:
                            year = result.published.year
                        except:
                            pass
                    
                    # Skip if date filtering enabled and outside range
                    if MIN_YEAR is not None and year is not None and year < MIN_YEAR:
                        continue
                    if MAX_YEAR is not None and year is not None and year > MAX_YEAR:
                        continue
                    
                    # Skip if no abstract
                    if not result.summary or not result.summary.strip():
                        continue
                    
                    paper = {
                        'id': paper_id,
                        'title': result.title.strip() if result.title else "",
                        'abstract': format_abstract(result.summary, max_length=300),
                        'year': year,
                        'categories': categories,
                        'pdf_url': f"https://arxiv.org/pdf/{paper_id}.pdf",
                    }
                    
                    # Write immediately
                    f.write(json.dumps(paper, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    # Track collected
                    self.collected_ids.add(paper_id)
                    self.total_collected += 1
                    papers_in_batch += 1
                    
                    # Stop when we've collected enough for this batch
                    if papers_in_batch >= batch_size:
                        break
                    
                    # Check RAM after every 5 papers
                    if papers_in_batch % 5 == 0:
                        ram_percent = self._get_ram_percent()
                        if ram_percent > self.ram_target_percent:
                            print(f"   ⚠️  RAM at {ram_percent:.0f}%, stopping batch early")
                            break
                    
                    # Rate limiting
                    time.sleep(self.rate_limit)
        
        except StopIteration:
            # End of results
            pass
        except Exception as e:
            print(f"   ⚠️  Batch error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        
        elapsed = time.time() - batch_start
        rate = papers_in_batch / elapsed if elapsed > 0 else 0
        
        print(f"   ✅ Batch {batch_num}: {papers_in_batch} papers ({rate:.1f} papers/sec, checked {results_checked} results)")
        self._log_memory("After batch:")
        
        return papers_in_batch, results_checked
    
    def collect_all(
        self,
        queries: List[Tuple[str, int]],
        total_target: int
    ):
        """
        Collect papers from multiple queries.
        
        Args:
            queries: List of (query_str, max_papers_per_query) tuples
            total_target: Total papers to collect
        """
        print("\n" + "="*70)
        print("🚀 RAM-Efficient ArXiv Collection")
        print("="*70)
        print(f"📊 Starting from: {self.total_collected} papers")
        print(f"🎯 Target: {total_target} papers")
        print(f"📦 Batch size: {self.batch_size} papers/batch")
        print(f"💾 RAM target: <{self.ram_target_percent:.0f}%")
        if MIN_YEAR is not None or MAX_YEAR is not None:
            date_range = f"{MIN_YEAR or 'any'}-{MAX_YEAR or 'any'}"
            print(f"📅 Date range: {date_range}")
        else:
            print(f"📅 Date range: All years (no filtering)")
        print()
        
        try:
            for query_num, (query, max_per_query) in enumerate(queries, 1):
                if self.total_collected >= total_target:
                    print(f"\n✅ Reached target of {total_target} papers")
                    break
                
                # Adjust max for this query
                remaining = total_target - self.total_collected
                max_for_query = min(max_per_query, remaining)
                
                # Collect query
                papers = self.collect_query(
                    query=query,
                    max_papers=max_for_query,
                    query_num=query_num,
                    total_queries=len(queries)
                )
        
        except KeyboardInterrupt:
            print("\n⏸️  Collection paused by user")
        
        finally:
            self._save_checkpoint()
            print(f"\n✅ Total collected: {self.total_collected} papers")
            self._log_memory("Final:")


def collect_arxiv_efficient(
    output_dir: str = "./data/arxiv",
    total_target: int = 10000,
    batch_size: int = 10,
    ram_target: float = 50.0,
    rate_limit: float = 0.33
):
    """
    Collect ArXiv papers efficiently without OOM.
    
    Args:
        output_dir: Output directory
        total_target: Total papers to collect
        batch_size: Papers per batch (adjust for RAM)
        ram_target: Target RAM percentage to stay below
        rate_limit: Delay between requests (seconds)
    """
    if not ARXIV_AVAILABLE:
        error_msg = "❌ Error: arxiv package not available. Install with: pip install arxiv"
        print(error_msg)
        raise ImportError(error_msg)
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "arxiv_papers.jsonl")
    checkpoint_file = os.path.join(output_dir, "collection_checkpoint.json")
    
    # Initialize collector
    collector = RAMEfficientArxivCollector(
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        batch_size=batch_size,
        ram_target_percent=ram_target,
        rate_limit=rate_limit
    )
    
    # Define queries with higher limits to reach target
    # Each query can contribute up to its limit, but we'll continue until total_target is reached
    # Use larger per-query limits to ensure we can reach the target
    query_limit_per_query = max(total_target // 3, 10000)  # Distribute target across queries with larger limits
    queries = [
        ("cat:cs.LG AND healthcare", query_limit_per_query),
        ("cat:cs.AI AND (neurodegeneration OR disease)", query_limit_per_query),
        ("cat:q-bio.NC AND (machine learning)", query_limit_per_query),
        ("cat:stat.ML AND medical", query_limit_per_query),
        # Additional broader queries to help reach target
        ("cat:cs.LG AND (medical OR clinical OR health)", query_limit_per_query),
        ("cat:cs.AI AND (disease OR diagnosis OR treatment)", query_limit_per_query),
        # Even broader queries as fallback
        ("cat:cs.LG AND (health OR medicine)", query_limit_per_query),
        ("cat:cs.AI AND health", query_limit_per_query),
    ]
    
    # Collect
    collector.collect_all(queries, total_target)
    
    print(f"\n📁 Papers saved to: {output_file}")
    print(f"💾 Checkpoint saved to: {checkpoint_file}")
    print(f"📖 To resume: Run collect_arxiv_efficient() again")


def collect_arxiv_papers(
    output_dir: str = "./data/arxiv",
    max_papers: int = 40000,
    cache_file: str = None,
    rate_limit_delay: float = RATE_LIMIT_DELAY,
    batch_size: int = 10,
    ram_target: float = 50.0
):
    """Main function to collect ArXiv papers using RAM-efficient batch collection.
    
    Args:
        output_dir: Directory to save output files
        max_papers: Maximum total papers to collect
        cache_file: Path to cache file (default: output_dir/arxiv_papers.jsonl)
        rate_limit_delay: Delay between API requests (seconds)
        batch_size: Papers per batch (default: 10 for RAM efficiency)
        ram_target: Target RAM percentage to stay below (default: 50%)
    """
    if cache_file is None:
        cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
    
    # Use the efficient collector
    collect_arxiv_efficient(
        output_dir=output_dir,
        total_target=max_papers,
        batch_size=batch_size,
        ram_target=ram_target,
        rate_limit=rate_limit_delay
    )


# PDF Extraction Constants
MAX_PAGES = 6  # Extract first 6 pages
MAX_CHARS = 12000  # Maximum characters per paper
PDF_RATE_LIMIT = 0.4  # Seconds between PDF downloads
CHECKPOINT_INTERVAL_PDF = 100  # Save checkpoint every 100 papers


def extract_text_from_pdf_url(pdf_url: str, max_pages: int = MAX_PAGES, max_chars: int = MAX_CHARS) -> Optional[str]:
    """Extract text from PDF URL, limiting to first N pages and max characters.
    
    Memory-efficient: streams PDF download and processes page-by-page without
    keeping full PDF in RAM.
    
    Args:
        pdf_url: URL to ArXiv PDF
        max_pages: Maximum number of pages to extract (default: 6)
        max_chars: Maximum characters to extract (default: 12000)
        
    Returns:
        Extracted text, or None if extraction failed
    """
    if not PDF_AVAILABLE or not REQUESTS_AVAILABLE:
        return None
    
    try:
        # Download PDF with timeout and streaming
        response = requests.get(pdf_url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Stream PDF to BytesIO (memory-efficient)
        from io import BytesIO
        pdf_buffer = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            pdf_buffer.write(chunk)
        pdf_buffer.seek(0)
        
        # Extract text based on available library
        text_parts = []
        total_chars = 0
        
        if USE_PDFPLUMBER:
            import pdfplumber
            with pdfplumber.open(pdf_buffer) as pdf:
                for page_num, page in enumerate(pdf.pages[:max_pages], 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                        total_chars += len(page_text)
                        # Stop if we've reached max_chars
                        if total_chars >= max_chars:
                            break
        else:
            # Use PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_buffer)
            for page_num in range(min(max_pages, len(pdf_reader.pages))):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    total_chars += len(page_text)
                    # Stop if we've reached max_chars
                    if total_chars >= max_chars:
                        break
        
        # Combine text parts
        full_text = '\n\n'.join(text_parts)
        
        # Truncate to max_chars if needed (at word boundary)
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars].rsplit(' ', 1)[0] + '...'
        
        # Clear buffer from memory
        pdf_buffer.close()
        
        return full_text.strip()
        
    except requests.exceptions.RequestException as e:
        # Network errors
        return None
    except Exception as e:
        # Invalid PDFs or other errors
        return None


def load_processed_ids(output_dir: str) -> Set[str]:
    """Load set of already processed ArXiv IDs from output directory and checkpoint.
    
    Args:
        output_dir: Directory containing extracted text files
        
    Returns:
        Set of processed ArXiv IDs
    """
    processed = set()
    
    # Check existing .txt files
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.txt'):
                arxiv_id = filename[:-4]  # Remove .txt extension
                processed.add(arxiv_id)
    
    # Check checkpoint file
    checkpoint_file = os.path.join(output_dir, 'pdf_extraction_checkpoint.json')
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed.update(checkpoint.get('processed_ids', []))
        except Exception:
            pass
    
    return processed


def save_checkpoint(output_dir: str, processed_ids: Set[str], stats: Dict):
    """Save extraction checkpoint.
    
    Args:
        output_dir: Output directory
        processed_ids: Set of processed ArXiv IDs
        stats: Statistics dictionary
    """
    checkpoint_file = os.path.join(output_dir, 'pdf_extraction_checkpoint.json')
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'processed_ids': list(processed_ids),
        'stats': stats,
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def process_single_paper(paper: Dict, output_dir: str, rate_limit_delay: float) -> tuple[str, bool, Optional[str]]:
    """Process a single paper: download PDF and extract text.
    
    Args:
        paper: Paper dictionary from JSONL
        output_dir: Output directory for text files
        rate_limit_delay: Delay between requests (seconds)
        
    Returns:
        Tuple of (arxiv_id, success, error_message)
    """
    arxiv_id = paper.get('id', '')
    pdf_url = paper.get('pdf_url', '')
    
    if not arxiv_id or not pdf_url:
        return arxiv_id, False, "Missing ID or PDF URL"
    
    # Rate limiting
    time.sleep(rate_limit_delay)
    
    # Extract text
    text = extract_text_from_pdf_url(pdf_url, max_pages=MAX_PAGES, max_chars=MAX_CHARS)
    
    if text is None or not text.strip():
        return arxiv_id, False, "Failed to extract text"
    
    # Write to file
    output_file = os.path.join(output_dir, f"{arxiv_id}.txt")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return arxiv_id, True, None
    except Exception as e:
        return arxiv_id, False, f"File write error: {str(e)}"


def extract_pdf_texts(
    input_jsonl: str,
    output_dir: str,
    num_workers: int = 3,
    rate_limit_delay: float = PDF_RATE_LIMIT
):
    """Extract text from ArXiv PDFs using streaming and parallel processing.
    
    Memory-efficient: streams PDFs, writes text files immediately, doesn't keep
    PDFs or full text in RAM. Handles errors gracefully and supports resume.
    
    Args:
        input_jsonl: Path to input JSONL file with paper metadata
        output_dir: Directory to save extracted text files
        num_workers: Number of worker threads (2-4 recommended for Colab)
        rate_limit_delay: Delay between PDF downloads (default: 0.4 seconds)
    """
    if not PDF_AVAILABLE:
        print("❌ Error: PDF library not available.")
        print("   Install with: pip install PyPDF2 or pip install pdfplumber")
        return
    
    if not REQUESTS_AVAILABLE:
        print("❌ Error: requests package not available.")
        print("   Install with: pip install requests")
        return
    
    # Validate num_workers
    if num_workers < 2 or num_workers > 4:
        print(f"⚠️  Warning: num_workers={num_workers} not in recommended range (2-4)")
        print("   Adjusting to 3...")
        num_workers = 3
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load already processed IDs (from files and checkpoint)
    processed_ids = load_processed_ids(output_dir)
    print(f"📖 Found {len(processed_ids)} already processed papers")
    
    # Load checkpoint if exists
    checkpoint_file = os.path.join(output_dir, 'pdf_extraction_checkpoint.json')
    checkpoint_stats = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                checkpoint_stats = checkpoint_data.get('stats', {})
                print(f"📖 Resuming from checkpoint: {checkpoint_stats.get('success', 0)} successful, {checkpoint_stats.get('failed', 0)} failed")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
    
    # Load papers from JSONL (stream, don't load all in memory)
    print(f"📚 Loading papers from {input_jsonl}...")
    papers_to_process = []
    total_papers = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                paper = json.loads(line)
                arxiv_id = paper.get('id', '')
                
                if not arxiv_id:
                    continue
                
                total_papers += 1
                
                # Skip if already processed
                if arxiv_id in processed_ids:
                    continue
                
                papers_to_process.append(paper)
                
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Invalid JSON on line {line_num}")
                continue
    
    print(f"📊 Total papers in file: {total_papers}")
    print(f"📊 Already processed: {len(processed_ids)}")
    print(f"📊 Remaining to process: {len(papers_to_process)}")
    print()
    
    if not papers_to_process:
        print("✅ All papers already processed!")
        return
    
    # Statistics
    stats = {
        'total': len(papers_to_process),
        'success': 0,
        'failed': 0,
        'errors': defaultdict(int)
    }
    
    # Process papers with thread pool
    print(f"🚀 Starting extraction with {num_workers} workers...")
    print(f"⏱️  Rate limit: {1.0/rate_limit_delay:.1f} requests/sec")
    print()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_paper = {
            executor.submit(process_single_paper, paper, output_dir, rate_limit_delay): paper
            for paper in papers_to_process
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_paper):
            completed += 1
            paper = future_to_paper[future]
            
            try:
                arxiv_id, success, error_msg = future.result()
                
                if success:
                    stats['success'] += 1
                    processed_ids.add(arxiv_id)
                else:
                    stats['failed'] += 1
                    if error_msg:
                        stats['errors'][error_msg] += 1
                
                # Log progress
                if completed % 50 == 0:
                    print(f"   📊 Progress: {completed}/{len(papers_to_process)} "
                          f"(✓ {stats['success']} success, ✗ {stats['failed']} failed)")
                
                # Checkpoint
                if completed % CHECKPOINT_INTERVAL_PDF == 0:
                    save_checkpoint(output_dir, processed_ids, stats)
                    print(f"   💾 Checkpoint saved: {completed} papers processed")
            
            except Exception as e:
                stats['failed'] += 1
                stats['errors'][f"Exception: {str(e)}"] += 1
                print(f"   ⚠️  Unexpected error processing paper: {e}")
    
    # Final checkpoint
    save_checkpoint(output_dir, processed_ids, stats)
    
    # Print summary
    print()
    print("=" * 60)
    print("✅ PDF Extraction Complete!")
    print("=" * 60)
    print(f"📊 Total processed: {stats['total']}")
    print(f"✅ Success: {stats['success']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📁 Output directory: {output_dir}")
    
    if stats['errors']:
        print(f"\n⚠️  Error breakdown:")
        for error, count in sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {error}: {count}")


# Domain Classification Constants
DOMAIN_KEYWORDS = {
    'neurodegeneration': [
        'alzheimer', 'parkinson', 'als', 'dementia', 'huntington', 
        'neurodegenerative', 'tau', 'amyloid', 'lewy body'
    ],
    'neuroscience': [
        'brain', 'neural', 'neuron', 'fmri', 'eeg', 'neuroimaging',
        'cortex', 'synapse', 'neurotransmitter', 'neural network'
    ],
    'medical_imaging': [
        'mri', 'ct scan', 'x-ray', 'ultrasound', 'pet scan', 'spect',
        'radiology', 'imaging', 'dicom', 'medical image'
    ],
    'clinical': [
        'patient', 'clinical', 'diagnosis', 'treatment', 'therapy',
        'symptom', 'disease', 'disorder', 'syndrome', 'prognosis'
    ],
    'drug_discovery': [
        'drug', 'molecule', 'protein', 'compound', 'pharmaceutical',
        'medication', 'therapeutic', 'biomarker', 'target'
    ],
}

NEURODEGENERATION_KEYWORDS = [
    'alzheimer', 'parkinson', 'als', 'dementia', 'huntington',
    'neurodegenerative', 'tau', 'amyloid', 'lewy body'
]

MAX_TEXT_LENGTH = 12000  # Maximum characters per paper
PREPROCESS_WORKERS = 4  # Number of parallel workers


def preprocess_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Preprocess text: remove URLs, emails, citations, normalize whitespace.
    
    Preserves medical terminology (doesn't lowercase disease names).
    
    Args:
        text: Raw text from PDF
        max_length: Maximum length to keep
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Remove URLs (http://, https://, www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove citation references like [1], [2-5], (Smith et al., 2020), etc.
    # Pattern: [numbers], (Author et al., year), [Author, year]
    text = re.sub(r'\[[\d\s,-]+\]', '', text)  # [1], [2-5], etc.
    text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s+\d{4}\)', '', text)  # (Author et al., 2020)
    text = re.sub(r'\[[A-Z][a-z]+(?:\s+et\s+al\.)?,?\s+\d{4}\]', '', text)  # [Author, 2020]
    
    # Remove arXiv references
    text = re.sub(r'arXiv:\d+\.\d+', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace: multiple spaces/tabs/newlines to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Truncate to max_length (try to break at word boundary)
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + '...'
    
    return text


def classify_domains(text: str) -> List[str]:
    """Classify text into domains using keyword matching.
    
    Args:
        text: Preprocessed text
        
    Returns:
        List of domain labels
    """
    if not text:
        return ['general_ml_health']
    
    text_lower = text.lower()
    domains = []
    
    # Check each domain
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                domains.append(domain)
                break  # Only need one match per domain
    
    # Default to general_ml_health if no domains found
    if not domains:
        domains = ['general_ml_health']
    
    return domains


def has_neurodegeneration(text: str) -> bool:
    """Check if text mentions neurodegeneration.
    
    Args:
        text: Preprocessed text
        
    Returns:
        True if neurodegeneration keywords found
    """
    if not text:
        return False
    
    text_lower = text.lower()
    for keyword in NEURODEGENERATION_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    
    return False


def process_single_paper_file(
    arxiv_id: str,
    text_file_path: str,
    metadata: Dict,
    output_file: str,
    lock: threading.Lock
) -> tuple[str, bool, Optional[str]]:
    """Process a single paper: read text, preprocess, classify, write to JSONL.
    
    Args:
        arxiv_id: ArXiv ID
        text_file_path: Path to text file
        metadata: Paper metadata from JSONL
        output_file: Output JSONL file path
        lock: Thread lock for file writing
        
    Returns:
        Tuple of (arxiv_id, success, error_message)
    """
    try:
        # Read text file
        if not os.path.exists(text_file_path):
            return arxiv_id, False, "Text file not found"
        
        with open(text_file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        if not raw_text.strip():
            return arxiv_id, False, "Empty text file"
        
        # Preprocess text
        processed_text = preprocess_text(raw_text, max_length=MAX_TEXT_LENGTH)
        
        if not processed_text.strip():
            return arxiv_id, False, "Text became empty after preprocessing"
        
        # Classify domains
        domains = classify_domains(processed_text)
        
        # Check for neurodegeneration
        has_nd = has_neurodegeneration(processed_text)
        
        # Get year from metadata
        year = metadata.get('year', None)
        
        # Create output record
        output_record = {
            'arxiv_id': arxiv_id,
            'text': processed_text,
            'domains': domains,
            'year': year,
            'has_neurodegeneration': has_nd,
        }
        
        # Write to JSONL (thread-safe)
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        
        return arxiv_id, True, None
        
    except Exception as e:
        return arxiv_id, False, f"Error: {str(e)}"


def preprocess_and_classify(
    metadata_jsonl: str,
    text_dir: str,
    output_jsonl: str,
    num_workers: int = PREPROCESS_WORKERS
):
    """Preprocess text files and classify domains using parallel processing.
    
    Args:
        metadata_jsonl: Path to metadata JSONL file
        text_dir: Directory containing extracted text files
        output_jsonl: Output JSONL file path
        num_workers: Number of parallel workers
    """
    print("=" * 60)
    print("🔬 Domain Classifier & Text Preprocessor")
    print("=" * 60)
    print(f"📁 Metadata file: {metadata_jsonl}")
    print(f"📁 Text directory: {text_dir}")
    print(f"📁 Output file: {output_jsonl}")
    print(f"👷 Workers: {num_workers}")
    print()
    
    # Load metadata
    print("📚 Loading metadata...")
    metadata_by_id = {}
    total_papers = 0
    
    with open(metadata_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                paper = json.loads(line)
                arxiv_id = paper.get('id', '')
                
                if not arxiv_id:
                    continue
                
                metadata_by_id[arxiv_id] = paper
                total_papers += 1
                
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Invalid JSON on line {line_num}")
                continue
    
    print(f"   ✅ Loaded {total_papers} papers from metadata")
    
    # Find text files
    print(f"\n📂 Scanning text directory...")
    text_files = {}
    if os.path.exists(text_dir):
        for filename in os.listdir(text_dir):
            if filename.endswith('.txt'):
                arxiv_id = filename[:-4]  # Remove .txt extension
                text_files[arxiv_id] = os.path.join(text_dir, filename)
    
    print(f"   ✅ Found {len(text_files)} text files")
    
    # Match metadata with text files
    papers_to_process = []
    for arxiv_id, metadata in metadata_by_id.items():
        if arxiv_id in text_files:
            papers_to_process.append((arxiv_id, text_files[arxiv_id], metadata))
    
    print(f"   ✅ {len(papers_to_process)} papers ready to process")
    print()
    
    if not papers_to_process:
        print("❌ No papers to process!")
        return
    
    # Clear output file if it exists
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    
    # Statistics
    stats = {
        'total': len(papers_to_process),
        'success': 0,
        'failed': 0,
        'errors': defaultdict(int),
        'domain_counts': defaultdict(int),
        'neurodegeneration_count': 0,
    }
    
    # Thread lock for file writing
    file_lock = threading.Lock()
    
    # Process papers with thread pool
    print(f"🚀 Starting preprocessing and classification...")
    print()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_paper = {
            executor.submit(
                process_single_paper_file,
                arxiv_id,
                text_path,
                metadata,
                output_jsonl,
                file_lock
            ): (arxiv_id, text_path, metadata)
            for arxiv_id, text_path, metadata in papers_to_process
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_paper):
            completed += 1
            arxiv_id, text_path, metadata = future_to_paper[future]
            
            try:
                paper_id, success, error_msg = future.result()
                
                if success:
                    stats['success'] += 1
                    
                    # Update domain counts (need to read back from file or track separately)
                    # For efficiency, we'll do a final pass at the end
                else:
                    stats['failed'] += 1
                    if error_msg:
                        stats['errors'][error_msg] += 1
                
                # Log progress
                if completed % 500 == 0:
                    print(f"   📊 Progress: {completed}/{len(papers_to_process)} "
                          f"(✓ {stats['success']} success, ✗ {stats['failed']} failed)")
            
            except Exception as e:
                stats['failed'] += 1
                stats['errors'][f"Exception: {str(e)}"] += 1
                print(f"   ⚠️  Unexpected error processing {arxiv_id}: {e}")
    
    # Final statistics pass (read output file to get domain counts)
    print(f"\n📊 Computing final statistics...")
    domain_counts = defaultdict(int)
    neurodegeneration_count = 0
    
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        for domain in record.get('domains', []):
                            domain_counts[domain] += 1
                        if record.get('has_neurodegeneration', False):
                            neurodegeneration_count += 1
                    except:
                        continue
    
    stats['domain_counts'] = dict(domain_counts)
    stats['neurodegeneration_count'] = neurodegeneration_count
    
    # Print summary
    print()
    print("=" * 60)
    print("✅ Preprocessing & Classification Complete!")
    print("=" * 60)
    print(f"📊 Total processed: {stats['total']}")
    print(f"✅ Success: {stats['success']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📁 Output file: {output_jsonl}")
    
    print(f"\n📊 Domain distribution:")
    for domain, count in sorted(stats['domain_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {domain}: {count} papers")
    
    print(f"\n🧠 Neurodegeneration papers: {stats['neurodegeneration_count']}")
    
    if stats['errors']:
        print(f"\n⚠️  Error breakdown:")
        for error, count in sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {error}: {count}")
    
    # Estimate file size
    if os.path.exists(output_jsonl):
        file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
        print(f"\n💾 Output file size: {file_size_mb:.2f} MB")


# NeMo Curator Pipeline
def create_healthcare_text_cleaner():
    """Create NeMo Curator text cleaner for healthcare papers.
    
    Returns:
        DocumentModifier for text cleaning
    """
    if not NEMO_CURATOR_AVAILABLE:
        return None
    
    # Custom modifier that preserves medical terminology
    # Extends: nemo_curator.modifiers.DocumentModifier
    _HealthcareTextCleanerBase = DocumentModifier if (NEMO_CURATOR_AVAILABLE and DocumentModifier is not None) else object
    class HealthcareTextCleaner(_HealthcareTextCleanerBase):
        def __init__(self):
            if NEMO_CURATOR_AVAILABLE and DocumentModifier is not None:
                try:
                    super().__init__()
                except:
                    pass  # If super() fails, continue without it
            # Medical terms to preserve (don't lowercase)
            self.medical_terms = {
                'Alzheimer', 'Parkinson', 'ALS', 'Dementia', 'Huntington',
                'MRI', 'EEG', 'fMRI', 'CT', 'PET', 'SPECT',
                'Alzheimer\'s', 'Parkinson\'s'
            }
        
        def modify_document(self, document):
            """Clean text while preserving medical terminology."""
            text = document.get('text', '')
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove emails
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove citation markers (e.g., [1], (Smith et al., 2020))
            text = re.sub(r'\[[\d,\s-]+\]', '', text)
            text = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.?[^)]*\)', '', text)
            
            # Normalize quotation marks
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Unicode normalization
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
            
            # Preserve medical terms (temporary replacement)
            replacements = {}
            for i, term in enumerate(self.medical_terms):
                placeholder = f'__MEDICAL_TERM_{i}__'
                text = text.replace(term, placeholder)
                replacements[placeholder] = term
            
            # Lowercase (except medical terms)
            text = text.lower()
            
            # Restore medical terms
            for placeholder, term in replacements.items():
                text = text.replace(placeholder, term)
            
            document['text'] = text
            return document
    
    return HealthcareTextCleaner()


class HealthcareTextModifier:
    """Custom healthcare text modifier for post-NeMo Curator processing.
    
    Extends NeMo Curator DocumentModifier interface to:
    - Extract section boundaries (Abstract, Introduction, Methods, Results, Discussion)
    - Preserve scientific abbreviations
    - Keep medical terminology intact
    - Remove page numbers, headers/footers, duplicate lines
    - Normalize citations format
    """
    
    def __init__(self):
        """Initialize healthcare text modifier."""
        # Scientific abbreviations to preserve
        self.scientific_abbreviations = {
            'MMSE', 'MRI', 'EEG', 'fMRI', 'PET', 'CSF', 'CT', 'SPECT', 'DICOM',
            'AD', 'PD', 'ALS', 'MCI', 'CSF', 'Aβ', 'tau', 'APOE', 'SNP',
            'ROI', 'VOI', 'FA', 'MD', 'DTI', 'BOLD', 'T1', 'T2', 'FLAIR'
        }
        
        # Section markers
        self.section_markers = {
            'abstract': [r'^abstract\s*:', r'^abstract\s*$', r'^\d+\.\s*abstract'],
            'introduction': [r'^introduction\s*:', r'^introduction\s*$', r'^\d+\.\s*introduction', r'^1\.\s*introduction'],
            'methods': [r'^methods?\s*:', r'^methods?\s*$', r'^methodology\s*:', r'^\d+\.\s*methods?', r'^2\.\s*methods?'],
            'results': [r'^results?\s*:', r'^results?\s*$', r'^\d+\.\s*results?', r'^3\.\s*results?'],
            'discussion': [r'^discussion\s*:', r'^discussion\s*$', r'^conclusion\s*:', r'^\d+\.\s*discussion', r'^4\.\s*discussion'],
        }
        
        # Medical terms patterns (for detection)
        self.medical_term_patterns = {
            'neurodegeneration': r'\b(alzheimer|parkinson|als|mci|dementia|tau|amyloid|huntington|neurodegenerative)\w*\b',
            'neuroscience': r'\b(brain|neural|neuron|fmri|eeg|synapse|cortex|neurotransmitter)\w*\b',
            'medical_imaging': r'\b(mri|ct|ultrasound|xray|x-ray|pet|spect|radiology|dicom|segmentation)\w*\b',
            'clinical': r'\b(patient|clinical|diagnosis|prognosis|treatment|symptom|disease|disorder|syndrome|therapy)\w*\b',
            'drug_discovery': r'\b(drug|molecule|protein|compound|binding|pharmaceutical|medication|therapeutic|biomarker|target)\w*\b',
        }
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract section boundaries from text.
        
        Args:
            text: Full text document
            
        Returns:
            Dictionary with section names as keys and section text as values
        """
        sections = {}
        lines = text.split('\n')
        
        # Find section boundaries
        current_section = None
        current_section_text = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_section_text.append('')
                continue
            
            # Check if this line is a section header
            detected_section = None
            for section_name, patterns in self.section_markers.items():
                for pattern in patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        detected_section = section_name
                        break
                if detected_section:
                    break
            
            if detected_section:
                # Save previous section
                if current_section and current_section_text:
                    sections[current_section] = '\n'.join(current_section_text).strip()
                
                # Start new section
                current_section = detected_section
                current_section_text = []
            else:
                if current_section:
                    current_section_text.append(line)
                elif not sections:  # Before any section detected, assume abstract/intro
                    if 'abstract' not in sections:
                        sections['abstract'] = ''
                    sections['abstract'] += line + '\n'
        
        # Save last section
        if current_section and current_section_text:
            sections[current_section] = '\n'.join(current_section_text).strip()
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def detect_medical_terms(self, text: str) -> List[str]:
        """Detect medical terms in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected medical terms
        """
        detected_terms = set()
        text_lower = text.lower()
        
        for domain, pattern in self.medical_term_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            detected_terms.update(matches)
        
        return sorted(list(detected_terms))
    
    def clean_text(self, text: str) -> str:
        """Clean text: remove page numbers, headers/footers, duplicate lines.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines (but keep one between paragraphs)
            if not line_stripped:
                if cleaned_lines and cleaned_lines[-1]:
                    cleaned_lines.append('')
                continue
            
            # Skip page numbers (standalone numbers, especially at start/end)
            if re.match(r'^\d+$', line_stripped):
                continue
            
            # Skip headers/footers (repeated lines, very short lines at start/end)
            if len(line_stripped) < 3:
                continue
            
            # Skip duplicate lines
            line_normalized = line_stripped.lower()
            if line_normalized in seen_lines:
                continue
            seen_lines.add(line_normalized)
            
            # Remove citation markers but keep content
            line_cleaned = re.sub(r'\[[\d,\s-]+\]', '', line_stripped)
            line_cleaned = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.?[^)]*\)', '', line_cleaned)
            line_cleaned = line_cleaned.strip()
            
            if line_cleaned:
                cleaned_lines.append(line_cleaned)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_citations(self, text: str) -> str:
        """Normalize citation format (optional).
        
        Args:
            text: Text with citations
            
        Returns:
            Text with normalized citations
        """
        # Convert [1,2,3] to [1, 2, 3]
        text = re.sub(r'\[(\d+),(\d+)', r'[\1, \2', text)
        # Convert (Smith et al., 2020) to [Smith et al., 2020]
        text = re.sub(r'\(([A-Z][a-z]+\s+et\s+al\.?[^)]*)\)', r'[\1]', text)
        return text
    
    def modify_document(self, document: Dict) -> Dict:
        """Modify document with healthcare-specific processing.
        
        Args:
            document: Document dictionary with 'text' field
            
        Returns:
            Modified document with processed text and metadata
        """
        text = document.get('text', '')
        if not text:
            return document
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract sections
        sections = self.extract_sections(cleaned_text)
        
        # Detect medical terms
        medical_terms = self.detect_medical_terms(cleaned_text)
        
        # Normalize citations (optional)
        cleaned_text = self.normalize_citations(cleaned_text)
        
        # Truncate to 12k chars
        max_chars = 12000
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars].rsplit(' ', 1)[0] + '...'
        
        # Estimate token count (rough: ~4 chars per token)
        token_count_estimate = len(cleaned_text) // 4
        
        # Update document
        document['text'] = cleaned_text
        document['sections'] = sections
        document['medical_terms_detected'] = medical_terms
        document['token_count_estimate'] = token_count_estimate
        
        return document


# Define base class for HealthcareDomainFilter
_HealthcareDomainFilterBase = DocumentFilter if (NEMO_CURATOR_AVAILABLE and DocumentFilter is not None) else object

class HealthcareDomainFilter(_HealthcareDomainFilterBase):
    """Custom domain classifier for healthcare papers extending NeMo Curator DocumentFilter interface.
    
    Scores documents based on healthcare+ML domain relevance and assigns domain tags.
    Compatible with NeMo Curator's ScoreFilter.
    
    Extends: nemo_curator.filters.DocumentFilter (if available)
    """
    
    def __init__(self):
        """Initialize domain filter with healthcare vocabulary."""
        if NEMO_CURATOR_AVAILABLE and DocumentFilter is not None:
            try:
                super().__init__()
            except:
                pass  # If super() fails, continue without it
        # Domain keywords with specific terms
        self.domain_keywords = {
            'neurodegeneration': [
                'alzheimer', 'parkinson', 'als', 'mci', 'tau', 'amyloid',
                'dementia', 'huntington', 'neurodegenerative', 'lewy body'
            ],
            'neuroscience': [
                'brain', 'neural', 'neuroimaging', 'fmri', 'eeg', 'synapse',
                'neuron', 'cortex', 'neurotransmitter', 'neural network'
            ],
            'medical_imaging': [
                'mri', 'ct', 'ultrasound', 'xray', 'segmentation',
                'x-ray', 'pet scan', 'spect', 'radiology', 'dicom', 'medical image'
            ],
            'clinical': [
                'patient', 'clinical', 'diagnosis', 'prognosis', 'treatment',
                'symptom', 'disease', 'disorder', 'syndrome', 'therapy'
            ],
            'drug_discovery': [
                'drug', 'molecule', 'protein', 'compound', 'binding',
                'pharmaceutical', 'medication', 'therapeutic', 'biomarker', 'target'
            ],
        }
        
        # Medical terms that indicate high relevance
        self.medical_terms = set()
        for keywords in self.domain_keywords.values():
            self.medical_terms.update(keywords)
    
    def score_document(self, document: Dict) -> Dict[str, float]:
        """Score document based on domain relevance.
        
        Args:
            document: Document dictionary with 'text' field
            
        Returns:
            Dictionary with domain scores and overall relevance score
        """
        text = document.get('text', '').lower()
        if not text:
            return {
                'neurodegeneration': 0.0,
                'neuroscience': 0.0,
                'medical_imaging': 0.0,
                'clinical': 0.0,
                'drug_discovery': 0.0,
                'relevance': 0.0
            }
        
        # Calculate domain scores
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            # Score: normalized by number of keywords (0-1)
            score = min(matches / len(keywords), 1.0)
            domain_scores[domain] = score
        
        # Calculate overall relevance
        # Base score: average of domain scores
        base_score = sum(domain_scores.values()) / len(domain_scores) if domain_scores else 0.0
        
        # Boost for multiple domains (multi-domain papers are more relevant)
        active_domains = sum(1 for score in domain_scores.values() if score > 0.1)
        if active_domains > 1:
            base_score *= (1.0 + 0.1 * (active_domains - 1))  # 10% boost per additional domain
        
        # Boost for paper length (longer papers tend to be more substantial)
        word_count = len(text.split())
        if word_count > 500:
            base_score *= 1.1  # 10% boost for longer papers
        elif word_count < 200:
            base_score *= 0.9  # Penalty for very short papers
        
        # Count medical terms presence
        medical_term_count = sum(1 for term in self.medical_terms if term in text)
        medical_boost = min(medical_term_count / 20.0, 0.2)  # Up to 20% boost
        base_score += medical_boost
        
        # Normalize to 0-1 range
        relevance = min(base_score, 1.0)
        
        # Add relevance to scores
        domain_scores['relevance'] = relevance
        
        return domain_scores
    
    def filter_document(self, document: Dict, min_relevance: float = 0.4) -> bool:
        """Filter document based on relevance score.
        
        Args:
            document: Document dictionary
            min_relevance: Minimum relevance score to keep (default: 0.4)
            
        Returns:
            True if document should be kept, False otherwise
        """
        scores = self.score_document(document)
        relevance = scores.get('relevance', 0.0)
        
        # Store scores in document
        document['domain_scores'] = scores
        document['relevance_score'] = relevance
        
        # Assign domain tags (domains with score > 0.2)
        detected_domains = [
            domain for domain, score in scores.items()
            if domain != 'relevance' and score > 0.2
        ]
        document['domains'] = detected_domains if detected_domains else ['general_ml_health']
        
        return relevance >= min_relevance


def create_domain_relevance_filter(min_score: float = 0.5):
    """Create domain-specific relevance filter (legacy function for compatibility).
    
    Args:
        min_score: Minimum relevance score to keep
        
    Returns:
        Custom filter function
    """
    filter_instance = HealthcareDomainFilter()
    
    def filter_document(document):
        """Filter document based on domain relevance."""
        return filter_instance.filter_document(document, min_relevance=min_score)
    
    return filter_document


# Language detection (optional)
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    try:
        from textblob import TextBlob
        LANGDETECT_AVAILABLE = True
        def detect(text):
            try:
                blob = TextBlob(text[:1000])  # Use first 1000 chars for speed
                return blob.detect_language()
            except:
                return 'en'  # Default to English if detection fails
    except ImportError:
        LANGDETECT_AVAILABLE = False
        def detect(text):
            return 'en'  # Default to English if langdetect not available


class HealthcareFilterStage(ProcessingStage if (NEMO_CURATOR_AVAILABLE and ProcessingStage_AVAILABLE) else object):
    """Custom NeMo Curator ProcessingStage for healthcare document filtering and classification.
    
    Extends: nemo_curator.stages.ProcessingStage
    
    Input: JSONL from ArxivDownloadExtractStage (raw ArXiv text)
    Output: JSONL with curated documents (text cleaned, quality filtered, domain classified)
    """
    
    def __init__(self):
        """Initialize healthcare filter stage."""
        if NEMO_CURATOR_AVAILABLE and ProcessingStage_AVAILABLE:
            super().__init__()
        
        # Domain keywords
        self.healthcare_domains = {
            'neurodegeneration': ['alzheimer', 'parkinson', 'als', 'dementia', 'mci', 'tau', 'amyloid'],
            'neuroscience': ['brain', 'neural', 'neuroimaging', 'fmri', 'eeg', 'synapse'],
            'medical_imaging': ['mri', 'ct', 'ultrasound', 'xray', 'segmentation'],
            'clinical': ['patient', 'clinical', 'diagnosis', 'prognosis', 'treatment'],
            'drug_discovery': ['drug', 'molecule', 'protein', 'compound', 'binding'],
        }
        
        # ML method keywords
        self.ml_keywords = [
            'machine learning', 'deep learning', 'neural network', 'transformer',
            'lstm', 'prediction', 'classification', 'regression', 'clustering',
            'supervised', 'unsupervised', 'reinforcement learning', 'cnn', 'rnn',
            'attention', 'bert', 'gpt', 'autoencoder', 'gan'
        ]
        
        # Statistics
        self.stats = {
            'total_in': 0,
            'passed_quality': 0,
            'passed_domain': 0,
            'total_out': 0
        }
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX artifacts from text.
        
        Args:
            text: Raw text with LaTeX commands
            
        Returns:
            Cleaned text
        """
        # Remove LaTeX commands: \command{...} or \command
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \command{content} -> content
        text = re.sub(r'\\[a-zA-Z]+\s+', ' ', text)  # \command -> space
        
        # Remove math blocks: $$...$$ or $...$
        text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
        text = re.sub(r'\$[^$]+\$', ' ', text)
        
        # Remove LaTeX environments: \begin{env}...\end{env}
        text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', ' ', text, flags=re.DOTALL)
        
        # Remove braces: {{content}} -> content
        text = re.sub(r'\{\{([^}]*)\}\}', r'\1', text)
        text = re.sub(r'\{([^}]*)\}', r'\1', text)
        
        # Remove special LaTeX characters
        text = text.replace('\\', ' ')
        text = text.replace('&', ' ')
        text = text.replace('%', ' ')
        
        return text
    
    def _remove_references(self, text: str) -> str:
        """Remove references section from text.
        
        Args:
            text: Full document text
            
        Returns:
            Text with references section removed
        """
        # Find references section
        ref_patterns = [
            r'(?i)\n\s*(references|bibliography)\s*\n.*$',
            r'(?i)\n\s*references\s*\n.*$',
            r'(?i)\n\s*bibliography\s*\n.*$',
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                text = text[:match.start()]
                break
        
        return text
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Number of sentences
        """
        # Simple sentence detection: count periods, exclamation, question marks
        # followed by space or newline
        sentences = re.findall(r'[.!?]+(?:\s+|$)', text)
        return len(sentences)
    
    def _check_quality(self, text: str) -> tuple[bool, float]:
        """Check document quality.
        
        Args:
            text: Document text
            
        Returns:
            (passed, quality_score) tuple
        """
        if not text or len(text.strip()) < 100:
            return False, 0.0
        
        # Word count check: 100-5000 tokens (estimate: 1 token = 4 chars)
        char_count = len(text)
        token_estimate = char_count // 4
        if not (100 <= token_estimate <= 5000):
            return False, 0.0
        
        # Alphanumeric ratio: > 40%
        alnum_chars = sum(1 for c in text if c.isalnum())
        alnum_ratio = alnum_chars / len(text) if len(text) > 0 else 0
        if alnum_ratio <= 0.4:
            return False, 0.0
        
        # Min unique sentences: 5
        sentence_count = self._count_sentences(text)
        if sentence_count < 5:
            return False, 0.0
        
        # Language detection (optional)
        if LANGDETECT_AVAILABLE:
            try:
                lang = detect(text[:1000])  # Use first 1000 chars for speed
                if lang != 'en':
                    return False, 0.0
            except:
                pass  # If detection fails, assume English
        
        # Quality score: combination of factors
        quality_score = (
            0.3 * min(token_estimate / 2000, 1.0) +  # Token count (normalized)
            0.3 * alnum_ratio +  # Alphanumeric ratio
            0.2 * min(sentence_count / 50, 1.0) +  # Sentence count (normalized)
            0.2 * (1.0 if LANGDETECT_AVAILABLE else 0.5)  # Language check
        )
        
        return True, quality_score
    
    def _classify_domains(self, text: str) -> tuple[list[str], float, float]:
        """Classify document domains and compute scores.
        
        Args:
            text: Document text (lowercase)
            
        Returns:
            (domains, domain_score, ml_score) tuple
        """
        text_lower = text.lower()
        
        # Check healthcare domains
        detected_domains = []
        domain_matches = 0
        total_domain_keywords = 0
        
        for domain, keywords in self.healthcare_domains.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                detected_domains.append(domain)
                domain_matches += matches
            total_domain_keywords += len(keywords)
        
        # Check ML methods
        ml_matches = sum(1 for keyword in self.ml_keywords if keyword in text_lower)
        
        # Compute scores
        domain_score = min(domain_matches / max(total_domain_keywords, 1), 1.0)
        ml_score = min(ml_matches / max(len(self.ml_keywords), 1), 1.0)
        
        # Combined relevance score
        relevance_score = (domain_score * 0.6 + ml_score * 0.4)
        
        return detected_domains, domain_score, ml_score
    
    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from filename if possible.
        
        Args:
            filename: Source tar filename
            
        Returns:
            Year as integer, or None if not found
        """
        # ArXiv filenames often contain dates: YYYYMMDD
        year_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
        if year_match:
            return int(year_match.group(1))
        
        # Try YYYY pattern
        year_match = re.search(r'\b(20\d{2})\b', filename)
        if year_match:
            year = int(year_match.group(1))
            if 2015 <= year <= 2024:
                return year
        
        return None
    
    def _process_document(self, doc: Dict) -> Dict:
        """Process a single document.
        
        Args:
            doc: Input document with 'text' and 'file_name' fields
            
        Returns:
            Processed document or None if filtered out
        """
        self.stats['total_in'] += 1
        
        text = doc.get('text', '')
        filename = doc.get('file_name', '')
        
        if not text:
            return None
        
        # Step 1: Text cleaning
        # Remove LaTeX artifacts
        cleaned_text = self._clean_latex(text)
        
        # Remove references section
        cleaned_text = self._remove_references(cleaned_text)
        
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # Truncate to 12k chars
        max_chars = 12000
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars].rsplit(' ', 1)[0] + '...'
        
        # Step 2: Quality filters
        passed_quality, quality_score = self._check_quality(cleaned_text)
        if not passed_quality:
            return None
        
        self.stats['passed_quality'] += 1
        
        # Step 3: Domain classification
        domains, domain_score, ml_score = self._classify_domains(cleaned_text)
        
        # Check requirements:
        # - At least 1 healthcare domain keyword
        # - At least 2 ML method keywords
        # - Domain + ML relevance score > 0.6
        healthcare_keywords_found = len(domains) > 0
        ml_keywords_found = sum(1 for keyword in self.ml_keywords if keyword in cleaned_text.lower()) >= 2
        relevance_score = (domain_score * 0.6 + ml_score * 0.4)
        
        if not (healthcare_keywords_found and ml_keywords_found and relevance_score > 0.6):
            return None
        
        self.stats['passed_domain'] += 1
        self.stats['total_out'] += 1
        
        # Extract year from filename
        year = self._extract_year_from_filename(filename)
        
        # Estimate tokens
        token_estimate = len(cleaned_text) // 4
        
        # Build output document
        output_doc = {
            'text': cleaned_text,
            'file_name': filename,
            'domains': domains,
            'domain_score': round(domain_score, 3),
            'ml_score': round(ml_score, 3),
            'quality_score': round(quality_score, 3),
            'year': year,
            'token_estimate': token_estimate
        }
        
        return output_doc
    
    def __call__(self, dataset):
        """Process dataset (NeMo Curator interface).
        
        Args:
            dataset: Input dataset (DocumentDataset or iterable)
            
        Returns:
            Processed dataset
        """
        # Reset stats
        self.stats = {
            'total_in': 0,
            'passed_quality': 0,
            'passed_domain': 0,
            'total_out': 0
        }
        
        # Process documents
        processed_docs = []
        for doc in dataset:
            processed = self._process_document(doc)
            if processed:
                processed_docs.append(processed)
        
        # Log stats
        print(f"📊 HealthcareFilterStage Statistics:")
        print(f"   Total in: {self.stats['total_in']}")
        print(f"   Passed quality: {self.stats['passed_quality']}")
        print(f"   Passed domain: {self.stats['passed_domain']}")
        print(f"   Total out: {self.stats['total_out']}")
        
        return processed_docs


class HealthcareQualityFilterStage(ProcessingStage if (NEMO_CURATOR_AVAILABLE and ProcessingStage_AVAILABLE) else object):
    """Custom NeMo Curator ProcessingStage for deduplication and quality verification.
    
    Extends: nemo_curator.stages.ProcessingStage
    
    Input: JSONL from HealthcareFilterStage (domain-classified documents)
    Output: JSONL with duplicates removed and quality verified
    """
    
    def __init__(self):
        """Initialize quality filter stage."""
        if NEMO_CURATOR_AVAILABLE and ProcessingStage_AVAILABLE:
            super().__init__()
        
        # Statistics
        self.stats = {
            'total_in': 0,
            'duplicates_removed': 0,
            'unique_out': 0,
            'language_filter_out': 0,
            'quality_filter_out': 0
        }
        
        # Track seen hashes for deduplication
        self.seen_hashes = set()
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash of first 1000 + last 1000 chars.
        
        Args:
            text: Document text
            
        Returns:
            Hash string
        """
        import hashlib
        
        # Get first 1000 and last 1000 chars
        first_part = text[:1000] if len(text) > 1000 else text
        last_part = text[-1000:] if len(text) > 1000 else ''
        
        # Combine and hash
        combined = first_part + last_part
        hash_obj = hashlib.md5(combined.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _check_language(self, text: str) -> bool:
        """Check if text is English.
        
        Args:
            text: Document text
            
        Returns:
            True if English, False otherwise
        """
        if not LANGDETECT_AVAILABLE:
            return True  # Assume English if detection not available
        
        try:
            lang = detect(text[:1000])  # Use first 1000 chars for speed
            return lang == 'en'
        except:
            return True  # Default to English if detection fails
    
    def _check_content_quality(self, text: str) -> bool:
        """Check content quality.
        
        Args:
            text: Document text
            
        Returns:
            True if passes quality checks, False otherwise
        """
        lines = text.split('\n')
        if len(lines) == 0:
            return False
        
        # Check: >30% of lines are repeated
        line_counts = {}
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
        
        total_lines = len([l for l in lines if l.strip()])
        if total_lines == 0:
            return False
        
        repeated_lines = sum(1 for count in line_counts.values() if count > 1)
        repeated_ratio = repeated_lines / total_lines if total_lines > 0 else 0
        
        if repeated_ratio > 0.3:
            return False
        
        # Check: <5 sentences detected
        sentence_count = len(re.findall(r'[.!?]+(?:\s+|$)', text))
        if sentence_count < 5:
            return False
        
        # Check: <50 alphanumeric characters
        alnum_chars = sum(1 for c in text if c.isalnum())
        if alnum_chars < 50:
            return False
        
        return True
    
    def _process_document(self, doc: Dict) -> Dict:
        """Process a single document.
        
        Args:
            doc: Input document
            
        Returns:
            Processed document or None if filtered out
        """
        self.stats['total_in'] += 1
        
        text = doc.get('text', '')
        if not text:
            return None
        
        # Step 1: Fuzzy deduplication
        doc_hash = self._compute_hash(text)
        if doc_hash in self.seen_hashes:
            self.stats['duplicates_removed'] += 1
            return None
        
        self.seen_hashes.add(doc_hash)
        
        # Step 2: Language detection
        if not self._check_language(text):
            self.stats['language_filter_out'] += 1
            return None
        
        # Step 3: Content quality checks
        if not self._check_content_quality(text):
            self.stats['quality_filter_out'] += 1
            return None
        
        self.stats['unique_out'] += 1
        
        # Add hash to document
        output_doc = doc.copy()
        output_doc['hash'] = doc_hash
        
        return output_doc
    
    def __call__(self, dataset):
        """Process dataset (NeMo Curator interface).
        
        Args:
            dataset: Input dataset (DocumentDataset or iterable)
            
        Returns:
            Processed dataset
        """
        # Reset stats (but keep seen_hashes for cross-batch deduplication)
        self.stats = {
            'total_in': 0,
            'duplicates_removed': 0,
            'unique_out': 0,
            'language_filter_out': 0,
            'quality_filter_out': 0
        }
        
        # Process documents
        processed_docs = []
        for doc in dataset:
            processed = self._process_document(doc)
            if processed:
                processed_docs.append(processed)
        
        # Log stats
        print(f"📊 HealthcareQualityFilterStage Statistics:")
        print(f"   Total in: {self.stats['total_in']}")
        print(f"   Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"   Language filter out: {self.stats['language_filter_out']}")
        print(f"   Quality filter out: {self.stats['quality_filter_out']}")
        print(f"   Unique out: {self.stats['unique_out']}")
        print(f"   Retention rate: {self.stats['unique_out'] / max(self.stats['total_in'], 1) * 100:.1f}%")
        
        return processed_docs


class HealthcareJsonlWriter(Stage if (NEMO_CURATOR_AVAILABLE and Stage_AVAILABLE) else object):
    """Custom NeMo Curator writer stage to save filtered documents with proper formatting.
    
    Extends: nemo_curator.stages.Stage
    
    Input: JSONL from HealthcareQualityFilterStage (curated documents)
    Output: Single JSONL file with final curated dataset
    """
    
    def __init__(self, output_path: str = "./curated_dataset.jsonl"):
        """Initialize writer stage.
        
        Args:
            output_path: Path to output JSONL file
        """
        if NEMO_CURATOR_AVAILABLE and Stage_AVAILABLE:
            super().__init__()
        
        self.output_path = output_path
        self.stats = {
            'total_processed': 0,
            'domain_counts': defaultdict(int),
            'year_counts': defaultdict(int),
            'total_tokens': 0,
            'quality_scores': [],
            'log_interval': 1000
        }
    
    def _extract_arxiv_id(self, file_name: str) -> str:
        """Extract ArXiv ID from filename.
        
        Args:
            file_name: Source tar filename or document identifier
            
        Returns:
            ArXiv ID string, or None if not found
        """
        if not file_name:
            return None
        
        # ArXiv IDs are typically in format: YYMM.NNNNN or YYMM.NNNNNvN
        # Or extracted from paths like: arxiv/src/YYMM/YYMM.NNNNN.tar.gz
        arxiv_id_match = re.search(r'(\d{4})\.(\d{5})(?:v\d+)?', file_name)
        if arxiv_id_match:
            return f"{arxiv_id_match.group(1)}.{arxiv_id_match.group(2)}"
        
        # Try alternative format: YYMMNNNNN
        arxiv_id_match = re.search(r'(\d{2})(\d{2})(\d{5})', file_name)
        if arxiv_id_match:
            year_part = arxiv_id_match.group(1) + arxiv_id_match.group(2)
            num_part = arxiv_id_match.group(3)
            return f"{year_part}.{num_part}"
        
        # If no match, use filename as ID (sanitized)
        filename_base = os.path.basename(file_name)
        filename_base = re.sub(r'[^a-zA-Z0-9._-]', '_', filename_base)
        return filename_base if filename_base else None
    
    def _format_document(self, doc: Dict) -> Dict:
        """Format document for output.
        
        Args:
            doc: Input document from previous stage
            
        Returns:
            Formatted document
        """
        # Extract arxiv_id from file_name
        file_name = doc.get('file_name', '')
        arxiv_id = self._extract_arxiv_id(file_name)
        
        # Build formatted document
        formatted_doc = {
            'arxiv_id': arxiv_id,
            'text': doc.get('text', ''),
            'domains': doc.get('domains', []),
            'domain_score': doc.get('domain_score', 0.0),
            'ml_score': doc.get('ml_score', 0.0),
            'quality_score': doc.get('quality_score', 0.0),
            'year': doc.get('year'),
            'token_count': doc.get('token_estimate', doc.get('token_count', 0))
        }
        
        return formatted_doc
    
    def _update_stats(self, doc: Dict):
        """Update statistics for a document.
        
        Args:
            doc: Formatted document
        """
        self.stats['total_processed'] += 1
        
        # Domain distribution
        domains = doc.get('domains', [])
        for domain in domains:
            self.stats['domain_counts'][domain] += 1
        
        # Year distribution
        year = doc.get('year')
        if year:
            self.stats['year_counts'][year] += 1
        
        # Token count
        token_count = doc.get('token_count', 0)
        self.stats['total_tokens'] += token_count
        
        # Quality score
        quality_score = doc.get('quality_score', 0.0)
        if quality_score > 0:
            self.stats['quality_scores'].append(quality_score)
    
    def _log_progress(self):
        """Log progress every N records."""
        if self.stats['total_processed'] % self.stats['log_interval'] == 0:
            print(f"📝 Processed {self.stats['total_processed']} documents...")
    
    def _log_final_stats(self):
        """Log final statistics."""
        print("\n" + "=" * 60)
        print("📊 HealthcareJsonlWriter Final Statistics")
        print("=" * 60)
        
        print(f"\n📄 Total documents processed: {self.stats['total_processed']}")
        
        # Domain distribution
        if self.stats['domain_counts']:
            print(f"\n🏷️  Domain distribution:")
            for domain, count in sorted(
                self.stats['domain_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / self.stats['total_processed']) * 100
                print(f"   {domain}: {count} ({percentage:.1f}%)")
        
        # Year distribution
        if self.stats['year_counts']:
            print(f"\n📅 Year distribution:")
            for year in sorted(self.stats['year_counts'].keys()):
                count = self.stats['year_counts'][year]
                percentage = (count / self.stats['total_processed']) * 100
                print(f"   {year}: {count} ({percentage:.1f}%)")
        
        # Token statistics
        if self.stats['total_processed'] > 0:
            avg_tokens = self.stats['total_tokens'] / self.stats['total_processed']
            print(f"\n🔢 Token statistics:")
            print(f"   Total estimated tokens: {self.stats['total_tokens']:,}")
            print(f"   Average tokens per document: {avg_tokens:.1f}")
        
        # Quality score distribution
        if self.stats['quality_scores']:
            quality_scores = self.stats['quality_scores']
            print(f"\n⭐ Quality score distribution:")
            print(f"   Min: {min(quality_scores):.3f}")
            print(f"   Max: {max(quality_scores):.3f}")
            print(f"   Mean: {sum(quality_scores) / len(quality_scores):.3f}")
            print(f"   Median: {sorted(quality_scores)[len(quality_scores) // 2]:.3f}")
        
        print(f"\n💾 Output file: {self.output_path}")
        print("=" * 60)
    
    def __call__(self, dataset):
        """Process dataset and write to JSONL file (NeMo Curator interface).
        
        Args:
            dataset: Input dataset (DocumentDataset or iterable)
            
        Returns:
            Path to output file
        """
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'domain_counts': defaultdict(int),
            'year_counts': defaultdict(int),
            'total_tokens': 0,
            'quality_scores': [],
            'log_interval': 1000
        }
        
        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"📝 Writing curated dataset to {self.output_path}...")
        
        # Write documents to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for doc in dataset:
                # Format document
                formatted_doc = self._format_document(doc)
                
                # Update statistics
                self._update_stats(formatted_doc)
                
                # Write to file
                f.write(json.dumps(formatted_doc, ensure_ascii=False) + '\n')
                
                # Log progress
                self._log_progress()
        
        # Log final statistics
        self._log_final_stats()
        
        return self.output_path


def load_download_checkpoint(checkpoint_file: str) -> Dict:
    """Load download checkpoint to resume from previous run.
    
    Args:
        checkpoint_file: Path to checkpoint JSON file
        
    Returns:
        Dictionary with checkpoint data, or empty dict if not found
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
    return {}


def save_download_checkpoint(checkpoint_file: str, downloaded_ids: Set[str], total_downloaded: int):
    """Save download checkpoint.
    
    Args:
        checkpoint_file: Path to checkpoint JSON file
        downloaded_ids: Set of already downloaded paper IDs
        total_downloaded: Total number of papers downloaded so far
    """
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'total_downloaded': total_downloaded,
        'downloaded_ids': list(downloaded_ids),
    }
    os.makedirs(os.path.dirname(checkpoint_file) if os.path.dirname(checkpoint_file) else '.', exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def run_nemo_curator_pipeline(
    output_path: str = "./curated_dataset.jsonl",
    raw_data_path: str = "./arxiv_raw_data",
    raw_output_path: str = "./arxiv_raw_output.jsonl",
    filter_query: str = "cs.LG OR cs.AI OR q-bio.NC",
    max_workers: int = 1,
    use_gpu: bool = False,
    batch_size: int = 5000,
    checkpoint_interval: int = 1000,
    max_papers: int = 40000,
    resume: bool = True
):
    """Run complete NeMo Curator pipeline with custom healthcare stages.
    
    This function uses NeMo Curator's FREE download_arxiv() function (no AWS needed):
    1. download_arxiv(): Downloads papers directly from ArXiv (FREE, no AWS charges)
    2. Pipeline API: Processes downloaded JSONL files using:
       - JsonlReader: Reads raw ArXiv JSONL files
       - HealthcareFilterStage: Text cleaning, quality filtering, domain classification
       - HealthcareQualityFilterStage: Deduplication and quality verification
       - JsonlWriter: Writes final curated dataset
    
    Features:
    - Checkpointing: Saves progress every N papers (default: 1000)
    - Batching: Processes papers in batches (default: 5000 per batch)
    - Resume: Automatically resumes from last checkpoint if interrupted
    
    Args:
        output_path: Path to final curated dataset JSONL file
        raw_data_path: Directory for raw ArXiv downloads
        raw_output_path: Path to raw output JSONL (before filtering)
        filter_query: ArXiv search query (default: healthcare+ML categories)
        max_workers: Number of workers for download (default: 1, Colab safe)
        use_gpu: Whether to use GPU (for future GPU-accelerated stages)
        batch_size: Number of papers to process per batch (default: 5000)
        checkpoint_interval: Save checkpoint every N papers (default: 1000)
        max_papers: Maximum total papers to download (default: 40000)
        resume: Whether to resume from checkpoint if available (default: True)
    
    Returns:
        Path to output file if successful, None otherwise
    """
    if not NEMO_CURATOR_AVAILABLE:
        error_msg = (
            "❌ Error: NeMo Curator not available.\n"
            "   Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'\n"
            "   Note: NeMo Curator only supports Linux systems"
        )
        print(error_msg)
        raise RuntimeError("NeMo Curator not available. Install with: pip install 'nemo-curator[text]'")
    
    if not Pipeline_AVAILABLE:
        error_msg = "❌ Error: Pipeline class not available. Check NeMo Curator installation."
        print(error_msg)
        raise RuntimeError(error_msg)
    
    if not download_arxiv_AVAILABLE:
        print("❌ Error: download_arxiv() function not available.")
        print("   This requires NeMo Curator with download support")
        return None
    
    print("=" * 60)
    print("🔬 NeMo Curator Healthcare Pipeline (FREE - No AWS Required)")
    print("=" * 60)
    print(f"📁 Raw data directory: {raw_data_path}")
    print(f"📁 Raw output file: {raw_output_path}")
    print(f"📁 Final curated output: {output_path}")
    print(f"🔍 Filter query: {filter_query}")
    print(f"👷 Max workers: {max_workers}")
    print(f"📦 Batch size: {batch_size} papers per batch")
    print(f"💾 Checkpoint interval: {checkpoint_interval} papers")
    print(f"🎯 Max papers: {max_papers}")
    print(f"💰 Cost: FREE (direct ArXiv access, no AWS charges)")
    print()
    
    # Setup checkpointing
    checkpoint_file = os.path.join(os.path.dirname(raw_output_path) if os.path.dirname(raw_output_path) else '.', 
                                    'download_checkpoint.json')
    downloaded_ids = set()
    total_downloaded = 0
    
    # Load checkpoint if resuming
    if resume:
        checkpoint = load_download_checkpoint(checkpoint_file)
        if checkpoint:
            downloaded_ids = set(checkpoint.get('downloaded_ids', []))
            total_downloaded = checkpoint.get('total_downloaded', 0)
            print(f"📖 Resuming from checkpoint: {total_downloaded} papers already downloaded")
            print(f"   Found {len(downloaded_ids)} unique paper IDs in checkpoint")
    
    try:
        # Initialize Dask client
        print("🔧 Initializing Dask client...")
        try:
            client = get_client(cluster_type="cpu")
            print(f"   ✅ Dask client initialized: {client}")
        except:
            # Fallback: create local client
            from dask.distributed import Client
            client = Client(processes=False, threads_per_worker=max_workers)
            print(f"   ✅ Created local Dask client: {client}")
        
        # Stage 1: Download ArXiv papers with checkpointing and batching
        print("\n📥 Stage 1: Downloading ArXiv Papers (FREE)")
        print("   Using download_arxiv() - direct ArXiv access...")
        print("   This may take 4-8 hours depending on network speed...")
        print("   Progress will be checkpointed every {} papers".format(checkpoint_interval))
        
        # Create output directory
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(os.path.dirname(raw_output_path) if os.path.dirname(raw_output_path) else '.', exist_ok=True)
        
        # Open raw output file in append mode if resuming
        raw_output_mode = 'a' if resume and os.path.exists(raw_output_path) else 'w'
        raw_output_handle = open(raw_output_path, raw_output_mode, encoding='utf-8')
        
        try:
            # Download papers using download_arxiv()
            # Note: download_arxiv() may download all at once, but we'll process in batches
            print(f"   🔄 Starting download (target: {max_papers} papers)...")
            
            dataset = download_arxiv(
                output_path=raw_data_path,
                max_workers=max_workers,
                filter_query=filter_query
            )
            
            # Process dataset in batches with checkpointing
            print(f"   📊 Processing downloaded dataset in batches of {batch_size}...")
            
            batch_count = 0
            papers_in_batch = 0
            
            # Convert dataset to iterable if needed
            if hasattr(dataset, '__iter__'):
                dataset_iter = iter(dataset)
            else:
                dataset_iter = [dataset] if dataset else []
            
            for doc in dataset_iter:
                # Check if we've reached max papers
                if total_downloaded >= max_papers:
                    print(f"\n   ✅ Reached target of {max_papers} papers")
                    break
                
                # Extract paper ID (try different field names)
                paper_id = doc.get('id') or doc.get('arxiv_id') or doc.get('file_name', '').replace('.txt', '')
                
                # Skip if already downloaded (resume support)
                if paper_id and paper_id in downloaded_ids:
                    continue
                
                # Write to raw output JSONL
                raw_output_handle.write(json.dumps(doc, ensure_ascii=False) + '\n')
                raw_output_handle.flush()  # Ensure immediate write
                
                # Track downloaded papers
                if paper_id:
                    downloaded_ids.add(paper_id)
                total_downloaded += 1
                papers_in_batch += 1
                
                # Log progress
                if total_downloaded % 100 == 0:
                    print(f"   📊 Progress: {total_downloaded}/{max_papers} papers downloaded...")
                
                # Checkpoint every N papers
                if total_downloaded % checkpoint_interval == 0:
                    save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
                    print(f"   💾 Checkpoint saved: {total_downloaded} papers downloaded")
                
                # Process batch if full
                if papers_in_batch >= batch_size:
                    batch_count += 1
                    print(f"   ✅ Batch {batch_count} complete: {papers_in_batch} papers")
                    papers_in_batch = 0
                    
                    # Save checkpoint after each batch
                    save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
            
            # Final checkpoint
            save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
            raw_output_handle.close()
            
            print(f"\n   ✅ Download complete: {total_downloaded} papers downloaded")
            print(f"   💾 Final checkpoint saved")
            
        except Exception as e:
            raw_output_handle.close()
            print(f"   ⚠️  Download interrupted: {e}")
            print(f"   💾 Saving checkpoint with {total_downloaded} papers...")
            save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
            raise
        
        # Stage 2: Process downloaded JSONL using Pipeline API
        print("\n🔍 Stage 2: Processing with NeMo Curator Pipeline API")
        print("   Creating pipeline with custom healthcare stages...")
        
        if JsonlReader_AVAILABLE and ProcessingStage_AVAILABLE:
            # Use Pipeline API (preferred method)
            print("   ✅ Using NeMo Curator Pipeline API")
            
            # Create pipeline
            pipeline = Pipeline(name="healthcare_curation_pipeline")
            
            # Step 1: Read JSONL files
            print("   📖 Adding JsonlReader stage...")
            reader = JsonlReader(
                file_paths=raw_output_path,
                files_per_partition=4,
                fields=["text", "file_name"]  # Read text and filename fields
            )
            pipeline.add_stage(reader)
            
            # Step 2: Healthcare filtering and classification
            print("   🔍 Adding HealthcareFilterStage...")
            healthcare_filter = HealthcareFilterStage()
            pipeline.add_stage(healthcare_filter)
            
            # Step 3: Quality filtering and deduplication
            print("   ✨ Adding HealthcareQualityFilterStage...")
            quality_filter = HealthcareQualityFilterStage()
            pipeline.add_stage(quality_filter)
            
            # Step 4: Write curated dataset
            print("   💾 Adding JsonlWriter stage...")
            if JsonlWriter_AVAILABLE:
                writer = JsonlWriter(path=output_path)
                pipeline.add_stage(writer)
            else:
                # Fallback: use custom writer
                print("   ⚠️  JsonlWriter not available, using custom writer...")
                custom_writer = HealthcareJsonlWriter(output_path=output_path)
                pipeline.add_stage(custom_writer)
            
            # Execute pipeline
            print("\n   🚀 Executing pipeline...")
            results = pipeline.run()
            print("   ✅ Pipeline execution complete!")
            
        else:
            # Fallback: manual processing (if Pipeline API not available)
            print("   ⚠️  Pipeline API not fully available, falling back to manual processing...")
            raw_documents = []
            with open(raw_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            doc = json.loads(line)
                            raw_documents.append(doc)
                        except json.JSONDecodeError:
                            continue
            
            print(f"   ✅ Loaded {len(raw_documents)} documents for processing")
            
            # Process with custom stages
            filtered_dataset = HealthcareFilterStage()(raw_documents)
            
            # Quality filtering and deduplication
            print("\n✨ Stage 3: Quality Filtering & Deduplication")
            print("   Using HealthcareQualityFilterStage...")
            quality_filtered_dataset = HealthcareQualityFilterStage()(filtered_dataset)
            
            # Write curated dataset
            print("\n💾 Stage 4: Writing Curated Dataset")
            print("   Using HealthcareJsonlWriter...")
            final_output = HealthcareJsonlWriter(output_path=output_path)(quality_filtered_dataset)
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Final curated dataset: {output_path}")
        print(f"📁 Raw data (for reference): {raw_output_path}")
        print(f"📊 Total papers downloaded: {total_downloaded}")
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        print(traceback.format_exc())
        print(f"\n💾 Checkpoint saved - you can resume by running again with resume=True")
        return None


def curate_with_nemo(
    text_dir: str,
    metadata_jsonl: str,
    output_jsonl: str,
    use_gpu: bool = False,
    skip_dedup: bool = False,
    min_relevance_score: float = 0.5
):
    """Curate healthcare papers using NeMo Curator pipeline.
    
    Args:
        text_dir: Directory containing raw text files
        metadata_jsonl: Input JSONL file with paper metadata
        output_jsonl: Output curated dataset JSONL file
        use_gpu: Whether to use GPU for deduplication
        skip_dedup: Skip deduplication stage (for memory-constrained environments)
        min_relevance_score: Minimum domain relevance score to keep
    """
    if not NEMO_CURATOR_AVAILABLE:
        error_msg = (
            "❌ Error: NeMo Curator not available.\n"
            "   Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'\n"
            "   Note: NeMo Curator only supports Linux systems"
        )
        print(error_msg)
        raise RuntimeError("NeMo Curator not available. Install with: pip install 'nemo-curator[text]'")
    
    print("=" * 60)
    print("🔬 NeMo Curator Text Curation Pipeline")
    print("=" * 60)
    print(f"📁 Text directory: {text_dir}")
    print(f"📁 Metadata file: {metadata_jsonl}")
    print(f"📁 Output file: {output_jsonl}")
    print(f"🎯 Min relevance score: {min_relevance_score}")
    print(f"🔧 GPU deduplication: {use_gpu}")
    print(f"🔧 Skip deduplication: {skip_dedup}")
    print(f"🔍 NEMO_CURATOR_AVAILABLE: {NEMO_CURATOR_AVAILABLE}")
    print()
    
    # Verify inputs exist
    if not os.path.exists(text_dir):
        error_msg = f"❌ Text directory does not exist: {text_dir}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not os.path.exists(metadata_jsonl):
        error_msg = f"❌ Metadata file does not exist: {metadata_jsonl}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    print(f"📊 Found {len(text_files)} text files in {text_dir}")
    
    if len(text_files) == 0:
        error_msg = f"❌ No text files found in {text_dir}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Initialize Dask client for parallelization
    try:
        if get_client is not None:
            try:
                client = get_client()
                print(f"✅ Dask client initialized: {client}")
            except:
                # Create local Dask client
                from dask.distributed import Client
                client = Client(processes=False, threads_per_worker=2)
                print(f"✅ Created local Dask client: {client}")
        else:
            # Create local Dask client manually
            from dask.distributed import Client
            client = Client(processes=False, threads_per_worker=2)
            print(f"✅ Created local Dask client: {client}")
    except Exception as e:
        print(f"⚠️  Dask client creation failed: {e}")
        print("   Continuing without Dask (will use sequential processing)")
        client = None
    
    # Load metadata
    print("📚 Loading metadata...")
    metadata_map = {}
    with open(metadata_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    paper = json.loads(line)
                    arxiv_id = paper.get('id', '')
                    if arxiv_id:
                        metadata_map[arxiv_id] = paper
                except:
                    continue
    print(f"   ✅ Loaded {len(metadata_map)} metadata entries")
    
    # Stage 1: Load and prepare documents
    print("\n" + "=" * 60)
    print("Stage 1: Loading Documents")
    print("=" * 60)
    
    documents = []
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    
    for filename in text_files:
        arxiv_id = filename[:-4]  # Remove .txt
        text_file = os.path.join(text_dir, filename)
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                continue
            
            # Get metadata
            metadata = metadata_map.get(arxiv_id, {})
            
            document = {
                'arxiv_id': arxiv_id,
                'text': text,
                'year': metadata.get('year'),
                'title': metadata.get('title', ''),
            }
            documents.append(document)
        except Exception as e:
            print(f"   ⚠️  Error loading {filename}: {e}")
            continue
    
    print(f"   ✅ Loaded {len(documents)} documents")
    initial_count = len(documents)
    
    # Create DocumentDataset (NeMo Curator format)
    # For compatibility, use list if NeMo Curator not available
    if NEMO_CURATOR_AVAILABLE:
        try:
            dataset = DocumentDataset(documents)
            print("   ✅ Using NeMo Curator DocumentDataset")
        except Exception as e:
            print(f"   ⚠️  DocumentDataset creation failed: {e}, using list")
            dataset = documents
    else:
        dataset = documents
        print("   ✅ Using list-based dataset (NeMo Curator not available)")
    
    # Stage 2: Text Cleaning & Normalization
    print("\n" + "=" * 60)
    print("Stage 2: Text Cleaning & Normalization")
    print("=" * 60)
    
    cleaner = create_healthcare_text_cleaner()
    if cleaner and NEMO_CURATOR_AVAILABLE:
        try:
            if hasattr(dataset, 'map'):
                dataset = dataset.map(cleaner.modify_document)
            else:
                # Fallback: apply to list
                dataset = [cleaner.modify_document(doc) for doc in dataset]
            print("   ✅ Text cleaning applied (NeMo Curator)")
        except Exception as e:
            print(f"   ⚠️  NeMo Curator cleaning failed: {e}, using fallback")
            # Fallback to simple cleaning
            def simple_clean(doc):
                text = doc.get('text', '')
                # Remove URLs, emails, citations
                text = re.sub(r'http[s]?://\S+', '', text)
                text = re.sub(r'\S+@\S+', '', text)
                text = re.sub(r'\[[\d,\s-]+\]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                doc['text'] = text
                return doc
            
            dataset = [simple_clean(doc) for doc in dataset] if isinstance(dataset, list) else dataset.map(simple_clean)
            print("   ✅ Simple text cleaning applied")
    else:
        # Fallback to simple cleaning
        def simple_clean(doc):
            text = doc.get('text', '')
            # Remove URLs, emails, citations
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'\[[\d,\s-]+\]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            doc['text'] = text
            return doc
        
        if isinstance(dataset, list):
            dataset = [simple_clean(doc) for doc in dataset]
        else:
            dataset = dataset.map(simple_clean)
        print("   ✅ Simple text cleaning applied (NeMo Curator not available)")
    
    # Stage 3: Quality Filtering
    print("\n" + "=" * 60)
    print("Stage 3: Quality Filtering")
    print("=" * 60)
    
    before_quality = len(dataset)
    
    # Word count filter: 100-5000 tokens
    def word_count_filter(doc):
        text = doc.get('text', '')
        words = text.split()
        return 100 <= len(words) <= 5000
    
    # Alphanumeric ratio filter: >40%
    def alphanumeric_filter(doc):
        text = doc.get('text', '')
        if not text:
            return False
        alnum_chars = sum(1 for c in text if c.isalnum())
        ratio = alnum_chars / len(text) if len(text) > 0 else 0
        return ratio > 0.4
    
    # Language filter: English only (simple heuristic)
    def language_filter(doc):
        text = doc.get('text', '')
        # Simple heuristic: check for common English words
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if word in text_lower)
        return english_count >= 3  # At least 3 common English words
    
    # Apply filters (handle both list and DocumentDataset)
    if isinstance(dataset, list):
        dataset = [doc for doc in dataset if word_count_filter(doc)]
        print(f"   ✅ Word count filter: {len(dataset)}/{before_quality} documents")
        
        dataset = [doc for doc in dataset if alphanumeric_filter(doc)]
        after_alnum = len(dataset)
        print(f"   ✅ Alphanumeric filter: {after_alnum} documents")
        
        dataset = [doc for doc in dataset if language_filter(doc)]
        after_lang = len(dataset)
        print(f"   ✅ Language filter: {after_lang} documents")
    else:
        dataset = dataset.filter(word_count_filter)
        print(f"   ✅ Word count filter: {len(dataset)}/{before_quality} documents")
        
        dataset = dataset.filter(alphanumeric_filter)
        after_alnum = len(dataset)
        print(f"   ✅ Alphanumeric filter: {after_alnum} documents")
        
        dataset = dataset.filter(language_filter)
        after_lang = len(dataset)
        print(f"   ✅ Language filter: {after_lang} documents")
    
    # Stage 4: Domain-Specific Filtering
    print("\n" + "=" * 60)
    print("Stage 4: Domain-Specific Filtering")
    print("=" * 60)
    
    before_domain = len(dataset)
    
    # Use HealthcareDomainFilter directly
    domain_filter = HealthcareDomainFilter()
    
    # Apply filter and score documents
    if isinstance(dataset, list):
        filtered_docs = []
        for doc in dataset:
            if domain_filter.filter_document(doc, min_relevance=min_relevance_score):
                filtered_docs.append(doc)
        dataset = filtered_docs
    else:
        # Use NeMo Curator ScoreFilter if available
        if NEMO_CURATOR_AVAILABLE:
            try:
                # Create a wrapper for ScoreFilter compatibility
                class FilterWrapper:
                    def __init__(self, filter_instance, min_score):
                        self.filter_instance = filter_instance
                        self.min_score = min_score
                    
                    def score_document(self, doc):
                        return self.filter_instance.score_document(doc)
                    
                    def filter_document(self, doc):
                        return self.filter_instance.filter_document(doc, self.min_score)
                
                wrapper = FilterWrapper(domain_filter, min_relevance_score)
                # Apply filter
                if hasattr(dataset, 'filter'):
                    dataset = dataset.filter(wrapper.filter_document)
                else:
                    # Fallback to list comprehension
                    dataset = [doc for doc in dataset if wrapper.filter_document(doc)]
            except Exception as e:
                print(f"   ⚠️  NeMo Curator ScoreFilter failed: {e}, using fallback")
                dataset = [doc for doc in dataset if domain_filter.filter_document(doc, min_relevance=min_relevance_score)]
        else:
            dataset = [doc for doc in dataset if domain_filter.filter_document(doc, min_relevance=min_relevance_score)]
    
    after_domain = len(dataset)
    print(f"   ✅ Domain relevance filter: {after_domain}/{before_domain} documents")
    print(f"   📊 Relevance threshold: {min_relevance_score}")
    
    # Count domain distribution
    domain_counts = defaultdict(int)
    for doc in (dataset if isinstance(dataset, list) else list(dataset)):
        domains = doc.get('domains', [])
        for domain in domains:
            domain_counts[domain] += 1
    
    if domain_counts:
        print(f"   📊 Domain distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {domain}: {count} documents")
    
    # Stage 5: Deduplication (optional)
    if not skip_dedup:
        print("\n" + "=" * 60)
        print("Stage 5: Deduplication")
        print("=" * 60)
        
        before_dedup = len(dataset)
        
        try:
            if NEMO_CURATOR_AVAILABLE and FuzzyDedup is not None:
                if use_gpu:
                    deduplicator = FuzzyDedup(similarity_threshold=0.95, use_gpu=True)
                else:
                    deduplicator = FuzzyDedup(similarity_threshold=0.95, use_gpu=False)
                
                # Apply deduplication
                if hasattr(dataset, 'map'):
                    # DocumentDataset
                    dataset = deduplicator(dataset)
                else:
                    # List - convert to DocumentDataset for deduplication
                    try:
                        temp_dataset = DocumentDataset(dataset)
                        temp_dataset = deduplicator(temp_dataset)
                        dataset = list(temp_dataset)
                    except:
                        # Fallback: simple deduplication by text hash
                        seen_texts = set()
                        unique_docs = []
                        for doc in dataset:
                            text_hash = hash(doc.get('text', ''))
                            if text_hash not in seen_texts:
                                seen_texts.add(text_hash)
                                unique_docs.append(doc)
                        dataset = unique_docs
                
                after_dedup = len(dataset) if isinstance(dataset, list) else len(list(dataset))
                print(f"   ✅ Deduplication: {after_dedup}/{before_dedup} documents")
            else:
                print("   ⚠️  NeMo Curator deduplication not available, using simple hash-based dedup")
                # Simple deduplication by text hash
                seen_texts = set()
                unique_docs = []
                for doc in (dataset if isinstance(dataset, list) else list(dataset)):
                    text_hash = hash(doc.get('text', ''))
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        unique_docs.append(doc)
                dataset = unique_docs
                print(f"   ✅ Simple deduplication: {len(dataset)}/{before_dedup} documents")
        except Exception as e:
            print(f"   ⚠️  Deduplication failed: {e}")
            print("   Continuing without deduplication...")
    else:
        print("\n" + "=" * 60)
        print("Stage 5: Deduplication (Skipped)")
        print("=" * 60)
        print("   ℹ️  Deduplication skipped as requested")
    
    # Stage 6: Format & Export
    print("\n" + "=" * 60)
    print("Stage 6: Format & Export")
    print("=" * 60)
    
    # Domains are already assigned by HealthcareDomainFilter in Stage 4
    # No need to add them again
    
    # Export to JSONL
    print(f"   💾 Exporting to {output_jsonl}...")
    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else '.', exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        # Handle both list and DocumentDataset
        docs_iter = dataset if isinstance(dataset, list) else iter(dataset)
        for doc in docs_iter:
            # Get domain scores if available
            domain_scores = doc.get('domain_scores', {})
            
            output_record = {
                'arxiv_id': doc.get('arxiv_id'),
                'text': doc.get('text'),
                'domains': doc.get('domains', []),
                'year': doc.get('year'),
                'quality_score': doc.get('relevance_score', 0.0),
                'domain_scores': {k: v for k, v in domain_scores.items() if k != 'relevance'} if domain_scores else {},
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    
    final_count = len(dataset) if isinstance(dataset, list) else len(list(dataset))
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Curation Complete!")
    print("=" * 60)
    print(f"📊 Initial documents: {initial_count}")
    print(f"📊 After quality filtering: {after_lang}")
    print(f"📊 After domain filtering: {after_domain}")
    print(f"📊 Final curated documents: {final_count}")
    print(f"📊 Retention rate: {final_count/initial_count*100:.1f}%")
    print(f"📁 Output file: {output_jsonl}")
    
    # Quality score distribution
    if final_count > 0:
        docs_list = dataset if isinstance(dataset, list) else list(dataset)
        scores = [doc.get('relevance_score', 0.0) for doc in docs_list]
        if scores:
            import numpy as np
            print(f"\n📊 Quality Score Distribution:")
            print(f"   Mean: {np.mean(scores):.3f}")
            print(f"   Median: {np.median(scores):.3f}")
            print(f"   Min: {np.min(scores):.3f}")
            print(f"   Max: {np.max(scores):.3f}")
    
    # Validation: Sample 100 documents and verify domain detection
    print("\n" + "=" * 60)
    print("Validation: Domain Detection Accuracy")
    print("=" * 60)
    
    if final_count > 0:
        docs_list = dataset if isinstance(dataset, list) else list(dataset)
        sample_size = min(100, final_count)
        sample_docs = random.sample(docs_list, sample_size)
        
        print(f"   📊 Sampling {sample_size} documents for validation...")
        
        validation_results = {
            'total_sampled': sample_size,
            'with_domains': 0,
            'domain_distribution': defaultdict(int),
            'avg_relevance': 0.0,
            'high_relevance': 0,  # relevance > 0.7
        }
        
        relevance_scores = []
        for doc in sample_docs:
            domains = doc.get('domains', [])
            if domains and domains != ['general_ml_health']:
                validation_results['with_domains'] += 1
                for domain in domains:
                    validation_results['domain_distribution'][domain] += 1
            
            relevance = doc.get('relevance_score', 0.0)
            relevance_scores.append(relevance)
            if relevance > 0.7:
                validation_results['high_relevance'] += 1
        
        validation_results['avg_relevance'] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        print(f"   ✅ Documents with detected domains: {validation_results['with_domains']}/{sample_size}")
        print(f"   📊 Average relevance score: {validation_results['avg_relevance']:.3f}")
        print(f"   📊 High relevance documents (>0.7): {validation_results['high_relevance']}/{sample_size}")
        print(f"   📊 Domain distribution in sample:")
        for domain, count in sorted(validation_results['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"      {domain}: {count} documents")
    
    # Close Dask client
    try:
        if client is not None:
            client.close()
            print("✅ Dask client closed")
    except Exception as e:
        print(f"⚠️  Error closing Dask client: {e}")
    
    # Final confirmation
    print("\n" + "=" * 60)
    print("✅ curate_with_nemo() completed successfully!")
    print(f"📁 Output file: {output_jsonl}")
    if os.path.exists(output_jsonl):
        file_size = os.path.getsize(output_jsonl) / (1024 * 1024)  # MB
        count = sum(1 for _ in open(output_jsonl))
        print(f"📊 Output file size: {file_size:.2f} MB")
        print(f"📊 Documents in output: {count}")
    print("=" * 60)


def process_curated_dataset(
    input_jsonl: str,
    output_jsonl: str,
    num_workers: int = 4
):
    """Process curated dataset with healthcare-specific preprocessing.
    
    Args:
        input_jsonl: Input curated dataset JSONL file (from curate command)
        output_jsonl: Output processed dataset JSONL file
        num_workers: Number of parallel workers
    """
    print("=" * 60)
    print("🔬 Healthcare Text Processing Pipeline")
    print("=" * 60)
    print(f"📁 Input file: {input_jsonl}")
    print(f"📁 Output file: {output_jsonl}")
    print(f"👷 Workers: {num_workers}")
    print()
    
    # Initialize modifier
    modifier = HealthcareTextModifier()
    
    # Load documents
    print("📚 Loading curated dataset...")
    documents = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    total_docs = len(documents)
    print(f"   ✅ Loaded {total_docs} documents")
    print()
    
    # Process documents in parallel
    print("🔄 Processing documents...")
    processed_docs = []
    lock = threading.Lock()
    
    def process_single_doc(doc):
        """Process a single document."""
        try:
            # Apply modifier
            processed_doc = modifier.modify_document(doc.copy())
            
            # Ensure all required fields are present
            output_doc = {
                'arxiv_id': processed_doc.get('arxiv_id', ''),
                'text': processed_doc.get('text', ''),
                'sections': processed_doc.get('sections', {}),
                'domains': processed_doc.get('domains', []),
                'year': processed_doc.get('year'),
                'quality_score': processed_doc.get('quality_score', processed_doc.get('relevance_score', 0.0)),
                'token_count_estimate': processed_doc.get('token_count_estimate', 0),
                'medical_terms_detected': processed_doc.get('medical_terms_detected', []),
            }
            
            return output_doc, None
        except Exception as e:
            return None, str(e)
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_doc, doc) for doc in documents]
        
        completed = 0
        errors = 0
        
        for future in as_completed(futures):
            completed += 1
            result, error = future.result()
            
            if error:
                errors += 1
                if errors <= 10:  # Only show first 10 errors
                    print(f"   ⚠️  Error processing document: {error}")
            else:
                processed_docs.append(result)
            
            # Progress update
            if completed % 500 == 0:
                print(f"   📊 Progress: {completed}/{total_docs} documents processed...")
    
    print(f"   ✅ Processed {len(processed_docs)} documents")
    if errors > 0:
        print(f"   ⚠️  Errors: {errors} documents failed")
    print()
    
    # Write output
    print("💾 Writing processed dataset...")
    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else '.', exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for doc in processed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Statistics
    print()
    print("=" * 60)
    print("✅ Processing Complete!")
    print("=" * 60)
    print(f"📊 Total documents: {total_docs}")
    print(f"✅ Successfully processed: {len(processed_docs)}")
    print(f"❌ Failed: {errors}")
    print(f"📁 Output file: {output_jsonl}")
    
    # Calculate statistics
    if processed_docs:
        total_tokens = sum(doc.get('token_count_estimate', 0) for doc in processed_docs)
        avg_tokens = total_tokens / len(processed_docs) if processed_docs else 0
        
        sections_found = defaultdict(int)
        for doc in processed_docs:
            sections = doc.get('sections', {})
            for section_name in sections:
                sections_found[section_name] += 1
        
        medical_terms_count = sum(len(doc.get('medical_terms_detected', [])) for doc in processed_docs)
        avg_medical_terms = medical_terms_count / len(processed_docs) if processed_docs else 0
        
        print(f"\n📊 Statistics:")
        print(f"   Average tokens per document: {avg_tokens:.0f}")
        print(f"   Average medical terms per document: {avg_medical_terms:.1f}")
        print(f"   Sections detected:")
        for section, count in sorted(sections_found.items(), key=lambda x: x[1], reverse=True):
            print(f"      {section}: {count} documents")
        
        # File size
        file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
        print(f"\n💾 Output file size: {file_size_mb:.2f} MB")


# Tokenizer Training Constants
TOKENIZER_VOCAB_SIZE = 50000
TOKENIZER_MODEL_TYPE = 'bpe'
TOKENIZER_CHAR_COVERAGE = 0.9995
TOKENIZER_SPECIAL_TOKENS = ['[DISEASE]', '[PROTEIN]', '[DRUG]', '[GENE]']
TOKENIZER_NORMALIZATION = 'identity'  # Don't normalize, preserve case

# Medical terms for validation
MEDICAL_TERMS = [
    'alzheimer', 'parkinson', 'protein', 'synapse', 'fmri', 'mri', 'eeg', 'neural',
    'dementia', 'tau', 'amyloid', 'neuron', 'cortex', 'neurotransmitter',
    'diagnosis', 'treatment', 'therapy', 'clinical', 'patient', 'disease'
]


def extract_texts_from_jsonl(input_jsonl: str, output_txt: str):
    """Extract all text from JSONL dataset and concatenate into single .txt file.
    
    Args:
        input_jsonl: Input JSONL file with processed papers
        output_txt: Output text file path
    """
    print(f"📚 Extracting texts from {input_jsonl}...")
    
    total_papers = 0
    total_chars = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
         open(output_txt, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                text = record.get('text', '')
                
                if text and text.strip():
                    # Split long texts into chunks to avoid SentencePiece max length issues
                    # Each chunk should be <= 15000 chars to stay well under the 20000 limit
                    max_chunk_size = 15000
                    if len(text) > max_chunk_size:
                        # Split by sentences (period + space/newline) or by paragraphs
                        chunks = []
                        current_chunk = ""
                        
                        # Try to split by paragraphs first (double newline)
                        paragraphs = text.split('\n\n')
                        for para in paragraphs:
                            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                                current_chunk += para + '\n\n'
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = para + '\n\n'
                        
                        # Add remaining chunk
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # If still too long, split by sentences
                        final_chunks = []
                        for chunk in chunks:
                            if len(chunk) <= max_chunk_size:
                                final_chunks.append(chunk)
                            else:
                                # Split by sentences
                                sentences = chunk.replace('. ', '.\n').split('\n')
                                current = ""
                                for sent in sentences:
                                    if len(current) + len(sent) + 1 <= max_chunk_size:
                                        current += sent + ' '
                                    else:
                                        if current:
                                            final_chunks.append(current.strip())
                                        current = sent + ' '
                                if current:
                                    final_chunks.append(current.strip())
                        
                        # Write chunks as separate "sentences"
                        for chunk in final_chunks:
                            if chunk.strip():
                                f_out.write(chunk.strip() + '\n')
                    else:
                        f_out.write(text + '\n\n')  # Add double newline between papers
                    
                    total_papers += 1
                    total_chars += len(text)
                
                if line_num % 1000 == 0:
                    print(f"   Processed {line_num} records, {total_papers} papers with text...")
            
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Invalid JSON on line {line_num}")
                continue
    
    file_size_mb = os.path.getsize(output_txt) / (1024 * 1024)
    print(f"✅ Extracted {total_papers} papers, {total_chars:,} characters")
    print(f"💾 Output file size: {file_size_mb:.2f} MB")
    print(f"📁 Output file: {output_txt}")
    
    return total_papers, total_chars


def train_tokenizer(
    input_txt: str,
    model_prefix: str = 'healthcare_tokenizer',
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    model_type: str = TOKENIZER_MODEL_TYPE,
    char_coverage: float = TOKENIZER_CHAR_COVERAGE,
    special_tokens: List[str] = None,
    normalization: str = TOKENIZER_NORMALIZATION
):
    """Train SentencePiece BPE tokenizer.
    
    Args:
        input_txt: Input text file for training
        model_prefix: Prefix for output model files
        vocab_size: Vocabulary size
        model_type: Model type ('bpe', 'unigram', etc.)
        char_coverage: Character coverage (0.9995 = 99.95%)
        special_tokens: List of special tokens to preserve
        normalization: Normalization rule name ('identity' = no normalization)
    """
    if not SENTENCEPIECE_AVAILABLE:
        print("❌ Error: sentencepiece package not available.")
        print("   Install with: pip install sentencepiece")
        return None
    
    if special_tokens is None:
        special_tokens = TOKENIZER_SPECIAL_TOKENS
    
    print("=" * 60)
    print("🔤 Training SentencePiece BPE Tokenizer")
    print("=" * 60)
    print(f"📁 Input file: {input_txt}")
    print(f"📊 Vocabulary size: {vocab_size}")
    print(f"🔧 Model type: {model_type}")
    print(f"📈 Character coverage: {char_coverage}")
    print(f"🔤 Special tokens: {special_tokens}")
    print(f"📝 Normalization: {normalization}")
    print()
    
    # Build SentencePiece training command
    # Note: sentencepiece.SentencePieceTrainer.train() uses command-line style args
    train_args = {
        'input': input_txt,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'model_type': model_type,
        'character_coverage': char_coverage,
        'normalization_rule_name': normalization,
        'shuffle_input_sentence': True,
        'input_sentence_size': 10000000,  # Process up to 10M sentences
        'seed_sentencepiece_size': 1000000,
        'shrinking_factor': 0.75,
        'num_threads': 4,
        'num_sub_iterations': 2,
        'max_sentence_length': 20000,  # Allow longer sentences (default is 4192)
    }
    
    # Add special tokens
    if special_tokens:
        train_args['user_defined_symbols'] = ','.join(special_tokens)
    
    print("🚀 Starting tokenizer training...")
    print("   This may take several minutes for large datasets...")
    print()
    
    try:
        spm.SentencePieceTrainer.train(**train_args)
        print("✅ Tokenizer training complete!")
        print(f"📁 Model file: {model_prefix}.model")
        print(f"📁 Vocab file: {model_prefix}.vocab")
        return model_prefix
    except Exception as e:
        print(f"❌ Error training tokenizer: {e}")
        return None


def validate_tokenizer(
    model_path: str,
    medical_terms: List[str] = None
) -> Dict:
    """Validate tokenizer with medical terms and generate report.
    
    Args:
        model_path: Path to tokenizer model file
        medical_terms: List of medical terms to validate
        
    Returns:
        Dictionary with validation results
    """
    if not SENTENCEPIECE_AVAILABLE:
        print("❌ Error: sentencepiece package not available.")
        return {}
    
    if medical_terms is None:
        medical_terms = MEDICAL_TERMS
    
    print("=" * 60)
    print("🔍 Tokenizer Validation")
    print("=" * 60)
    
    # Load tokenizer
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        print(f"✅ Loaded tokenizer from {model_path}")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return {}
    
    # Validate medical terms
    print(f"\n📊 Validating {len(medical_terms)} medical terms...")
    single_token_count = 0
    multi_token_count = 0
    term_results = {}
    
    for term in medical_terms:
        tokens = sp.encode(term, out_type=str)
        num_tokens = len(tokens)
        
        if num_tokens == 1:
            single_token_count += 1
            term_results[term] = {
                'tokens': tokens,
                'num_tokens': 1,
                'is_single': True
            }
        else:
            multi_token_count += 1
            term_results[term] = {
                'tokens': tokens,
                'num_tokens': num_tokens,
                'is_single': False
            }
    
    efficiency = (single_token_count / len(medical_terms)) * 100 if medical_terms else 0
    
    print(f"✅ Single-token coverage: {single_token_count}/{len(medical_terms)} ({efficiency:.1f}%)")
    print(f"⚠️  Multi-token terms: {multi_token_count}/{len(medical_terms)}")
    
    # Generate sample tokenizations
    print(f"\n📝 Sample tokenizations (100 examples):")
    sample_texts = [
        "Alzheimer's disease is a neurodegenerative disorder",
        "Parkinson's disease affects dopamine neurons",
        "Protein aggregation in tau and amyloid",
        "Synapse formation and neural plasticity",
        "fMRI and EEG are neuroimaging techniques",
        "MRI scans show brain atrophy",
        "Clinical diagnosis of dementia",
        "Treatment with therapeutic drugs",
        "Neural network models for brain imaging",
        "Patient data from clinical trials",
    ]
    
    # Repeat to get 100 samples
    all_samples = []
    for i in range(10):
        for text in sample_texts:
            tokens = sp.encode(text, out_type=str)
            all_samples.append({
                'text': text,
                'tokens': tokens,
                'num_tokens': len(tokens)
            })
    
    # Print first 20 samples
    for i, sample in enumerate(all_samples[:20], 1):
        tokens_str = ' '.join(sample['tokens'][:10])  # Show first 10 tokens
        if len(sample['tokens']) > 10:
            tokens_str += f" ... ({sample['num_tokens']} total)"
        print(f"   {i:2d}. Text: {sample['text'][:50]}...")
        print(f"       Tokens: {tokens_str}")
    
    if len(all_samples) > 20:
        print(f"   ... ({len(all_samples) - 20} more samples)")
    
    # Build validation report
    validation_report = {
        'model_path': model_path,
        'vocab_size': sp.get_piece_size(),
        'medical_terms_tested': len(medical_terms),
        'single_token_count': single_token_count,
        'multi_token_count': multi_token_count,
        'efficiency_percent': efficiency,
        'term_results': term_results,
        'sample_tokenizations': all_samples[:100],  # Keep first 100
    }
    
    return validation_report


def save_validation_report(report: Dict, output_file: str):
    """Save validation report to JSON file.
    
    Args:
        report: Validation report dictionary
        output_file: Output JSON file path
    """
    # Convert to JSON-serializable format
    report_json = {
        'model_path': report.get('model_path', ''),
        'vocab_size': report.get('vocab_size', 0),
        'medical_terms_tested': report.get('medical_terms_tested', 0),
        'single_token_count': report.get('single_token_count', 0),
        'multi_token_count': report.get('multi_token_count', 0),
        'efficiency_percent': report.get('efficiency_percent', 0.0),
        'term_results': report.get('term_results', {}),
        'sample_tokenizations': [
            {
                'text': s['text'],
                'tokens': s['tokens'],
                'num_tokens': s['num_tokens']
            }
            for s in report.get('sample_tokenizations', [])[:100]
        ],
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Validation report saved to: {output_file}")


def train_healthcare_tokenizer(
    input_jsonl: str,
    output_dir: str = './data/arxiv',
    model_prefix: str = 'healthcare_tokenizer',
    vocab_size: int = TOKENIZER_VOCAB_SIZE
):
    """Complete pipeline: extract texts, train tokenizer, validate.
    
    Args:
        input_jsonl: Input JSONL file with processed papers
        output_dir: Output directory for tokenizer files
        model_prefix: Prefix for tokenizer model files
        vocab_size: Vocabulary size for tokenizer
    """
    print("=" * 60)
    print("🔤 Healthcare Tokenizer Training Pipeline")
    print("=" * 60)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract texts
    temp_txt_file = os.path.join(output_dir, 'training_texts.txt')
    print("Step 1: Extracting texts from JSONL...")
    total_papers, total_chars = extract_texts_from_jsonl(input_jsonl, temp_txt_file)
    print()
    
    if total_papers == 0:
        error_msg = f"❌ No papers found in input file: {input_jsonl}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Step 2: Train tokenizer
    print("Step 2: Training SentencePiece tokenizer...")
    model_path = os.path.join(output_dir, model_prefix)
    trained_model = train_tokenizer(
        input_txt=temp_txt_file,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=TOKENIZER_MODEL_TYPE,
        char_coverage=TOKENIZER_CHAR_COVERAGE,
        special_tokens=TOKENIZER_SPECIAL_TOKENS,
        normalization=TOKENIZER_NORMALIZATION
    )
    print()
    
    if not trained_model:
        error_msg = "❌ Tokenizer training failed!"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Step 3: Validate tokenizer
    print("Step 3: Validating tokenizer...")
    model_file = f"{model_path}.model"
    validation_report = validate_tokenizer(model_file, medical_terms=MEDICAL_TERMS)
    print()
    
    # Step 4: Save validation report
    if validation_report:
        report_file = os.path.join(output_dir, 'tokenizer_validation_report.json')
        save_validation_report(validation_report, report_file)
        print()
        
        # Print summary
        print("=" * 60)
        print("✅ Tokenizer Training Complete!")
        print("=" * 60)
        print(f"📁 Model file: {model_file}")
        print(f"📁 Vocab file: {model_path}.vocab")
        print(f"📄 Validation report: {report_file}")
        print(f"📊 Vocabulary size: {validation_report.get('vocab_size', vocab_size)}")
        print(f"📊 Medical term efficiency: {validation_report.get('efficiency_percent', 0):.1f}%")
        print()
        
        # Print term results
        print("📊 Medical term tokenization results:")
        term_results = validation_report.get('term_results', {})
        for term, result in sorted(term_results.items()):
            status = "✓" if result['is_single'] else "✗"
            print(f"   {status} {term}: {result['num_tokens']} token(s) - {result['tokens']}")
    
    # Cleanup temp file (optional - comment out if you want to keep it)
    # if os.path.exists(temp_txt_file):
    #     os.remove(temp_txt_file)
    #     print(f"\n🧹 Cleaned up temporary file: {temp_txt_file}")


def run_full_pipeline(config_path: str = "config.yaml"):
    """Run complete end-to-end pipeline with validation and error recovery.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        True if successful, False otherwise
    """
    print("=" * 80)
    print("🚀 Healthcare MoE Data Pipeline - Full Orchestration")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_config(config_path)
    pipeline_config = config.get('pipeline', {})
    output_dir = pipeline_config.get('output_dir', './data/arxiv')
    max_papers = pipeline_config.get('max_papers', 30000)
    skip_stages = pipeline_config.get('skip_stages', [])
    resume = pipeline_config.get('resume', True)
    
    # Run diagnostics
    print("🔍 Running diagnostics...")
    arxiv_ok = print_diagnostics(config)
    
    if not arxiv_ok:
        print("\n⚠️  ArXiv connection check failed. Continuing anyway...")
        print("   If collection fails, you can use local test data or retry later")
    
    # Pipeline stages
    stages = {
        'collect': 'Collect ArXiv Papers',
        'extract': 'Extract PDF Texts',
        'curate': 'NeMo Curator Curation',
        'preprocess': 'Preprocess & Classify',
        'tokenize': 'Train Tokenizer'
    }
    
    stage_results = {}
    start_time = time.time()
    
    try:
        # Stage 1: Collect papers
        if 'collect' not in skip_stages:
            stage_name = 'collect'
            print("\n" + "=" * 80)
            print(f"📦 Stage 1: {stages[stage_name]}")
            print("=" * 80)
            
            stage_start = time.time()
            collection_config = config.get('collection', {})
            rate_limit = collection_config.get('rate_limit', 0.33)
            retry_max = collection_config.get('retry_max', 5)
            
            cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
            
            try:
                collect_arxiv_papers(
                    output_dir=output_dir,
                    max_papers=max_papers,
                    cache_file=cache_file,
                    rate_limit_delay=1.0 / rate_limit
                )
                
                # Validate collection
                if os.path.exists(cache_file):
                    count = sum(1 for line in open(cache_file) if line.strip())
                    if count == 0:
                        print("⚠️  Warning: 0 papers collected")
                        print("   This might be due to:")
                        print("   - ArXiv API rate limiting")
                        print("   - Network issues")
                        print("   - Query syntax problems")
                        print("   - All papers filtered out")
                        stage_results[stage_name] = {'success': False, 'papers': 0}
                    else:
                        stage_results[stage_name] = {'success': True, 'papers': count}
                        print(f"✅ Collected {count} papers")
                else:
                    stage_results[stage_name] = {'success': False, 'papers': 0}
                    
            except Exception as e:
                print(f"❌ Stage 1 failed: {e}")
                import traceback
                print(traceback.format_exc())
                stage_results[stage_name] = {'success': False, 'error': str(e)}
                if not resume:
                    raise
            
            stage_elapsed = time.time() - stage_start
            print(f"⏱️  Stage 1 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\n⏭️  Skipping stage: {stages['collect']}")
            stage_results['collect'] = {'success': True, 'skipped': True}
        
        # Stage 2: Extract PDFs
        if 'extract' not in skip_stages:
            stage_name = 'extract'
            print("\n" + "=" * 80)
            print(f"📦 Stage 2: {stages[stage_name]}")
            print("=" * 80)
            
            stage_start = time.time()
            extraction_config = config.get('extraction', {})
            text_dir = os.path.join(output_dir, "texts")
            cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
            
            if not os.path.exists(cache_file):
                print(f"⚠️  Metadata file not found: {cache_file}")
                print("   Skipping extraction stage")
                stage_results[stage_name] = {'success': False, 'error': 'No metadata file'}
            else:
                try:
                    extract_pdf_texts(
                        input_jsonl=cache_file,
                        output_dir=text_dir,
                        num_workers=extraction_config.get('workers', 2),
                        rate_limit_delay=extraction_config.get('rate_limit', 0.4)
                    )
                    
                    # Count extracted files
                    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')] if os.path.exists(text_dir) else []
                    stage_results[stage_name] = {'success': True, 'files': len(text_files)}
                    print(f"✅ Extracted {len(text_files)} text files")
                    
                except Exception as e:
                    print(f"❌ Stage 2 failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    stage_results[stage_name] = {'success': False, 'error': str(e)}
                    if not resume:
                        raise
            
            stage_elapsed = time.time() - stage_start
            print(f"⏱️  Stage 2 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\n⏭️  Skipping stage: {stages['extract']}")
            stage_results['extract'] = {'success': True, 'skipped': True}
        
        # Stage 3: NeMo Curator curation (optional)
        if 'curate' not in skip_stages:
            curation_config = config.get('curation', {})
            use_nemo = curation_config.get('use_nemo_curator', True) and NEMO_CURATOR_AVAILABLE
            
            if use_nemo:
                stage_name = 'curate'
                print("\n" + "=" * 80)
                print(f"📦 Stage 3: {stages[stage_name]}")
                print("=" * 80)
                
                stage_start = time.time()
                text_dir = os.path.join(output_dir, "texts")
                cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
                curated_file = os.path.join(output_dir, "curated_dataset.jsonl")
                
                if not os.path.exists(text_dir) or not os.listdir(text_dir):
                    print(f"⚠️  Text directory empty or missing: {text_dir}")
                    print("   Skipping curation stage")
                    stage_results[stage_name] = {'success': False, 'error': 'No text files'}
                else:
                    try:
                        curate_with_nemo(
                            text_dir=text_dir,
                            metadata_jsonl=cache_file,
                            output_jsonl=curated_file,
                            use_gpu=curation_config.get('use_gpu', False),
                            skip_dedup=curation_config.get('skip_deduplication', False),
                            min_relevance_score=curation_config.get('min_relevance_score', 0.5)
                        )
                        
                        if os.path.exists(curated_file):
                            count = sum(1 for _ in open(curated_file))
                            stage_results[stage_name] = {'success': True, 'papers': count}
                            print(f"✅ Curated {count} papers")
                        else:
                            stage_results[stage_name] = {'success': False, 'error': 'Output file not created'}
                            
                    except Exception as e:
                        print(f"❌ Stage 3 failed: {e}")
                        import traceback
                        print(traceback.format_exc())
                        stage_results[stage_name] = {'success': False, 'error': str(e)}
                        if not resume:
                            raise
                
                stage_elapsed = time.time() - stage_start
                print(f"⏱️  Stage 3 completed in {stage_elapsed:.1f}s")
            else:
                print(f"\n⏭️  Skipping NeMo Curator curation (not available or disabled)")
                stage_results['curate'] = {'success': True, 'skipped': True}
        else:
            print(f"\n⏭️  Skipping stage: {stages['curate']}")
            stage_results['curate'] = {'success': True, 'skipped': True}
        
        # Stage 4: Preprocess and classify
        if 'preprocess' not in skip_stages:
            stage_name = 'preprocess'
            print("\n" + "=" * 80)
            print(f"📦 Stage 4: {stages[stage_name]}")
            print("=" * 80)
            
            stage_start = time.time()
            preprocessing_config = config.get('preprocessing', {})
            
            # Determine input file (curated if available, otherwise raw)
            curated_file = os.path.join(output_dir, "curated_dataset.jsonl")
            cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
            text_dir = os.path.join(output_dir, "texts")
            processed_file = os.path.join(output_dir, "processed_dataset.jsonl")
            
            if os.path.exists(curated_file):
                input_file = curated_file
                print(f"   Using curated dataset: {curated_file}")
            elif os.path.exists(cache_file):
                input_file = cache_file
                print(f"   Using raw metadata: {cache_file}")
            else:
                print(f"⚠️  No input file found")
                stage_results[stage_name] = {'success': False, 'error': 'No input file'}
                input_file = None
            
            if input_file:
                try:
                    if os.path.exists(curated_file):
                        # Process curated dataset
                        process_curated_dataset(
                            input_jsonl=curated_file,
                            output_jsonl=processed_file,
                            num_workers=preprocessing_config.get('workers', 4)
                        )
                    else:
                        # Preprocess from text files
                        preprocess_and_classify(
                            metadata_jsonl=cache_file,
                            text_dir=text_dir,
                            output_jsonl=processed_file,
                            num_workers=preprocessing_config.get('workers', 4)
                        )
                    
                    if os.path.exists(processed_file):
                        count = sum(1 for _ in open(processed_file))
                        stage_results[stage_name] = {'success': True, 'papers': count}
                        print(f"✅ Processed {count} papers")
                    else:
                        stage_results[stage_name] = {'success': False, 'error': 'Output file not created'}
                        
                except Exception as e:
                    print(f"❌ Stage 4 failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    stage_results[stage_name] = {'success': False, 'error': str(e)}
                    if not resume:
                        raise
            
            stage_elapsed = time.time() - stage_start
            print(f"⏱️  Stage 4 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\n⏭️  Skipping stage: {stages['preprocess']}")
            stage_results['preprocess'] = {'success': True, 'skipped': True}
        
        # Stage 5: Train tokenizer
        if 'tokenize' not in skip_stages:
            stage_name = 'tokenize'
            print("\n" + "=" * 80)
            print(f"📦 Stage 5: {stages[stage_name]}")
            print("=" * 80)
            
            stage_start = time.time()
            tokenizer_config = config.get('tokenizer', {})
            processed_file = os.path.join(output_dir, "processed_dataset.jsonl")
            
            if not os.path.exists(processed_file):
                print(f"⚠️  Processed dataset not found: {processed_file}")
                stage_results[stage_name] = {'success': False, 'error': 'No processed dataset'}
            else:
                try:
                    train_healthcare_tokenizer(
                        input_jsonl=processed_file,
                        output_dir=output_dir,
                        vocab_size=tokenizer_config.get('vocab_size', 50000),
                        model_prefix=tokenizer_config.get('model_prefix', 'healthcare_tokenizer')
                    )
                    
                    model_file = os.path.join(output_dir, f"{tokenizer_config.get('model_prefix', 'healthcare_tokenizer')}.model")
                    if os.path.exists(model_file):
                        stage_results[stage_name] = {'success': True}
                        print(f"✅ Tokenizer trained: {model_file}")
                    else:
                        stage_results[stage_name] = {'success': False, 'error': 'Model file not created'}
                        
                except Exception as e:
                    print(f"❌ Stage 5 failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    stage_results[stage_name] = {'success': False, 'error': str(e)}
                    if not resume:
                        raise
            
            stage_elapsed = time.time() - stage_start
            print(f"⏱️  Stage 5 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\n⏭️  Skipping stage: {stages['tokenize']}")
            stage_results['tokenize'] = {'success': True, 'skipped': True}
        
        # Generate final report
        total_elapsed = time.time() - start_time
        generate_final_report(stage_results, output_dir, total_elapsed)
        
        print("\n" + "=" * 80)
        print("✅ Pipeline Complete!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        print(traceback.format_exc())
        generate_final_report(stage_results, output_dir, time.time() - start_time, error=str(e))
        return False


def generate_final_report(stage_results: Dict, output_dir: str, total_elapsed: float, error: str = None):
    """Generate comprehensive validation report.
    
    Args:
        stage_results: Dictionary with results from each stage
        output_dir: Output directory
        total_elapsed: Total time elapsed
        error: Error message if pipeline failed
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_elapsed_seconds': total_elapsed,
        'total_elapsed_hours': total_elapsed / 3600,
        'stages': stage_results,
        'error': error
    }
    
    # Collect statistics
    stats = {
        'papers_collected': 0,
        'papers_extracted': 0,
        'papers_curated': 0,
        'papers_processed': 0,
        'domain_distribution': defaultdict(int),
        'year_distribution': defaultdict(int),
        'file_sizes': {}
    }
    
    # Count papers from each stage
    if 'collect' in stage_results and stage_results['collect'].get('success'):
        stats['papers_collected'] = stage_results['collect'].get('papers', 0)
    
    if 'extract' in stage_results and stage_results['extract'].get('success'):
        stats['papers_extracted'] = stage_results['extract'].get('files', 0)
    
    if 'curate' in stage_results and stage_results['curate'].get('success'):
        stats['papers_curated'] = stage_results['curate'].get('papers', 0)
    
    if 'preprocess' in stage_results and stage_results['preprocess'].get('success'):
        stats['papers_processed'] = stage_results['preprocess'].get('papers', 0)
        
        # Try to load domain/year distribution from processed file
        processed_file = os.path.join(output_dir, "processed_dataset.jsonl")
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            doc = json.loads(line)
                            domains = doc.get('domains', [])
                            for domain in domains:
                                stats['domain_distribution'][domain] += 1
                            year = doc.get('year')
                            if year:
                                stats['year_distribution'][year] += 1
            except:
                pass
    
    # Get file sizes
    files_to_check = [
        ('metadata', os.path.join(output_dir, "arxiv_papers.jsonl")),
        ('curated', os.path.join(output_dir, "curated_dataset.jsonl")),
        ('processed', os.path.join(output_dir, "processed_dataset.jsonl")),
        ('tokenizer_model', os.path.join(output_dir, "healthcare_tokenizer.model")),
        ('tokenizer_vocab', os.path.join(output_dir, "healthcare_tokenizer.vocab"))
    ]
    
    for name, path in files_to_check:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            stats['file_sizes'][name] = size_mb
    
    report['statistics'] = stats
    
    # Save report
    report_file = os.path.join(output_dir, 'pipeline_report.json')
    report_text_file = os.path.join(output_dir, 'pipeline_report.txt')
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Human-readable report
        with open(report_text_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Healthcare MoE Data Pipeline - Final Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Total Time: {report['total_elapsed_hours']:.2f} hours ({report['total_elapsed_seconds']:.1f} seconds)\n\n")
            
            f.write("Stage Results:\n")
            f.write("-" * 80 + "\n")
            for stage, result in stage_results.items():
                status = "✅ Success" if result.get('success') else "❌ Failed"
                f.write(f"{stage}: {status}\n")
                if 'papers' in result:
                    f.write(f"  Papers: {result['papers']}\n")
                if 'files' in result:
                    f.write(f"  Files: {result['files']}\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                f.write("\n")
            
            f.write("Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Papers Collected: {stats['papers_collected']}\n")
            f.write(f"Papers Extracted: {stats['papers_extracted']}\n")
            f.write(f"Papers Curated: {stats['papers_curated']}\n")
            f.write(f"Papers Processed: {stats['papers_processed']}\n\n")
            
            if stats['domain_distribution']:
                f.write("Domain Distribution:\n")
                for domain, count in sorted(stats['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {domain}: {count}\n")
                f.write("\n")
            
            if stats['year_distribution']:
                f.write("Year Distribution:\n")
                for year in sorted(stats['year_distribution'].keys()):
                    f.write(f"  {year}: {stats['year_distribution'][year]}\n")
                f.write("\n")
            
            if stats['file_sizes']:
                f.write("File Sizes:\n")
                for name, size_mb in stats['file_sizes'].items():
                    f.write(f"  {name}: {size_mb:.2f} MB\n")
        
        print(f"\n📊 Final report saved:")
        print(f"   JSON: {report_file}")
        print(f"   Text: {report_text_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ArXiv paper collection and PDF text extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect ArXiv papers
  python data_pipeline.py collect --max-papers 40000
  
  # Extract PDF texts from collected papers
  python data_pipeline.py extract --input ./data/arxiv/arxiv_papers.jsonl --output-dir ./data/arxiv/texts
  
  # Extract with custom workers and rate limit
  python data_pipeline.py extract --input ./data/arxiv/arxiv_papers.jsonl --output-dir ./data/arxiv/texts --workers 4 --rate-limit 0.3
  
  # Preprocess and classify domains
  python data_pipeline.py preprocess --metadata ./data/arxiv/arxiv_papers.jsonl --text-dir ./data/arxiv/texts --output ./data/arxiv/processed_dataset.jsonl
  
  # Curate with NeMo Curator
  python data_pipeline.py curate --text-dir ./data/arxiv/texts --metadata ./data/arxiv/arxiv_papers.jsonl --output ./data/arxiv/curated_dataset.jsonl
  
  # Train SentencePiece tokenizer
  python data_pipeline.py tokenize --input ./data/arxiv/curated_dataset.jsonl --output-dir ./data/arxiv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect ArXiv papers')
    collect_parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/arxiv',
        help='Output directory for collected papers (default: ./data/arxiv)'
    )
    collect_parser.add_argument(
        '--max-papers',
        type=int,
        default=40000,
        help='Maximum number of papers to collect (default: 40000)'
    )
    collect_parser.add_argument(
        '--cache-file',
        type=str,
        default=None,
        help='Path to cache file (default: output_dir/arxiv_papers.jsonl)'
    )
    collect_parser.add_argument(
        '--rate-limit',
        type=float,
        default=3.0,
        help='Rate limit in requests per second (default: 3.0)'
    )
    collect_parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Papers per batch for RAM-efficient collection (default: 10)'
    )
    collect_parser.add_argument(
        '--ram-target',
        type=float,
        default=50.0,
        help='Target RAM percentage to stay below (default: 50.0)'
    )
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract text from PDFs')
    extract_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file with paper metadata'
    )
    extract_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for extracted text files'
    )
    extract_parser.add_argument(
        '--workers',
        type=int,
        default=3,
        choices=[2, 3, 4],
        help='Number of worker threads (default: 3)'
    )
    extract_parser.add_argument(
        '--rate-limit',
        type=float,
        default=PDF_RATE_LIMIT,
        help=f'Rate limit delay in seconds (default: {PDF_RATE_LIMIT})'
    )
    
    # Curate command (NeMo Curator)
    curate_parser = subparsers.add_parser('curate', help='Curate text using NeMo Curator')
    curate_parser.add_argument(
        '--text-dir',
        type=str,
        required=True,
        help='Directory containing raw text files from extract step'
    )
    curate_parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Input JSONL file with paper metadata'
    )
    curate_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output curated dataset JSONL file'
    )
    curate_parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for deduplication (if available)'
    )
    curate_parser.add_argument(
        '--skip-dedup',
        action='store_true',
        help='Skip deduplication stage (for memory-constrained environments)'
    )
    curate_parser.add_argument(
        '--min-relevance-score',
        type=float,
        default=0.5,
        help='Minimum domain relevance score to keep (default: 0.5)'
    )
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess text and classify domains')
    preprocess_parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Input JSONL file with paper metadata'
    )
    preprocess_parser.add_argument(
        '--text-dir',
        type=str,
        required=True,
        help='Directory containing extracted text files'
    )
    preprocess_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    preprocess_parser.add_argument(
        '--workers',
        type=int,
        default=PREPROCESS_WORKERS,
        choices=[2, 3, 4, 6, 8],
        help=f'Number of parallel workers (default: {PREPROCESS_WORKERS})'
    )
    
    # Process command (post-NeMo Curator processing)
    process_parser = subparsers.add_parser('process', help='Process curated dataset with healthcare-specific preprocessing')
    process_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input curated dataset JSONL file (from curate command)'
    )
    process_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output processed dataset JSONL file'
    )
    process_parser.add_argument(
        '--workers',
        type=int,
        default=4,
        choices=[2, 4, 6, 8],
        help='Number of parallel workers (default: 4)'
    )
    
    # Tokenizer command
    tokenizer_parser = subparsers.add_parser('tokenize', help='Train SentencePiece BPE tokenizer')
    tokenizer_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file with processed papers'
    )
    tokenizer_parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/arxiv',
        help='Output directory for tokenizer files (default: ./data/arxiv)'
    )
    tokenizer_parser.add_argument(
        '--model-prefix',
        type=str,
        default='healthcare_tokenizer',
        help='Prefix for tokenizer model files (default: healthcare_tokenizer)'
    )
    tokenizer_parser.add_argument(
        '--vocab-size',
        type=int,
        default=TOKENIZER_VOCAB_SIZE,
        help=f'Vocabulary size (default: {TOKENIZER_VOCAB_SIZE})'
    )
    
    # Pipeline command (full orchestration)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full end-to-end pipeline')
    pipeline_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config.yaml file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'pipeline':
        run_full_pipeline(config_path=args.config)
    elif args.command == 'collect':
        rate_limit_delay = 1.0 / args.rate_limit
        collect_arxiv_papers(
            output_dir=args.output_dir,
            max_papers=args.max_papers,
            cache_file=args.cache_file,
            rate_limit_delay=rate_limit_delay,
            batch_size=args.batch_size,
            ram_target=args.ram_target
        )
    elif args.command == 'extract':
        extract_pdf_texts(
            input_jsonl=args.input,
            output_dir=args.output_dir,
            num_workers=args.workers,
            rate_limit_delay=args.rate_limit
        )
    elif args.command == 'preprocess':
        preprocess_and_classify(
            metadata_jsonl=args.metadata,
            text_dir=args.text_dir,
            output_jsonl=args.output,
            num_workers=args.workers
        )
    elif args.command == 'curate':
        curate_with_nemo(
            text_dir=args.text_dir,
            metadata_jsonl=args.metadata,
            output_jsonl=args.output,
            use_gpu=args.use_gpu,
            skip_dedup=args.skip_dedup,
            min_relevance_score=args.min_relevance_score
        )
    elif args.command == 'tokenize':
        train_healthcare_tokenizer(
            input_jsonl=args.input,
            output_dir=args.output_dir,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
