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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional, List
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Use CORRECT imports as specified:
# - from nemo_curator.datasets import DocumentDataset
# - from nemo_curator.modifiers import DocumentModifier
# - from nemo_curator.filters import DocumentFilter
# - from nemo_curator.utils.decorators import log_stage
try:
    import platform
    if platform.system() == 'Linux':
        # Core NeMo Curator imports (CORRECT paths)
        from nemo_curator.datasets import DocumentDataset
        from nemo_curator.modifiers import DocumentModifier
        from nemo_curator.filters import DocumentFilter
        from nemo_curator.utils.decorators import log_stage
        
        # Additional components (try multiple import paths for compatibility)
        ScoreFilter = None
        WordCountFilter = None
        AlphanumericFilter = None
        LanguageFilter = None
        RepeatedLineFilter = None
        FuzzyDedup = None
        get_client = None
        
        # Try to import additional filters
        try:
            from nemo_curator.filters import (
                ScoreFilter, WordCountFilter, AlphanumericFilter,
                LanguageFilter, RepeatedLineFilter
            )
        except ImportError:
            try:
                # Alternative import path
                from nemo_curator import (
                    ScoreFilter, WordCountFilter, AlphanumericFilter,
                    LanguageFilter, RepeatedLineFilter
                )
            except ImportError:
                pass  # Use None, will fall back to custom implementations
        
        # Try to import deduplication
        try:
            from nemo_curator.dedup import FuzzyDedup
        except ImportError:
            try:
                from nemo_curator import FuzzyDedup
            except ImportError:
                pass  # Use None, will skip deduplication
        
        # Try to import Dask client utility
        try:
            from nemo_curator.utils.distributed_utils import get_client
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
            except ImportError:
                pass  # Will create client manually
        
        # Try to import ProcessingStage and Stage for custom stages
        try:
            from nemo_curator.stages import ProcessingStage, Stage
            ProcessingStage_AVAILABLE = True
            Stage_AVAILABLE = True
        except ImportError:
            ProcessingStage = None
            ProcessingStage_AVAILABLE = False
            try:
                from nemo_curator.stages import Stage
                Stage_AVAILABLE = True
            except ImportError:
                Stage = None
                Stage_AVAILABLE = False
        
        # Try to import ArxivDownloadExtractStage and JsonlWriter
        try:
            from nemo_curator.stages import ArxivDownloadExtractStage
            ArxivDownloadExtractStage_AVAILABLE = True
        except ImportError:
            ArxivDownloadExtractStage = None
            ArxivDownloadExtractStage_AVAILABLE = False
        
        try:
            from nemo_curator.stages import JsonlWriter
            JsonlWriter_AVAILABLE = True
        except ImportError:
            JsonlWriter = None
            JsonlWriter_AVAILABLE = False
        
        try:
            from nemo_curator import Pipeline
            Pipeline_AVAILABLE = True
        except ImportError:
            Pipeline = None
            Pipeline_AVAILABLE = False
        
        import dask
        NEMO_CURATOR_AVAILABLE = True
        print("✅ NeMo Curator imported successfully")
    else:
        NEMO_CURATOR_AVAILABLE = False
        print("⚠️  NeMo Curator only supports Linux systems (current: {})".format(platform.system()))
except (ImportError, ValueError) as e:
    NEMO_CURATOR_AVAILABLE = False
    print("⚠️  nemo-curator package not available. Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'")
    print(f"   Error: {e}")


# Rate limiting: 3 requests per second
RATE_LIMIT_DELAY = 1.0 / 3.0  # ~0.33 seconds between requests
CHECKPOINT_INTERVAL = 5000  # Save checkpoint every 5000 papers
LOG_INTERVAL = 500  # Log progress every 500 papers

# Target date range
MIN_YEAR = 2015
MAX_YEAR = 2024

# ArXiv search queries
ARXIV_QUERIES = [
    "cat:cs.LG AND (healthcare OR medical OR clinical)",
    "cat:cs.AI AND (neurodegeneration OR disease)",
    "cat:q-bio.NC AND (machine learning)",
]

# Output fields (minimal metadata)
OUTPUT_FIELDS = ['id', 'title', 'abstract', 'year', 'categories', 'pdf_url']


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
    
    # Skip papers outside target date range
    if year is not None and (year < MIN_YEAR or year > MAX_YEAR):
        return None
    
    # Format categories
    categories = [cat.term for cat in paper.categories] if paper.categories else []
    
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


def search_arxiv_query(
    query: str,
    max_results: int = 10000,
    existing_ids: Set[str] = None,
    rate_limit_delay: float = RATE_LIMIT_DELAY
) -> list[Dict]:
    """Search ArXiv with a single query and return papers.
    
    Args:
        query: ArXiv search query
        max_results: Maximum number of results to fetch
        existing_ids: Set of IDs to skip (already in cache)
        rate_limit_delay: Delay between requests (seconds)
        
    Returns:
        List of paper dictionaries (streamed, not all in memory)
    """
    if existing_ids is None:
        existing_ids = set()
    
    papers = []
    print(f"\n🔍 Searching ArXiv: {query}")
    print(f"   Max results: {max_results}")
    
    try:
        # Create ArXiv client with rate limiting
        client = arxiv.Client(
            page_size=100,  # Fetch 100 papers per request
            delay_seconds=rate_limit_delay,
            num_retries=3
        )
        
        # Search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # Stream results (yield immediately, don't accumulate)
        count = 0
        skipped_no_abstract = 0
        skipped_date_range = 0
        skipped_duplicate = 0
        
        print(f"   🔄 Fetching results from ArXiv API...")
        results_iter = client.results(search)
        
        result_idx = -1
        for result_idx, result in enumerate(results_iter):
            # Rate limiting (client handles this, but we add extra safety)
            time.sleep(rate_limit_delay)
            
            # Check if already in cache
            paper_id = result.entry_id.split('/')[-1]
            if paper_id in existing_ids:
                skipped_duplicate += 1
                continue
            
            # Convert to dict
            paper_dict = paper_to_dict(result)
            if paper_dict is None:
                # Check why it was skipped
                if not result.summary or not result.summary.strip():
                    skipped_no_abstract += 1
                else:
                    year = extract_year_from_date(result.published)
                    if year is not None and (year < MIN_YEAR or year > MAX_YEAR):
                        skipped_date_range += 1
                continue
            
            papers.append(paper_dict)
            count += 1
            
            # Log progress every 500 papers
            if count % LOG_INTERVAL == 0:
                print(f"   ✅ Collected {count} new papers from this query...")
            
            # Safety limit: don't process more than max_results
            if count >= max_results:
                break
        
        print(f"   ✅ Query complete: {count} new papers collected")
        if skipped_duplicate > 0:
            print(f"   ⏭️  Skipped {skipped_duplicate} duplicates")
        if skipped_no_abstract > 0:
            print(f"   ⏭️  Skipped {skipped_no_abstract} papers without abstracts")
        if skipped_date_range > 0:
            print(f"   ⏭️  Skipped {skipped_date_range} papers outside date range ({MIN_YEAR}-{MAX_YEAR})")
        
        if count == 0:
            if result_idx == -1:
                print(f"   ⚠️  Warning: No results found for query. This might indicate:")
                print(f"      - Query syntax issue")
                print(f"      - No papers match the criteria")
                print(f"      - ArXiv API issue")
            else:
                print(f"   ⚠️  Warning: Processed {result_idx + 1} results but none matched criteria")
                print(f"      - All papers may have been filtered out (no abstract, wrong date range, duplicates)")
        
    except Exception as e:
        print(f"   ❌ Error in query '{query}': {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        print(f"   Continuing with {len(papers)} papers collected so far...")
    
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


def collect_arxiv_papers(
    output_dir: str = "./data/arxiv",
    max_papers: int = 40000,
    cache_file: str = None,
    rate_limit_delay: float = RATE_LIMIT_DELAY
):
    """Main function to collect ArXiv papers.
    
    Args:
        output_dir: Directory to save output files
        max_papers: Maximum total papers to collect
        cache_file: Path to cache file (default: output_dir/arxiv_papers.jsonl)
        rate_limit_delay: Delay between API requests (seconds)
    """
    if not ARXIV_AVAILABLE:
        error_msg = "❌ Error: arxiv package not available. Install with: pip install arxiv"
        print(error_msg)
        raise ImportError(error_msg)
    
    # Setup paths
    os.makedirs(output_dir, exist_ok=True)
    if cache_file is None:
        cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
    
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    print("=" * 60)
    print("📚 ArXiv Paper Collector")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Cache file: {cache_file}")
    print(f"🎯 Target: {max_papers} papers")
    print(f"📅 Date range: {MIN_YEAR}-{MAX_YEAR}")
    print(f"⏱️  Rate limit: {1.0/rate_limit_delay:.1f} requests/sec")
    print()
    
    # Load existing cache
    existing_ids = load_existing_ids(cache_file)
    total_collected = len(existing_ids)
    
    if total_collected >= max_papers:
        print(f"✅ Already have {total_collected} papers (target: {max_papers})")
        print("   Collection complete!")
        return
    
    # If cache file exists but is empty, reset it
    if total_collected == 0 and os.path.exists(cache_file):
        print(f"⚠️  Cache file exists but is empty. Starting fresh collection...")
        # Open in write mode to clear it, then switch to append mode
        with open(cache_file, 'w', encoding='utf-8') as f:
            pass  # Clear the file
        existing_ids = set()
        total_collected = 0
    
    print(f"📊 Starting collection: {total_collected}/{max_papers} papers already cached")
    print()
    
    # Open cache file for streaming writes
    cache_file_handle = open(cache_file, 'a', encoding='utf-8')
    
    # Track statistics
    total_new_papers = 0
    papers_by_query = defaultdict(int)
    checkpoint_counter = 0
    
    try:
        for query_idx, query in enumerate(ARXIV_QUERIES, 1):
            if total_collected + total_new_papers >= max_papers:
                print(f"\n✅ Target reached! Collected {total_collected + total_new_papers} papers")
                break
            
            # Calculate how many more papers we need
            remaining = max_papers - (total_collected + total_new_papers)
            query_max = min(remaining * 2, 15000)  # Fetch extra to account for deduplication
            
            # Search this query
            print(f"\n🔍 Processing query {query_idx}/{len(ARXIV_QUERIES)}: {query}")
            query_papers = search_arxiv_query(
                query=query,
                max_results=query_max,
                existing_ids=existing_ids,
                rate_limit_delay=rate_limit_delay
            )
            
            if not query_papers:
                print(f"   ⚠️  No papers returned from query. Skipping...")
                papers_by_query[query] = 0
                continue
            
            print(f"   📊 Received {len(query_papers)} papers from query")
            
            # Stream papers to disk immediately (deduplicate and write)
            seen_in_query = set()
            query_paper_count = 0
            
            for paper in query_papers:
                paper_id = paper['id']
                
                # Skip if already seen
                if paper_id in seen_in_query or paper_id in existing_ids:
                    continue
                
                # Mark as seen
                seen_in_query.add(paper_id)
                existing_ids.add(paper_id)
                
                # Stream to disk immediately
                cache_file_handle.write(json.dumps(paper, ensure_ascii=False) + '\n')
                cache_file_handle.flush()  # Ensure immediate write
                
                query_paper_count += 1
                total_new_papers += 1
                checkpoint_counter += 1
                
                # Log progress every 500 papers
                if total_new_papers % LOG_INTERVAL == 0:
                    print(f"   📊 Progress: {total_new_papers} new papers collected...")
                
                # Checkpoint: save metadata every CHECKPOINT_INTERVAL papers
                if checkpoint_counter >= CHECKPOINT_INTERVAL:
                    checkpoint_data = {
                        'timestamp': datetime.now().isoformat(),
                        'total_papers': total_collected + total_new_papers,
                        'new_papers': total_new_papers,
                        'queries_completed': query_idx,
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                    
                    print(f"\n💾 Checkpoint: {total_new_papers} papers written to disk")
                    checkpoint_counter = 0
                
                # Check if target reached
                if total_collected + total_new_papers >= max_papers:
                    break
            
            papers_by_query[query] = query_paper_count
            print(f"   📊 Query {query_idx}/{len(ARXIV_QUERIES)}: {query_paper_count} unique papers")
            print(f"   📊 Total new papers: {total_new_papers}")
    
    finally:
        # Close file handle
        cache_file_handle.close()
        
        # Final checkpoint
        if checkpoint_counter > 0:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'total_papers': total_collected + total_new_papers,
                'new_papers': total_new_papers,
                'queries_completed': len(ARXIV_QUERIES),
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"\n💾 Final checkpoint: {total_new_papers} papers written to disk")
    
    # Final statistics
    final_count = load_existing_ids(cache_file)
    print(f"\n" + "=" * 60)
    print("✅ Collection Complete!")
    print("=" * 60)
    print(f"📊 Total papers collected: {len(final_count)}")
    print(f"📁 Output file: {cache_file}")
    
    # Print breakdown by query
    print(f"\n📊 Breakdown by query:")
    for query, papers in papers_by_query.items():
        print(f"   {query[:50]}...: {len(papers)} papers")
    
    # Print year distribution
    print(f"\n📅 Loading year distribution...")
    year_counts = defaultdict(int)
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    paper = json.loads(line)
                    year = paper.get('year')
                    if year:
                        year_counts[year] += 1
                except:
                    continue
    
    print(f"📅 Year distribution:")
    for year in sorted(year_counts.keys()):
        print(f"   {year}: {year_counts[year]} papers")


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
    class HealthcareTextCleaner(DocumentModifier):
        def __init__(self):
            if NEMO_CURATOR_AVAILABLE:
                super().__init__()
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


class HealthcareDomainFilter(DocumentFilter if NEMO_CURATOR_AVAILABLE else object):
    """Custom domain classifier for healthcare papers extending NeMo Curator DocumentFilter interface.
    
    Scores documents based on healthcare+ML domain relevance and assigns domain tags.
    Compatible with NeMo Curator's ScoreFilter.
    
    Extends: nemo_curator.filters.DocumentFilter
    """
    
    def __init__(self):
        """Initialize domain filter with healthcare vocabulary."""
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
        print("❌ Error: NeMo Curator not available.")
        print("   Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'")
        print("   Note: NeMo Curator only supports Linux systems")
        return
    
    print("=" * 60)
    print("🔬 NeMo Curator Text Curation Pipeline")
    print("=" * 60)
    print(f"📁 Text directory: {text_dir}")
    print(f"📁 Metadata file: {metadata_jsonl}")
    print(f"📁 Output file: {output_jsonl}")
    print(f"🎯 Min relevance score: {min_relevance_score}")
    print(f"🔧 GPU deduplication: {use_gpu}")
    print(f"🔧 Skip deduplication: {skip_dedup}")
    print()
    
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
        client.close()
    except:
        pass


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
        print("❌ No papers found in input file!")
        return
    
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
        print("❌ Tokenizer training failed!")
        return
    
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
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        rate_limit_delay = 1.0 / args.rate_limit
        collect_arxiv_papers(
            output_dir=args.output_dir,
            max_papers=args.max_papers,
            cache_file=args.cache_file,
            rate_limit_delay=rate_limit_delay
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
