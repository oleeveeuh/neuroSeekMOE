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
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# For query analysis
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("requests package not available. Install with: pip install requests")

# Configuration management
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("yaml package not available. Install with: pip install pyyaml")

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil package not available. Install with: pip install psutil")

try:
    import arxiv
    ARXIV_AVAILABLE = True
    # Try to import specific exceptions for better error handling
    try:
        from arxiv import UnexpectedEmptyPageError
    except ImportError:
        UnexpectedEmptyPageError = None
except ImportError:
    ARXIV_AVAILABLE = False
    UnexpectedEmptyPageError = None
    print("arxiv package not available. Install with: pip install arxiv")

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
        print("PDF library not available. Install with: pip install PyPDF2 or pip install pdfplumber")

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("sentencepiece package not available. Install with: pip install sentencepiece")

# NeMo Curator imports (optional, Linux only)
NEMO_CURATOR_AVAILABLE = False
try:
    import platform
    if platform.system() == 'Linux':
        # Core Pipeline API (CORRECT path)
        try:
            from nemo_curator.pipeline import Pipeline
            from nemo_curator.stages.text.io.reader import JsonlReader
            from nemo_curator.stages.text.modules import ScoreFilter
            NEMO_CURATOR_AVAILABLE = True
        except ImportError as e:
            print(f"NeMo Curator core Pipeline API not available: {e}")
            NEMO_CURATOR_AVAILABLE = False
except Exception as e:
    print(f"NeMo Curator import failed: {e}")
    NEMO_CURATOR_AVAILABLE = False

# PDF extraction rate limiting (2.5 requests/sec = 0.4s between requests)
PDF_RATE_LIMIT = 0.4

# Default queries for different categories
DEFAULT_QUERIES = [
    # Machine Learning + Healthcare
    "cat:cs.LG AND (healthcare OR medical OR clinical OR medicine)",
    "cat:cs.LG AND (diagnosis OR disease OR treatment OR patient)",
    "cat:cs.LG AND (neuroscience OR brain OR neural OR eeg)",

    # AI + Healthcare
    "cat:cs.AI AND (healthcare OR medical OR clinical OR medicine)",
    "cat:cs.AI AND (diagnosis OR disease OR treatment OR patient)",

    # Computer Vision + Medical
    "cat:cs.CV AND (medical OR healthcare OR clinical OR diagnosis)",
    "cat:cs.CV AND (segmentation OR detection) AND (disease OR medical)",

    # Bioinformatics
    "cat:q-bio.NC AND (machine learning OR deep learning OR neural)",
    "cat:q-bio.GN AND (computational OR analysis OR prediction)",

    # Computational Biology
    "cat:q-bio.QM AND (molecular OR drug OR protein OR structure)",
    "cat:q-bio.CB AND (healthcare OR medicine OR therapeutic)",

    # Signal Processing + Medical
    "cat:eess.SP AND (medical OR healthcare OR clinical OR eeg)",

    # NLP + Medical
    "cat:cs.CL AND (medical OR healthcare OR clinical OR medicine)",
]

# Optimized queries for better results
OPTIMIZED_QUERIES = {
    "tier1": [
        # High-quality healthcare ML papers (primary targets)
        ("cat:cs.LG AND (healthcare OR medical OR clinical) AND (diagnosis OR prediction OR treatment)", 2000),
        ("cat:cs.LG AND (neuroscience OR brain) AND (eeg OR fmri OR imaging)", 1500),
        ("cat:cs.AI AND (healthcare OR medical) AND (diagnosis OR decision)", 1500),
        ("cat:cs.CV AND (medical OR healthcare) AND (segmentation OR detection OR classification)", 1500),
    ],
    "tier2": [
        # Broader healthcare applications
        ("cat:cs.LG AND (drug OR protein OR molecular) AND (discovery OR design)", 1000),
        ("cat:q-bio.NC AND (machine learning OR deep learning)", 1500),
        ("cat:q-bio.QM AND (machine learning OR computational OR drug)", 1000),
        ("cat:cs.CL AND (medical OR healthcare) AND (text OR classification)", 1000),
    ],
    "tier3": [
        # General ML/AI with medical relevance
        ("cat:cs.LG AND (time series OR longitudinal) AND (medical OR health)", 800),
        ("cat:cs.AI AND (robotics OR surgical) AND (medical OR healthcare)", 800),
        ("cat:eess.SP AND (biomedical OR biosignal) AND (analysis OR processing)", 800),
    ]
}

# ============================================================================
# PDF EXTRACTION FUNCTIONS
# ============================================================================

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
        print("Error: PDF library not available.")
        print("   Install with: pip install PyPDF2 or pip install pdfplumber")
        return

    if not REQUESTS_AVAILABLE:
        print("Error: requests package not available.")
        print("   Install with: pip install requests")
        return

    # Validate num_workers
    if num_workers < 2 or num_workers > 4:
        print(f"Warning: num_workers={num_workers} not in recommended range (2-4)")
        print("   Adjusting to 3...")
        num_workers = 3

    os.makedirs(output_dir, exist_ok=True)

    # Progress tracking
    processed_file = os.path.join(output_dir, ".processed.txt")
    processed_ids = set()

    # Load previously processed papers
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r') as f:
                for line in f:
                    if line.strip():
                        processed_ids.add(line.strip())
            print(f"Found {len(processed_ids)} already processed papers")
        except Exception as e:
            print(f"Warning: Could not load processed file: {e}")

    # Count total papers
    total_papers = 0
    try:
        with open(input_jsonl, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        json.loads(line)
                        total_papers += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON on line {i+1}")
                        continue
        print(f"Total papers in file: {total_papers}")
    except Exception as e:
        print(f"Error counting papers: {e}")
        return

    # Check how many are already processed
    already_processed = len(processed_ids)
    remaining_to_process = total_papers - already_processed

    print(f"Already processed: {already_processed}")
    print(f"Remaining to process: {remaining_to_process}")

    if remaining_to_process == 0:
        print("All papers already processed!")
        return

    # Function to extract text from a single PDF
    def extract_single_pdf(paper_data: Dict) -> Tuple[str, bool, str]:
        """Extract text from a single PDF.

        Returns:
            Tuple of (paper_id, success, text_or_error)
        """
        try:
            paper_id = paper_data.get('id', '').split('/')[-1]
            if not paper_id:
                return ("unknown", False, "No paper ID")

            # Skip if already processed
            if paper_id in processed_ids:
                return (paper_id, True, "Already processed")

            pdf_url = paper_data.get('pdf_url')
            if not pdf_url:
                return (paper_id, False, "No PDF URL")

            # Download PDF with timeout and headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ArXiv PDF Extractor)',
                'Accept': 'application/pdf,application/octet-stream'
            }

            try:
                response = requests.get(
                    pdf_url,
                    timeout=30,  # 30 second timeout
                    headers=headers,
                    stream=True
                )
                response.raise_for_status()

                # Check if we got PDF content
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    return (paper_id, False, f"Unexpected content type: {content_type}")

                # Read PDF content
                pdf_content = response.content

                if len(pdf_content) == 0:
                    return (paper_id, False, "Empty PDF file")

                # Check PDF header
                if not pdf_content.startswith(b'%PDF'):
                    return (paper_id, False, "Invalid PDF header")

            except requests.exceptions.RequestException as e:
                return (paper_id, False, f"Download failed: {str(e)[:50]}")
            except Exception as e:
                return (paper_id, False, f"Download error: {str(e)[:50]}")

            # Extract text from PDF
            text_content = ""

            try:
                if USE_PDFPLUMBER:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                        text_content = ""
                        for page in pdf.pages[:20]:  # Limit to first 20 pages
                            try:
                                page_text = page.extract_text()
                                if page_text and len(page_text.strip()) > 10:
                                    text_content += page_text + "\n"
                            except Exception as e:
                                print(f"Warning: Error extracting page {page.page_number}: {e}")
                                continue
                else:
                    import PyPDF2
                    import io
                    with io.BytesIO(pdf_content) as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)
                        text_content = ""
                        for page_num in range(min(len(pdf_reader.pages), 20)):
                            try:
                                page = pdf_reader.pages[page_num]
                                page_text = page.extract_text()
                                if page_text and len(page_text.strip()) > 10:
                                    text_content += page_text + "\n"
                            except Exception as e:
                                print(f"Warning: Error extracting page {page_num}: {e}")
                                continue

                if not text_content or len(text_content.strip()) < 50:
                    return (paper_id, False, "No readable text extracted")

                # Clean up the text
                text_content = re.sub(r'\n{3,}', '\n\n', text_content)  # Reduce excessive newlines
                text_content = text_content.strip()

                # Save to file
                output_file = os.path.join(output_dir, f"{paper_id}.txt")
                with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(text_content)

                return (paper_id, True, f"Extracted {len(text_content)} characters")

            except Exception as e:
                return (paper_id, False, f"Text extraction failed: {str(e)[:50]}")

        except Exception as e:
            return (paper_id, False, f"Processing failed: {str(e)[:50]}")

    # Main extraction logic
    print(f"Starting extraction with {num_workers} workers...")
    print(f"Rate limit: {1.0/rate_limit_delay:.1f} requests/sec")

    success_count = 0
    fail_count = 0
    checkpoint_interval = 100

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        # Submit papers for processing
        with open(input_jsonl, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    paper_data = json.loads(line)
                    paper_id = paper_data.get('id', '').split('/')[-1]

                    if paper_id in processed_ids:
                        continue

                    # Submit to executor
                    future = executor.submit(extract_single_pdf, paper_data)
                    futures.append(future)

                    # Control submission rate
                    if len(futures) >= num_workers * 10:  # Don't queue too many ahead
                        # Wait for some to complete
                        completed = 0
                        for future in as_completed(futures[:num_workers]):
                            try:
                                pid, success, message = future.result()
                                if success:
                                    success_count += 1
                                    processed_ids.add(pid)
                                else:
                                    fail_count += 1
                                    print(f"Failed {pid}: {message}")
                                completed += 1
                            except Exception as e:
                                print(f"Future error: {e}")
                                fail_count += 1
                            futures.remove(future)

                        # Rate limiting
                        time.sleep(rate_limit_delay)

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue

        # Process remaining futures
        for future in as_completed(futures):
            try:
                pid, success, message = future.result()
                if success:
                    success_count += 1
                    processed_ids.add(pid)
                else:
                    fail_count += 1
                    print(f"Failed {pid}: {message}")

                # Checkpoint saving
                if (success_count + fail_count) % checkpoint_interval == 0:
                    try:
                        with open(processed_file, 'w') as f:
                            for pid in processed_ids:
                                f.write(f"{pid}\n")
                        print(f"Checkpoint saved: {success_count + fail_count} papers processed")
                    except Exception as e:
                        print(f"Warning: Could not save checkpoint: {e}")

            except Exception as e:
                print(f"Future error: {e}")
                fail_count += 1

    # Final save
    try:
        with open(processed_file, 'w') as f:
            for pid in processed_ids:
                f.write(f"{pid}\n")
    except Exception as e:
        print(f"Warning: Could not save final checkpoint: {e}")

    print(f"\nExtraction complete:")
    print(f"   Successfully extracted: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"   Success rate: {(success_count/(success_count+fail_count)*100):.1f}%" if (success_count+fail_count) > 0 else "N/A")


# ============================================================================
# PAPER COLLECTION FUNCTIONS
# ============================================================================

def collect_arxiv_papers(
    output_dir: str = "./data/arxiv",
    max_papers: int = 40000,
    cache_file: Optional[str] = None,
    rate_limit_delay: float = 3.0,
    batch_size: int = 1000,
    ram_target: float = 85.0,
    use_drive: bool = True
):
    """Collect ArXiv papers with healthcare focus.

    Args:
        output_dir: Directory to save papers
        max_papers: Maximum number of papers to collect
        cache_file: Optional cache file for deduplication
        rate_limit_delay: Delay between API requests (seconds)
        batch_size: Number of papers to collect per batch
        ram_target: Target RAM usage percentage (stops if exceeded)
        use_drive: Whether to use Google Drive (if available)
    """
    if not ARXIV_AVAILABLE:
        print("Error: arxiv package not available")
        print("Install with: pip install arxiv")
        return

    if use_drive:
        try:
            output_dir = get_drive_output_dir(output_dir)
        except:
            print("Using local output directory")

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "arxiv_papers.jsonl")

    # Use balanced collector if available
    try:
        print("Using balanced collector for even distribution across years...")
        collector = BalancedArxivCollector(
            output_dir=output_dir,
            target_papers=max_papers,
            rate_limit_delay=rate_limit_delay,
            checkpoint_interval=batch_size
        )
        collector.collect_balanced()
        return
    except Exception as e:
        print(f"Balanced collector failed, falling back to standard collection: {e}")

    # Standard collection fallback
    print(f"Starting ArXiv paper collection...")
    print(f"Output: {output_file}")
    print(f"Target: {max_papers} papers")
    print(f"Rate limit: {rate_limit_delay}s delay")

    collected_ids = set()
    collected_papers = []

    # Load existing collection if resuming
    if os.path.exists(output_file):
        print(f"Resuming from existing collection...")
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        paper = json.loads(line)
                        paper_id = paper.get('id', '')
                        if paper_id:
                            collected_ids.add(paper_id)
                            collected_papers.append(paper)

            print(f"Found {len(collected_papers)} existing papers")
            if len(collected_papers) >= max_papers:
                print("Target already reached!")
                return
        except Exception as e:
            print(f"Error loading existing collection: {e}")
            collected_papers = []

    # Initialize ArXiv client
    client = arxiv.Client(
        delay_seconds=rate_limit_delay,
        num_retries=3,
        page_size=100
    )

    total_collected = len(collected_papers)

    # Process queries
    for tier_name, tier_queries in OPTIMIZED_QUERIES.items():
        if total_collected >= max_papers:
            break

        print(f"\nProcessing {tier_name} queries...")

        for query, target in tier_queries:
            if total_collected >= max_papers:
                break

            print(f"Query: {query[:80]}...")

            try:
                search = arxiv.Search(
                    query=query,
                    max_results=min(target, max_papers - total_collected),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )

                batch_collected = 0
                for result in client.results(search):
                    if total_collected >= max_papers:
                        break

                    paper_id = result.entry_id

                    # Skip duplicates
                    if paper_id in collected_ids:
                        continue

                    # Extract paper data
                    paper_data = {
                        'id': paper_id,
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'published': result.published.isoformat() if hasattr(result.published, 'isoformat') else str(result.published),
                        'summary': result.summary,
                        'categories': result.categories,
                        'pdf_url': result.pdf_url,
                        'arxiv_url': result.entry_id,
                    }

                    collected_papers.append(paper_data)
                    collected_ids.add(paper_id)
                    batch_collected += 1
                    total_collected += 1

                print(f"  Collected {batch_collected} papers (total: {total_collected})")

                # Save batch
                if batch_collected > 0:
                    with open(output_file, 'a') as f:
                        for paper in collected_papers[-batch_collected:]:
                            f.write(json.dumps(paper, ensure_ascii=False) + '\n')

                # Check memory usage
                if PSUTIL_AVAILABLE:
                    ram_usage = psutil.virtual_memory().percent
                    if ram_usage > ram_target:
                        print(f"Warning: RAM usage at {ram_usage:.1f}% > {ram_target}%")
                        print("Stopping collection to prevent memory issues")
                        break

            except Exception as e:
                print(f"Error processing query: {e}")
                continue

    print(f"\nCollection complete!")
    print(f"Total papers collected: {total_collected}")
    print(f"Output file: {output_file}")


# ============================================================================
# NEMO CURATOR FUNCTIONS
# ============================================================================

def curate_with_nemo(
    text_dir: str,
    metadata_jsonl: str,
    output_jsonl: str,
    use_gpu: bool = False,
    skip_dedup: bool = False,
    min_relevance_score: float = 0.5
):
    """Curate extracted texts using NeMo Curator.

    Args:
        text_dir: Directory containing extracted text files
        metadata_jsonl: JSONL file with paper metadata
        output_jsonl: Output file for curated papers
        use_gpu: Whether to use GPU for processing
        skip_dedup: Whether to skip deduplication
        min_relevance_score: Minimum relevance score for healthcare content
    """
    if not NEMO_CURATOR_AVAILABLE:
        print("NeMo Curator not available. Please install on Linux system.")
        print("Install with: pip install 'nemo-curator[text]'")
        return

    print("Starting NeMo Curator processing...")
    print(f"Input text directory: {text_dir}")
    print(f"Input metadata: {metadata_jsonl}")
    print(f"Output file: {output_jsonl}")

    # This is a placeholder implementation
    # In a real scenario, you would use NeMo Curator's Pipeline API
    # For now, we'll do basic filtering and processing with streaming to disk

    curated_count = 0
    processed_count = 0
    start_time = time.time()

    print("Starting streaming curation...", flush=True)

    # Stream output directly to disk to avoid memory accumulation
    with open(output_jsonl, 'w', encoding='utf-8') as outfile:
        # Read metadata and filter one paper at a time
        with open(metadata_jsonl, 'r') as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue

                processed_count += 1

                # More frequent progress reporting for large datasets
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    papers_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {processed_count:,} papers, curated {curated_count:,} ({papers_per_sec:.1f} papers/sec)...", flush=True)

                # Additional milestone reporting every 1000 papers
                if processed_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    papers_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    eta_seconds = (50000 - processed_count) / papers_per_sec if papers_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60
                    print(f"📍 Milestone: {processed_count:,} papers | Rate: {papers_per_sec:.1f}/sec | ETA: {eta_minutes:.1f} min", flush=True)

                try:
                    paper = json.loads(line)
                    paper_id = paper.get('id', '').split('/')[-1]

                    # Check if corresponding text file exists
                    text_file = os.path.join(text_dir, f"{paper_id}.txt")
                    if not os.path.exists(text_file):
                        continue

                    # Read text content
                    with open(text_file, 'r', encoding='utf-8', errors='ignore') as tf:
                        text_content = tf.read()

                    if len(text_content.strip()) < 100:
                        continue

                    # Basic healthcare relevance check
                    healthcare_keywords = [
                        'healthcare', 'medical', 'clinical', 'patient', 'diagnosis',
                        'treatment', 'disease', 'medicine', 'health', 'hospital'
                    ]

                    text_lower = text_content.lower()
                    relevance_score = sum(1 for keyword in healthcare_keywords if keyword in text_lower)
                    relevance_score = min(relevance_score / len(healthcare_keywords), 1.0)

                    if relevance_score < min_relevance_score:
                        continue

                    # Create curated paper object
                    curated_paper = {
                        'arxiv_id': paper_id,
                        'title': paper.get('title', ''),
                        'authors': paper.get('authors', []),
                        'published': paper.get('published', ''),
                        'categories': paper.get('categories', []),
                        'text': text_content,
                        'relevance_score': relevance_score,
                        'text_length': len(text_content),
                        'curated_at': datetime.now().isoformat()
                    }

                    # Write directly to output file (streaming)
                    outfile.write(json.dumps(curated_paper, ensure_ascii=False) + '\n')
                    curated_count += 1

                    # Periodic flush to ensure data is written
                    if curated_count % 50 == 0:
                        outfile.flush()

                except Exception as e:
                    print(f"Error processing paper {line_num}: {e}")
                    continue

    # Final summary
    total_time = time.time() - start_time
    avg_rate = processed_count / total_time if total_time > 0 else 0
    retention_rate = (curated_count / processed_count * 100) if processed_count > 0 else 0

    print(f"\n{'='*60}")
    print(f"🎉 CURATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Total papers processed: {processed_count:,}")
    print(f"✅ Papers curated: {curated_count:,}")
    print(f"📈 Retention rate: {retention_rate:.1f}%")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"⚡ Average processing rate: {avg_rate:.1f} papers/second")
    print(f"💾 Output file: {output_jsonl}")
    print(f"📁 File size: {os.path.getsize(output_jsonl) / (1024*1024):.1f} MB")
    print(f"{'='*60}")


def run_nemo_curator_pipeline(
    output_path: str,
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
    """Run NeMo Curator pipeline with ArXiv download.

    Args:
        output_path: Final curated output path
        raw_data_path: Directory for raw downloaded data
        raw_output_path: Path for raw extracted output
        filter_query: Query for filtering papers
        max_workers: Number of workers for processing
        use_gpu: Whether to use GPU
        batch_size: Batch size for processing
        checkpoint_interval: Checkpoint interval
        max_papers: Maximum papers to process
        resume: Whether to resume from checkpoint
    """
    print("Running NeMo Curator pipeline with ArXiv download...")

    if not NEMO_CURATOR_AVAILABLE:
        print("NeMo Curator not available")
        return None

    # This is a placeholder - real implementation would use NeMo Curator's Pipeline API
    print("Placeholder: NeMo Curator pipeline would run here")
    print("This would download from ArXiv and process with various stages")

    # For now, return a simple success indicator
    return {"status": "completed", "papers_processed": 0}


def process_curated_dataset(
    input_jsonl: str,
    output_jsonl: str,
    num_workers: int = 4
):
    """Process curated dataset for training.

    Args:
        input_jsonl: Input curated JSONL file
        output_jsonl: Output processed JSONL file
        num_workers: Number of workers for processing
    """
    print(f"Processing curated dataset...")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_jsonl}")

    processed_papers = []

    with open(input_jsonl, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                paper = json.loads(line)

                # Extract domains from categories
                domains = []
                categories = paper.get('categories', [])
                for cat in categories:
                    if 'cs' in cat:
                        domains.append('computer_science')
                    elif 'q-bio' in cat:
                        domains.append('biology')
                    elif 'stat' in cat:
                        domains.append('statistics')
                    elif 'eess' in cat:
                        domains.append('engineering')

                if not domains:
                    domains = ['other']

                # Extract year from published date
                published = paper.get('published', '')
                year = 2024  # Default
                if published:
                    try:
                        if hasattr(published, 'year'):
                            year = published.year
                        else:
                            year = int(published.split('-')[0])
                    except:
                        pass

                # Process text
                text = paper.get('text', '')
                if len(text) > 100000:  # Truncate very long texts
                    text = text[:100000] + "..."

                processed_paper = {
                    'arxiv_id': paper.get('arxiv_id', ''),
                    'text': text,
                    'domains': domains,
                    'year': year,
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', [])[:10],  # Limit authors
                    'categories': categories,
                    'relevance_score': paper.get('relevance_score', 0.5),
                }

                processed_papers.append(processed_paper)

            except Exception as e:
                print(f"Error processing paper {line_num}: {e}")
                continue

    # Write processed output
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for paper in processed_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')

    print(f"Processed {len(processed_papers)} papers")
    print(f"Output: {output_jsonl}")


def train_healthcare_tokenizer(
    input_jsonl: str,
    output_dir: str,
    model_prefix: str = "healthcare_tokenizer",
    vocab_size: int = 32000
):
    """Train SentencePiece tokenizer on healthcare texts.

    Args:
        input_jsonl: Input JSONL file with processed papers
        output_dir: Output directory for tokenizer files
        model_prefix: Prefix for tokenizer model files
        vocab_size: Vocabulary size for tokenizer
    """
    if not SENTENCEPIECE_AVAILABLE:
        print("Error: sentencepiece not available")
        print("Install with: pip install sentencepiece")
        return

    print(f"Training SentencePiece tokenizer...")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_dir}")
    print(f"Model prefix: {model_prefix}")
    print(f"Vocab size: {vocab_size}")

    os.makedirs(output_dir, exist_ok=True)

    # Collect all text
    all_text = []
    with open(input_jsonl, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                paper = json.loads(line)
                text = paper.get('text', '')
                if text:
                    all_text.append(text)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    if not all_text:
        print("Error: No text found in input file")
        return

    print(f"Collected text from {len(all_text)} papers")

    # Write all text to temporary file for SentencePiece training
    temp_text_file = os.path.join(output_dir, "all_text.txt")
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        for text in all_text:
            f.write(text + '\n')

    # Train SentencePiece model
    model_prefix_path = os.path.join(output_dir, model_prefix)

    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=temp_text_file,
        model_prefix=model_prefix_path,
        vocab_size=vocab_size,
        model_type='bpe',
        max_sentence_length=4096,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
        character_coverage=0.995,
        num_threads=os.cpu_count() or 4
    )

    # Clean up temp file
    os.remove(temp_text_file)

    # Verify tokenizer was created
    model_file = f"{model_prefix_path}.model"
    vocab_file = f"{model_prefix_path}.vocab"

    if os.path.exists(model_file) and os.path.exists(vocab_file):
        print(f"Tokenizer trained successfully!")
        print(f"Model: {model_file}")
        print(f"Vocab: {vocab_file}")
    else:
        print("Error: Tokenizer files not created")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_drive_output_dir(local_output_dir: str = "./data/arxiv",
                        drive_base: str = "/content/drive/MyDrive/neuroMOE_results") -> str:
    """Get output directory, preferring Google Drive if available.

    Args:
        local_output_dir: Local directory path
        drive_base: Base directory in Google Drive

    Returns:
        Output directory path (Drive if available, otherwise local)
    """
    try:
        if is_colab_environment() and is_drive_mounted():
            # Use Drive
            drive_dir = os.path.join(drive_base, "arxiv_data")
            os.makedirs(drive_dir, exist_ok=True)
            return drive_dir
    except:
        pass

    # Use local directory
    os.makedirs(local_output_dir, exist_ok=True)
    return local_output_dir


def is_colab_environment() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except:
        return False


def is_drive_mounted(drive_path: str = "/content/drive/MyDrive") -> bool:
    """Check if Google Drive is mounted."""
    return os.path.exists(drive_path)


def optimize_and_execute_queries(
    base_queries: List[str],
    output_dir: str = "./data/arxiv",
    run_diagnostic: bool = True,
    run_collection: bool = True,
    max_papers_per_query: int = 1000,
    rate_limit_delay: float = 2.0
):
    """Optimize and execute queries for paper collection.

    Args:
        base_queries: List of base queries to optimize
        output_dir: Output directory for papers
        run_diagnostic: Whether to run diagnostic analysis
        run_collection: Whether to run actual collection
        max_papers_per_query: Maximum papers per query
        rate_limit_delay: Rate limit delay
    """
    print("Query optimization and execution pipeline")
    print(f"Base queries: {len(base_queries)}")
    print(f"Output dir: {output_dir}")

    if run_diagnostic:
        print("\nRunning diagnostic analysis...")
        # Analyze queries
        for i, query in enumerate(base_queries[:5]):  # Limit diagnostic
            try:
                search = arxiv.Search(query, max_results=10)
                results = list(search.results())
                print(f"Query {i+1}: Found {len(results)} sample results")
            except Exception as e:
                print(f"Query {i+1}: Error - {e}")

    if run_collection:
        print("\nRunning optimized collection...")
        # Simple implementation - just run balanced collector
        collect_arxiv_papers(
            output_dir=output_dir,
            max_papers=50000,
            rate_limit_delay=rate_limit_delay
        )


# ============================================================================
# BALANCED COLLECTOR (from the replacement file)
# ============================================================================

# Import the balanced collector from the replacement file
import io
import sys
from collections import defaultdict
from typing import Dict, Set, Optional, List, Tuple, Any

# Target distribution: evenly spread across years and categories
BALANCED_CONFIG = {
    "target_papers": 50000,
    "target_per_year": 7150,  # 50000 / 7 years (2020-2025, ~2020 onwards)
    "years_to_cover": [2020, 2021, 2022, 2023, 2024, 2025],
    "min_papers_per_year": 6000,  # Minimum acceptable
    "max_papers_per_year": 8500,  # Maximum acceptable
}

# Query structure: (query_string, target_papers_for_this_query, description)
BALANCED_QUERIES = {
    "primary_ml": [
        ("cat:cs.LG AND healthcare", 3000, "ML + Healthcare (general)"),
        ("cat:cs.LG AND (medical OR diagnosis OR clinical)", 3000, "ML + Medical/Clinical"),
        ("cat:cs.LG AND (neuroscience OR brain OR neuroimaging OR eeg OR fmri)", 2500, "ML + Neuroscience"),
        ("cat:cs.LG AND (drug OR protein OR molecular OR genetics)", 2500, "ML + Molecular/Drug"),
    ],
    "ai_specific": [
        ("cat:cs.AI AND (healthcare OR medical OR clinical)", 2500, "AI + Healthcare"),
        ("cat:cs.AI AND (neuroscience OR brain OR neural)", 2000, "AI + Brain/Neural"),
        ("cat:cs.AI AND (diagnosis OR prediction OR disease)", 2000, "AI + Diagnosis/Prediction"),
    ],
    "vision_imaging": [
        ("cat:cs.CV AND (medical OR diagnosis OR pathology OR imaging)", 2500, "CV + Medical Imaging"),
        ("cat:cs.CV AND (segmentation OR detection) AND (disease OR diagnostic)", 2000, "CV + Seg/Detection"),
    ],
    "bioinformatics": [
        ("cat:q-bio.NC AND (machine learning OR deep learning OR neural network)", 2500, "Neuro + ML"),
        ("cat:q-bio.NC AND (brain OR fmri OR connectivity OR analysis)", 2000, "Neuro + Brain Analysis"),
        ("cat:q-bio.QM AND (machine learning OR computational)", 1500, "BioMol + Computational"),
    ],
    "signal_processing": [
        ("cat:eess.SP AND (brain OR eeg OR medical OR diagnosis)", 1500, "Signal Processing + Medical"),
    ],
    "nlp_medical": [
        ("cat:cs.CL AND (medical OR clinical OR healthcare OR diagnosis)", 1500, "NLP + Medical"),
    ],
}


class BalancedArxivCollector:
    """
    Balanced collector that maintains even distribution across years and categories.

    Key improvements:
    1. Tracks per-year and per-category progress
    2. Uses intelligent year-splitting with deduplication
    3. Respects per-query design limits
    4. Implements checkpoint recovery
    5. Handles duplicates effectively
    """

    def __init__(
        self,
        output_dir: str = "./data/arxiv",
        target_papers: int = 50000,
        rate_limit_delay: float = 3.0,
        checkpoint_interval: int = 500,
    ):
        """
        Initialize the balanced collector.

        Args:
            output_dir: Directory for output files
            target_papers: Total target papers to collect (default 50,000)
            rate_limit_delay: Seconds between API requests (respects ArXiv recommendation of 3s)
            checkpoint_interval: Save checkpoint every N papers
        """
        self.output_dir = output_dir
        self.target_papers = target_papers
        self.rate_limit_delay = rate_limit_delay
        self.checkpoint_interval = checkpoint_interval

        # Output files
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, "arxiv_papers.jsonl")
        self.checkpoint_file = os.path.join(output_dir, "collection_checkpoint.json")
        self.stats_file = os.path.join(output_dir, "collection_stats.json")

        # State tracking
        self.collected_ids: Set[str] = set()
        self.papers_by_year: Dict[int, List[Dict]] = defaultdict(list)
        self.papers_by_category: Dict[str, List[Dict]] = defaultdict(list)
        self.total_collected = 0
        self.total_processed = 0
        self.duplicates_count = 0
        self.queries_executed = 0

        # ArXiv client (respecting rate limits)
        self.client = arxiv.Client(
            delay_seconds=3.0,
            num_retries=3,
            page_size=100
        )

        # Load checkpoint if exists
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load previous collection state from checkpoint."""
        if not os.path.exists(self.checkpoint_file):
            return

        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.collected_ids = set(checkpoint.get('collected_ids', []))
                self.total_collected = checkpoint.get('total_collected', 0)
                self.total_processed = checkpoint.get('total_processed', 0)
                self.duplicates_count = checkpoint.get('duplicates_count', 0)
                print(f"✓ Loaded checkpoint: {len(self.collected_ids)} existing papers")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        """Save current collection state."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'collected_ids': list(self.collected_ids),
            'total_collected': self.total_collected,
            'total_processed': self.total_processed,
            'duplicates_count': self.duplicates_count,
            'queries_executed': self.queries_executed,
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def collect_balanced(self) -> int:
        """
        Collect balanced 50,000 paper dataset.

        Strategy:
        1. Execute primary queries across all years (maintains category diversity)
        2. Monitor year-by-year distribution
        3. Fill gaps with supplementary queries
        4. Ensure ~7,150 papers per year across 2020-2025

        Returns:
            Total papers collected
        """
        print("\n" + "="*80)
        print("STARTING BALANCED ARXIV COLLECTION (50,000 PAPERS)")
        print("="*80)
        print(f"Target: {self.target_papers} papers")
        print(f"Target per year: {BALANCED_CONFIG['target_per_year']} papers")
        print(f"Coverage: {BALANCED_CONFIG['years_to_cover']}")
        print()

        # Phase 1: Primary balanced queries
        print("="*80)
        print("PHASE 1: Primary Balanced Queries")
        print("="*80)

        for category_name, category_queries in BALANCED_QUERIES.items():
            print(f"\n📁 Category: {category_name.upper()}")

            for query_str, target, description in category_queries:
                if self.total_collected >= self.target_papers:
                    print("✓ Target reached, stopping collection")
                    break

                self.queries_executed += 1
                print(f"\n  [{self.queries_executed}] {description}")
                print(f"      Query: {query_str[:70]}...")
                print(f"      Target: {target} papers")

                # Collect without year-split first (to establish baseline)
                new_papers, processed = self._collect_from_query(
                    query=query_str,
                    target_papers=target,
                    category=category_name,
                )

                dup_rate = (1 - new_papers/processed * 100) if processed > 0 else 0
                print(f"      → {new_papers} new papers ({processed} processed, {dup_rate:.1f}% dups)")
                print(f"      ✓ Total so far: {self.total_collected}/{self.target_papers}")

        # Phase 2: Year-balanced supplementary queries
        print("\n" + "="*80)
        print("PHASE 2: Year-Balanced Supplementary Queries")
        print("="*80)

        year_distribution = self._get_year_distribution()
        print("\nCurrent year distribution:")
        for year in BALANCED_CONFIG['years_to_cover']:
            count = year_distribution.get(year, 0)
            pct = (count / self.total_collected * 100) if self.total_collected > 0 else 0
            target_pct = (BALANCED_CONFIG['target_per_year'] / self.target_papers * 100)
            status = "✓" if count >= BALANCED_CONFIG['min_papers_per_year'] else "⚠️ "
            print(f"  {status} {year}: {count:5d} papers ({pct:5.1f}% of total, target: {target_pct:.1f}%)")

        # Find underfilled years
        underfilled = {}
        for year in BALANCED_CONFIG['years_to_cover']:
            count = year_distribution.get(year, 0)
            if count < BALANCED_CONFIG['min_papers_per_year']:
                underfilled[year] = BALANCED_CONFIG['min_papers_per_year'] - count

        if underfilled:
            print(f"\n⚠️  Underfilled years detected: {underfilled}")
            print("Executing supplementary queries for balance...")

            # Create supplementary broad queries for underfilled years
            supplementary = [
                ("cat:cs.LG", "Pure CS.LG"),
                ("cat:cs.AI", "Pure CS.AI"),
                ("cat:q-bio", "Pure Bioinformatics"),
                ("healthcare OR medical OR clinical", "Healthcare keywords"),
                ("machine learning OR deep learning", "ML keywords"),
            ]

            for supp_query, supp_desc in supplementary:
                if self.total_collected >= self.target_papers:
                    break

                for year in sorted(underfilled.keys()):
                    if self.total_collected >= self.target_papers:
                        break

                    if year_distribution.get(year, 0) >= BALANCED_CONFIG['min_papers_per_year']:
                        del underfilled[year]
                        continue

                    deficit = BALANCED_CONFIG['min_papers_per_year'] - year_distribution.get(year, 0)

                    # Create year-specific supplementary query
                    year_supp_query = f"{supp_query} AND submittedDate:[{year}0101000000 TO {year}1231235959]"

                    self.queries_executed += 1
                    print(f"\n  [{self.queries_executed}] {supp_desc} for year {year}")
                    print(f"      Query: {year_supp_query[:70]}...")
                    print(f"      Target: {deficit} papers (to fill gap)")

                    new_papers, processed = self._collect_from_query(
                        query=year_supp_query,
                        target_papers=deficit,
                        category=f"{year}",
                    )

                    dup_rate = (1 - new_papers/processed * 100) if processed > 0 else 0
                    print(f"      → {new_papers} new papers ({processed} processed, {dup_rate:.1f}% dups)")
                    print(f"      ✓ Total so far: {self.total_collected}/{self.target_papers}")

        # Phase 3: Final statistics and report
        print("\n" + "="*80)
        print("COLLECTION COMPLETE")
        print("="*80)

        final_distribution = self._get_year_distribution()

        print(f"\n📊 Final Statistics:")
        print(f"   Total papers collected: {self.total_collected}")
        print(f"   Total papers processed: {self.total_processed}")
        print(f"   Duplicate rate: {(self.duplicates_count / self.total_processed * 100) if self.total_processed > 0 else 0:.1f}%")
        print(f"   Queries executed: {self.queries_executed}")

        print(f"\n📅 Distribution by Year:")
        for year in BALANCED_CONFIG['years_to_cover']:
            count = final_distribution.get(year, 0)
            pct = (count / self.total_collected * 100) if self.total_collected > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"   {year}: {count:5d} ({pct:5.1f}%) {bar}")

        print(f"\n📂 Distribution by Category:")
        for cat in sorted(self.papers_by_category.keys()):
            count = len(self.papers_by_category[cat])
            pct = (count / self.total_collected * 100) if self.total_collected > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"   {cat:20s}: {count:5d} ({pct:5.1f}%) {bar}")

        print(f"\n✅ Output files:")
        print(f"   Papers: {self.output_file}")
        print(f"   Stats:  {self.stats_file}")
        print(f"   Checkpoint: {self.checkpoint_file}")

        # Final save
        self._save_checkpoint()

        return self.total_collected

    def _collect_from_query(
        self,
        query: str,
        target_papers: int,
        category: str,
    ) -> Tuple[int, int]:
        """
        Collect papers from a single query.

        Args:
            query: ArXiv query string
            target_papers: Target papers for this query
            category: Category name for tracking

        Returns:
            Tuple of (new_papers_collected, total_papers_processed)
        """
        new_papers = 0
        papers_processed = 0
        papers_list = []

        try:
            # Execute query with pagination handling
            search = arxiv.Search(
                query=query,
                max_results=target_papers,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            for paper in search.results():
                papers_processed += 1

                # Check for duplicate
                if paper.entry_id in self.collected_ids:
                    self.duplicates_count += 1
                    continue

                # Extract metadata
                try:
                    published_year = int(paper.published.year) if hasattr(paper.published, 'year') else int(paper.published[:4])
                except:
                    published_year = 0

                paper_dict = {
                    'id': paper.entry_id,
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'published': paper.published.isoformat() if hasattr(paper.published, 'isoformat') else str(paper.published),
                    'published_year': published_year,
                    'summary': paper.summary,
                    'categories': paper.categories,
                    'pdf_url': paper.pdf_url,
                    'arxiv_url': paper.entry_id,
                }

                papers_list.append(paper_dict)
                self.collected_ids.add(paper.entry_id)
                new_papers += 1

                # Stop if we've reached target
                if new_papers >= target_papers:
                    break

                # Rate limiting (respect ArXiv's 3-second requirement)
                time.sleep(0.1)  # Small inter-paper delay

        except Exception as e:
            print(f"   ⚠️  Error collecting from query: {str(e)[:100]}")

        # Write papers to output file
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for paper in papers_list:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')

        # Track by year and category
        self.total_collected += new_papers
        self.total_processed += papers_processed

        for paper in papers_list:
            year_key = paper['published_year']
            if year_key > 0:
                self.papers_by_year[year_key].append(paper)
            self.papers_by_category[category].append(paper)

        # Checkpoint periodically
        if self.total_collected % self.checkpoint_interval == 0:
            self._save_checkpoint()

        return new_papers, papers_processed

    def _get_year_distribution(self) -> Dict[int, int]:
        """Get count of papers per year from current collection."""
        distribution = {}
        for year, papers in self.papers_by_year.items():
            distribution[year] = len(papers)
        return distribution


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="ArXiv Healthcare Paper Collection and Processing Pipeline"
    )

    parser.add_argument(
        '--mode',
        choices=['collect', 'extract', 'curate', 'train-tokenizer'],
        default='collect',
        help='Operation mode'
    )

    parser.add_argument(
        '--output-dir',
        default='./data/arxiv',
        help='Output directory'
    )

    parser.add_argument(
        '--max-papers',
        type=int,
        default=40000,
        help='Maximum papers to collect'
    )

    parser.add_argument(
        '--input',
        help='Input file (for extract, curate, train-tokenizer modes)'
    )

    args = parser.parse_args()

    if args.mode == 'collect':
        collect_arxiv_papers(
            output_dir=args.output_dir,
            max_papers=args.max_papers
        )
    elif args.mode == 'extract':
        if not args.input:
            print("Error: --input required for extract mode")
            return
        extract_pdf_texts(
            input_jsonl=args.input,
            output_dir=os.path.join(args.output_dir, "texts")
        )
    elif args.mode == 'curate':
        if not args.input:
            print("Error: --input required for curate mode")
            return
        curate_with_nemo(
            text_dir=os.path.join(args.output_dir, "texts"),
            metadata_jsonl=args.input,
            output_jsonl=os.path.join(args.output_dir, "curated_dataset.jsonl")
        )
    elif args.mode == 'train-tokenizer':
        if not args.input:
            print("Error: --input required for train-tokenizer mode")
            return
        train_healthcare_tokenizer(
            input_jsonl=args.input,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()