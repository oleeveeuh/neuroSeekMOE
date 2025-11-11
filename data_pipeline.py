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
            DocumentDataset_AVAILABLE = True
        except ImportError:
            DocumentDataset = None
            DocumentDataset_AVAILABLE = False
        
        try:
            from nemo_curator.modifiers import DocumentModifier
            DocumentModifier_AVAILABLE = True
        except ImportError:
            DocumentModifier = None
            DocumentModifier_AVAILABLE = False
        
        try:
            from nemo_curator.filters import DocumentFilter
            DocumentFilter_AVAILABLE = True
        except ImportError:
            DocumentFilter = None
            DocumentFilter_AVAILABLE = False
        
        try:
            from nemo_curator.dedup import FuzzyDedup
            FuzzyDedup_AVAILABLE = True
        except ImportError:
            try:
                from nemo_curator.filters import FuzzyDedup
                FuzzyDedup_AVAILABLE = True
            except ImportError:
                FuzzyDedup = None
                FuzzyDedup_AVAILABLE = False
        
        import dask
        NEMO_CURATOR_AVAILABLE = True
        print("NeMo Curator imported successfully")
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
        DocumentDataset = None
        DocumentDataset_AVAILABLE = False
        DocumentModifier = None
        DocumentModifier_AVAILABLE = False
        DocumentFilter = None
        DocumentFilter_AVAILABLE = False
        FuzzyDedup = None
        FuzzyDedup_AVAILABLE = False
        ProcessingStage_AVAILABLE = False
        Stage_AVAILABLE = False
        download_arxiv_AVAILABLE = False
        get_client_AVAILABLE = False
        print("NeMo Curator only supports Linux systems (current: {})".format(platform.system()))
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
    DocumentDataset = None
    DocumentDataset_AVAILABLE = False
    DocumentModifier = None
    DocumentModifier_AVAILABLE = False
    DocumentFilter = None
    DocumentFilter_AVAILABLE = False
    FuzzyDedup = None
    FuzzyDedup_AVAILABLE = False
    print("nemo-curator package not available. Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'")
    print(f"   Error: {e}")


# Rate limiting: ArXiv recommends 3 seconds between requests
RATE_LIMIT_DELAY = 3.0  # 3 seconds between requests (ArXiv recommendation)
CHECKPOINT_INTERVAL = 5000  # Save checkpoint every 5000 papers
LOG_INTERVAL = 500  # Log progress every 500 papers

# Target date range (set to None to disable date filtering)
# NOTE: Date filtering significantly reduces collection efficiency
# If you need recent papers, consider filtering AFTER collection
MIN_YEAR = None  # None = no minimum (accept all years) - DISABLED for efficiency
MAX_YEAR = None  # None = no maximum (accept all years)
# Alternative: MIN_YEAR = 2015, MAX_YEAR = 2024 for date filtering (but reduces efficiency)

# ArXiv search queries (legacy - used by older collection functions)
# Note: The main collection now uses diverse ML + healthcare/neuroscience combinations
# Optimized tiered query structure for healthcare + ML paper collection
OPTIMIZED_QUERIES = {
    'tier_1_primary': [
        ("cat:cs.LG AND healthcare", 5000),
        ("cat:cs.LG AND (medical OR diagnosis OR clinical)", 5000),
        ("cat:cs.LG AND (machine learning OR deep learning) AND (neurodegeneration OR alzheimer OR parkinson)", 3000),
        ("cat:cs.LG AND (machine learning OR deep learning) AND (drug OR protein OR molecular)", 2000),
    ],
    'tier_2_neuro_ml': [
        ("cat:q-bio AND (machine learning OR deep learning OR neural network OR computational)", 4000),
        ("cat:q-bio AND (neuroscience OR brain OR fmri OR eeg OR neuroimaging) AND (learning OR analysis OR computational)", 2000),
        ("cat:q-bio AND (neural OR brain) AND (network OR mapping OR connectivity OR analysis)", 1000),
    ],
    'tier_3_medical_imaging': [
        ("cat:cs.LG AND (image OR imaging OR segmentation) AND (medical OR diagnosis OR pathology)", 3000),
        ("cat:cs.CV AND (medical OR clinical OR diagnosis OR treatment)", 2000),
    ],
    'tier_4_broad_coverage': [
        ("(cat:cs.LG OR cat:cs.AI OR cat:cs.CV) AND (machine learning OR deep learning OR artificial intelligence) AND (healthcare OR hospital OR patient OR clinical)", 5000),
        ("(cat:cs.LG OR cat:cs.AI) AND (model OR algorithm OR network) AND (disease OR diagnosis OR prognosis OR treatment)", 4000),
        ("cat:cs.LG AND (learning OR neural) AND (medical OR health OR clinical OR patient)", 3000),
        ("cat:cs AND (data OR prediction OR classification) AND (medical OR diagnosis OR disease OR health)", 3000),
    ],
}

# Legacy flat query list (for backward compatibility)
ARXIV_QUERIES = [
    "cat:cs.LG AND (healthcare OR medical OR clinical)",
    "cat:cs.AI AND (neuroscience OR brain OR neural)",
    "cat:q-bio.NC AND (machine learning OR deep learning)",
    "cat:cs.AI AND (neurodegeneration OR alzheimer OR parkinson)",
    "cat:cs.LG AND (mri OR ct OR medical imaging)",
]


# ============================================================================
# Query Feasibility Analysis Tool
# ============================================================================

def _parse_xml_total_results(root: ET.Element, debug: bool = False) -> Optional[int]:
    """
    Robustly parse totalResults from ArXiv XML response.
    Tries multiple namespace approaches to handle different XML formats.
    
    Args:
        root: XML root element
        debug: If True, print XML structure on failure
        
    Returns:
        Total results count, or None if not found
    """
    ns = {'os': 'http://a9.com/-/spec/opensearch/1.1/'}
    opensearch_ns = 'http://a9.com/-/spec/opensearch/1.1/'
    
    # Method 1: Try with namespace prefix
    total_elem = root.find('.//os:totalResults', ns)
    
    # Method 2: Try with full namespace in tag
    if total_elem is None:
        total_elem = root.find(f'.//{{{opensearch_ns}}}totalResults')
    
    # Method 3: Search all elements for totalResults tag
    if total_elem is None:
        for elem in root.iter():
            if 'totalResults' in elem.tag:
                total_elem = elem
                break
    
    # Method 4: Try without namespace (direct tag name)
    if total_elem is None:
        total_elem = root.find('.//totalResults')
    
    if total_elem is not None and total_elem.text:
        try:
            return int(total_elem.text)
        except ValueError:
            if debug:
                print(f"   DEBUG: totalResults found but invalid value: '{total_elem.text}'")
            return None
    
    # Debug: Print XML structure if parsing failed
    if debug:
        print("   DEBUG: Could not find totalResults. XML structure:")
        print(f"   Root tag: {root.tag}")
        print(f"   Root children: {[child.tag for child in list(root)[:5]]}")
        # Print first few elements
        for i, elem in enumerate(root.iter()):
            if i < 10:
                print(f"   Element {i}: {elem.tag} = {elem.text[:50] if elem.text else 'None'}")
    
    return None


def analyze_query_feasibility(query: str) -> Dict:
    """
    Test a single ArXiv query to determine:
    - Total results available
    - Results per year (2015-2024)
    - Estimated retrieval rate with year-split
    - Pagination risk indicators
    - Recommended strategy (direct/year/month)
    
    Returns: {
        'query': str,
        'total_results': int,
        'by_year': {2024: 412, 2023: 387, ...},
        'risky_years': [2020, 2019],  # Years with >1000 results
        'retrieval_rate': 0.87,  # Realistic % of papers we'll get
        'estimated_papers': 3571,
        'strategy': 'year_split',  # or 'direct', 'month_split'
        'time_estimate_seconds': 45,
        'error': None or {'type': str, 'message': str},
    }
    """
    if not REQUESTS_AVAILABLE:
        return {
            'query': query,
            'error': {'type': 'ImportError', 'message': 'requests package not available. Install with: pip install requests'},
            'total_results': 0,
            'by_year': {},
            'risky_years': [],
            'retrieval_rate': 0.0,
            'estimated_papers': 0,
            'strategy': None,
            'time_estimate_seconds': 0,
        }
    
    results = {
        'query': query,
        'total_results': 0,
        'by_year': {},
        'risky_years': [],
        'retrieval_rate': 0.0,
        'estimated_papers': 0,
        'strategy': None,
        'time_estimate_seconds': 0,
        'error': None,
    }
    
    base_url = "http://export.arxiv.org/api/query"
    
    # Retry logic with exponential backoff for getting total results
    max_retries = 3
    retry_delays = [5, 10, 20]  # Exponential backoff: 5s, 10s, 20s
    
    # Get total results with retry logic
    for retry_attempt in range(max_retries + 1):
        try:
            params = {
                'search_query': query,
                'start': 0,
                'max_results': 1,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            if retry_attempt > 0:
                print(f"   Retry attempt {retry_attempt}/{max_retries} for total results...")
            
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML to get total with robust namespace handling
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                error_msg = f"Invalid XML response: {e}"
                print(f"   DEBUG: {error_msg}")
                print(f"   DEBUG: Response content (first 500 chars): {response.content[:500]}")
                if retry_attempt < max_retries:
                    delay = retry_delays[retry_attempt]
                    print(f"   Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    results['error'] = {'type': 'ParseError', 'message': error_msg}
                    return results
            
            total_results = _parse_xml_total_results(root, debug=(retry_attempt == max_retries))
            
            if total_results is not None:
                results['total_results'] = total_results
                break  # Success - exit retry loop
            else:
                # Could not parse totalResults
                if retry_attempt < max_retries:
                    delay = retry_delays[retry_attempt]
                    print(f"   Could not parse totalResults, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    results['error'] = {'type': 'ParseError', 'message': 'Could not find totalResults in API response'}
                    return results
        
        except requests.exceptions.Timeout as e:
            error_msg = "Request timed out"
            print(f"   {error_msg}: {e}")
            if retry_attempt < max_retries:
                delay = retry_delays[retry_attempt]
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                results['error'] = {'type': 'Timeout', 'message': error_msg}
                return results
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 'unknown'
            error_msg = f"HTTP error: {status_code}"
            print(f"   {error_msg}: {e}")
            if retry_attempt < max_retries and status_code in [429, 500, 502, 503, 504]:
                # Retry on rate limit or server errors
                delay = retry_delays[retry_attempt]
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                results['error'] = {'type': 'HTTPError', 'message': f"{error_msg} ({status_code})"}
                return results
        
        except requests.exceptions.ConnectionError as e:
            error_msg = "Connection failed"
            print(f"   {error_msg}: {e}")
            if retry_attempt < max_retries:
                delay = retry_delays[retry_attempt]
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                results['error'] = {'type': 'ConnectionError', 'message': error_msg}
                return results
        
        except ValueError as e:
            error_msg = f"Invalid result count: {e}"
            print(f"   {error_msg}")
            if retry_attempt < max_retries:
                delay = retry_delays[retry_attempt]
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                results['error'] = {'type': 'ValueError', 'message': error_msg}
                return results
        
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"   {error_msg}")
            if retry_attempt < max_retries:
                delay = retry_delays[retry_attempt]
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                results['error'] = {'type': type(e).__name__, 'message': str(e)}
                return results
    
    # Test each year (2015-2024) with retry logic
    for year in range(2024, 2014, -1):
        date_filter = f" AND submittedDate:[{year}01010000 TO {year}12312359]"
        yearly_query = query + date_filter
        
        # Retry logic for yearly queries
        yearly_success = False
        for retry_attempt in range(max_retries + 1):
            try:
                if retry_attempt > 0:
                    print(f"   Year {year}: Retry attempt {retry_attempt}/{max_retries}...")
                
                # Count results for this year
                response = requests.get(base_url, params={
                    'search_query': yearly_query,
                    'start': 0,
                    'max_results': 1
                }, timeout=15)
                
                response.raise_for_status()
                
                # Parse XML with robust namespace handling
                try:
                    root = ET.fromstring(response.content)
                except ET.ParseError as e:
                    if retry_attempt < max_retries:
                        delay = retry_delays[retry_attempt]
                        print(f"   Year {year}: XML parse error, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"   Year {year}: XML parse error after {max_retries} retries, skipping")
                        results['by_year'][year] = 0
                        break
                
                yearly_total = _parse_xml_total_results(root, debug=False)
                
                if yearly_total is not None:
                    results['by_year'][year] = yearly_total
                    
                    # Flag risky years (>1000 means pagination limit possible)
                    if yearly_total > 1000:
                        results['risky_years'].append(year)
                    yearly_success = True
                    break  # Success - exit retry loop
                else:
                    # Could not parse
                    if retry_attempt < max_retries:
                        delay = retry_delays[retry_attempt]
                        print(f"   Year {year}: Could not parse results, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        results['by_year'][year] = 0
                        break
            
            except requests.exceptions.Timeout:
                if retry_attempt < max_retries:
                    delay = retry_delays[retry_attempt]
                    print(f"   Year {year}: Timeout, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"   Year {year}: Timeout after {max_retries} retries, skipping")
                    results['by_year'][year] = 0
                    break
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 'unknown'
                if retry_attempt < max_retries and status_code in [429, 500, 502, 503, 504]:
                    delay = retry_delays[retry_attempt]
                    print(f"   Year {year}: HTTP {status_code}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"   Year {year}: HTTP error {status_code}, skipping")
                    results['by_year'][year] = 0
                    break
            
            except requests.exceptions.ConnectionError:
                if retry_attempt < max_retries:
                    delay = retry_delays[retry_attempt]
                    print(f"   Year {year}: Connection error, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"   Year {year}: Connection error after {max_retries} retries, skipping")
                    results['by_year'][year] = 0
                    break
            
            except ValueError as e:
                if retry_attempt < max_retries:
                    delay = retry_delays[retry_attempt]
                    print(f"   Year {year}: Invalid result count, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"   Year {year}: Invalid result count after {max_retries} retries, skipping")
                    results['by_year'][year] = 0
                    break
            
            except Exception as e:
                if retry_attempt < max_retries:
                    delay = retry_delays[retry_attempt]
                    print(f"   Year {year}: Error {type(e).__name__}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"   Year {year}: Error after {max_retries} retries: {e}")
                    results['by_year'][year] = 0
                    break
        
        # Small delay to avoid rate limiting between years
        if yearly_success:
            time.sleep(0.5)
    
    # Calculate retrieval rate and strategy
    total_by_year = sum(results['by_year'].values())
    if total_by_year > 0:
        # With year-split: lose papers from risky years (estimate ~200 per risky year)
        papers_lost = len(results['risky_years']) * 200  # Rough estimate
        results['retrieval_rate'] = max(0.0, min(1.0, (total_by_year - papers_lost) / total_by_year))
        results['estimated_papers'] = int(total_by_year * results['retrieval_rate'])
    else:
        results['retrieval_rate'] = 0.0
        results['estimated_papers'] = 0
    
    # Determine strategy
    if results['total_results'] < 500:
        results['strategy'] = 'direct'
        results['time_estimate_seconds'] = 5
    elif len(results['risky_years']) == 0:
        results['strategy'] = 'year_split'
        results['time_estimate_seconds'] = 60
    else:
        results['strategy'] = 'year_split'  # Still use year_split, lose some papers
        results['time_estimate_seconds'] = 80
    
    return results


def analyze_all_queries(queries: List[str]) -> Dict:
    """Test all queries and generate report."""
    
    print("\n" + "="*70)
    print("QUERY FEASIBILITY ANALYSIS")
    print("="*70 + "\n")
    
    analyses = []
    total_available = 0
    total_retrievable = 0
    total_time = 0
    
    for i, query in enumerate(queries, 1):
        print(f"Testing query {i}/{len(queries)}: {query[:60]}...", end=" ", flush=True)
        analysis = analyze_query_feasibility(query)
        analyses.append(analysis)
        
        if analysis.get('error'):
            print(f"ERROR: {analysis['error']}")
        else:
            total_available += analysis['total_results']
            total_retrievable += analysis['estimated_papers']
            total_time += analysis['time_estimate_seconds']
            print(f"OK ({analysis['total_results']} -> {analysis['estimated_papers']} est.)")
    
    # Generate report
    report = {
        'queries_tested': len(queries),
        'total_available': total_available,
        'total_retrievable': total_retrievable,
        'retrieval_rate': total_retrievable / total_available if total_available > 0 else 0,
        'estimated_time_minutes': total_time / 60,
        'analyses': analyses,
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total papers available: {total_available:,}")
    print(f"Estimated retrievable: {total_retrievable:,} ({report['retrieval_rate']*100:.0f}%)")
    print(f"Estimated time: {report['estimated_time_minutes']:.0f} minutes")
    print(f"Risky queries: {len([a for a in analyses if a.get('risky_years')])}")
    
    # Detailed breakdown
    print("\n" + "-"*70)
    print("DETAILED BREAKDOWN BY QUERY")
    print("-"*70)
    for i, analysis in enumerate(analyses, 1):
        if analysis.get('error'):
            print(f"\nQuery {i}: {analysis['query'][:60]}...")
            print(f"  ERROR: {analysis['error']}")
        else:
            print(f"\nQuery {i}: {analysis['query'][:60]}...")
            print(f"  Total results: {analysis['total_results']:,}")
            print(f"  Estimated retrievable: {analysis['estimated_papers']:,} ({analysis['retrieval_rate']*100:.0f}%)")
            print(f"  Strategy: {analysis['strategy']}")
            print(f"  Risky years: {analysis['risky_years'] if analysis['risky_years'] else 'None'}")
            if analysis['risky_years']:
                print(f"  Year breakdown (risky years marked with *):")
                for year in range(2024, 2014, -1):
                    count = analysis['by_year'].get(year, 0)
                    marker = "*" if year in analysis['risky_years'] else " "
                    print(f"    {marker} {year}: {count:,} papers")
    
    return report


def analyze_optimized_queries() -> Dict:
    """Analyze all queries from OPTIMIZED_QUERIES."""
    # Flatten the tiered structure
    queries = []
    for tier_name, tier_queries in OPTIMIZED_QUERIES.items():
        for query, max_papers in tier_queries:
            queries.append(query)
    
    return analyze_all_queries(queries)


def refine_query_for_efficiency(query: str, current_results: int) -> List[str]:
    """
    If query returns too many results (>3000), refine it to be more specific:
    - Replace broad ORs with narrower AND combinations
    - Add date filters if needed
    - Suggest splitting into multiple narrow queries
    
    Examples:
    "cat:cs.LG AND (healthcare OR medical OR clinical)"
    → Too broad (14k results)
    → Refine to: "cat:cs.LG AND healthcare" (4k)
    →           "cat:cs.LG AND medical diagnosis"
    →           "cat:cs.LG AND clinical treatment"
    
    Returns: List of refined queries
    """
    if current_results < 1000:
        return [query]  # Already good
    
    refined_queries = []
    
    # Strategy 1: Split OR clauses into separate queries
    if " OR " in query.upper():
        # Find OR clauses (case-insensitive)
        import re
        # Match patterns like: (term1 OR term2 OR term3)
        or_pattern = r'\(([^)]+)\)'
        matches = re.findall(or_pattern, query)
        
        for match in matches:
            if " OR " in match.upper():
                # Split OR terms
                terms = re.split(r'\s+OR\s+', match, flags=re.IGNORECASE)
                # Get the base query (everything before the OR clause)
                base = query[:query.find(match) - 1].strip()  # Remove opening paren
                base = base.rstrip(" AND").strip()
                
                # Create separate queries for each term
                for term in terms:
                    term = term.strip()
                    if term:
                        refined_query = f"{base} AND {term}"
                        refined_queries.append(refined_query)
        
        # If we didn't find OR in parentheses, try splitting the whole query
        if not refined_queries:
            parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Try to preserve category prefix
                if query.startswith("cat:"):
                    category = query.split()[0]
                    for part in parts[1:]:  # Skip category part
                        refined_queries.append(f"{category} AND {part.strip()}")
                else:
                    refined_queries = [p.strip() for p in parts if p.strip()]
    
    # Strategy 2: Add more specific terms for healthcare queries
    elif "healthcare" in query.lower() and current_results > 3000:
        # Try adding more specific terms
        base = query.replace("healthcare", "").strip().rstrip("AND").strip()
        refined_queries.append(f"{base} AND healthcare AND diagnosis")
        refined_queries.append(f"{base} AND healthcare AND treatment")
        refined_queries.append(f"{base} AND healthcare AND clinical")
    
    # Strategy 3: Add date filters for very large queries
    elif current_results > 10000:
        # Split by recent years (2020-2024) and older (2015-2019)
        base = query
        refined_queries.append(f"{base} AND submittedDate:[202001010000 TO 202412312359]")
        refined_queries.append(f"{base} AND submittedDate:[201501010000 TO 201912312359]")
    
    # If no refinement strategy worked, return original
    if not refined_queries:
        refined_queries = [query]
    
    return refined_queries


def build_optimized_queries(base_queries: List[str]) -> List[Tuple[str, str]]:
    """
    Convert base queries into optimized query list with strategies.
    
    Returns: List of (query, strategy) tuples
    
    Example output:
    [
        ("cat:cs.LG AND healthcare", "year_split"),
        ("cat:cs.LG AND medical", "year_split"),
        ("cat:cs.LG AND clinical", "year_split"),
        ("cat:cs.AI AND healthcare", "year_split"),
        ...
    ]
    """
    optimized = []
    
    print(f"\nBuilding optimized query list from {len(base_queries)} base queries...")
    
    for i, query in enumerate(base_queries, 1):
        print(f"Analyzing query {i}/{len(base_queries)}: {query[:60]}...", end=" ", flush=True)
        
        # Test query
        analysis = analyze_query_feasibility(query)
        
        if analysis.get('error'):
            print(f"ERROR: {analysis['error']}")
            # Skip this query
            continue
        
        total_results = analysis['total_results']
        print(f"{total_results:,} results")
        
        if total_results < 500:
            # Small result set - query directly
            optimized.append((query, "direct"))
        
        elif total_results < 2000:
            # Medium result set - year split is fine
            optimized.append((query, "year_split"))
        
        elif total_results > 5000:
            # Large result set - split query or use month_split
            if " OR " in query.upper():
                # Split broad query into narrower ones
                print(f"  Refining query (too many results: {total_results:,})...")
                refined = refine_query_for_efficiency(query, total_results)
                for q in refined:
                    # Test refined query
                    refined_analysis = analyze_query_feasibility(q)
                    if not refined_analysis.get('error'):
                        refined_results = refined_analysis['total_results']
                        if refined_results < 2000:
                            optimized.append((q, "year_split"))
                        else:
                            optimized.append((q, "year_split_truncated"))
                print(f"  Split into {len(refined)} refined queries")
            else:
                # Use as-is but expect to lose ~20% due to pagination
                optimized.append((query, "year_split_truncated"))
        
        else:
            optimized.append((query, "year_split"))
    
    print(f"\nOptimized query list: {len(optimized)} queries")
    print(f"  Strategies: {len([q for q, s in optimized if s == 'direct'])} direct, "
          f"{len([q for q, s in optimized if s == 'year_split'])} year_split, "
          f"{len([q for q, s in optimized if s == 'year_split_truncated'])} year_split_truncated")
    
    return optimized


def generate_year_split_queries(
    base_query: str,
    start_year: int = 2015,
    end_year: int = 2024
) -> List[str]:
    """
    Split query by year to avoid pagination limits.
    
    Returns list of year-specific queries:
    ["cat:cs.LG AND healthcare AND submittedDate:[201501010000 TO 201512312359]",
     "cat:cs.LG AND healthcare AND submittedDate:[201601010000 TO 201612312359]",
     ...]
    
    Args:
        base_query: Base ArXiv query (e.g., "cat:cs.LG AND healthcare")
        start_year: Starting year (default: 2015)
        end_year: Ending year (default: 2024)
    
    Returns:
        List of year-specific queries
    """
    queries = []
    for year in range(end_year, start_year - 1, -1):  # 2024 down to 2015
        date_filter = f" AND submittedDate:[{year}01010000 TO {year}12312359]"
        queries.append(base_query + date_filter)
    
    return queries


def collect_with_pagination_safety(
    client,
    query: str,
    max_papers_to_retrieve: int = 1000,  # Stop at 1000 to avoid pagination limits
    existing_ids: Optional[Set[str]] = None,
    output_file_handle = None
) -> Tuple[List[Dict], int, bool]:
    """
    Collect papers with built-in pagination safety:
    - Stop after max_papers_to_retrieve papers (default 1000, ArXiv pagination limit)
    - Detect when ArXiv truncates results silently
    - Track which pagination level we hit
    - Return papers collected + count + pagination status
    
    Args:
        client: arxiv.Client instance
        query: ArXiv query string
        max_papers_to_retrieve: Maximum papers to retrieve (default: 1000)
        existing_ids: Set of already collected paper IDs (for deduplication)
        output_file_handle: Optional file handle to write papers immediately
    
    Returns:
        Tuple of (list of papers dict, count of papers collected, hit_pagination_limit: bool)
    """
    if existing_ids is None:
        existing_ids = set()
    
    papers = []
    papers_collected = 0
    hit_pagination_limit = False
    papers_in_last_100 = 0  # Track papers in current 100-paper window
    duplicates_skipped = 0  # Track duplicates skipped during collection
    
    search = arxiv.Search(
        query=query,
        max_results=10000,  # Request all, but limit retrieval
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # Retry logic with exponential backoff for rate limiting
    max_retries = 3
    retry_delays = [5, 10, 20]  # Exponential backoff: 5s, 10s, 20s
    
    for retry_attempt in range(max_retries + 1):
        try:
            results_iter = iter(client.results(search))
            papers_in_last_100 = 0
            duplicates_in_last_100 = 0
            
            for i, result in enumerate(results_iter):
                if papers_collected >= max_papers_to_retrieve:
                    # Stop at limit - ArXiv pagination depth limit
                    print(f"   ⚠️  Reached requested limit ({max_papers_to_retrieve} papers) for query")
                    break
                
                # Extract paper ID
                paper_id = result.entry_id.split('/')[-1]
                
                # Skip if already collected
                if paper_id in existing_ids:
                    duplicates_skipped += 1
                    duplicates_in_last_100 += 1
                    # Log periodically if high duplicate rate
                    if (i + 1) % 100 == 0:
                        if duplicates_in_last_100 >= 80:
                            print(f"   ⚠️  High duplicate rate in last 100 results: {duplicates_in_last_100} duplicates, {papers_in_last_100} new")
                        duplicates_in_last_100 = 0
                        papers_in_last_100 = 0
                    continue
                
                # Extract year
                year = None
                if result.published:
                    try:
                        year = result.published.year
                    except:
                        pass
                
                # Extract categories
                categories = []
                if result.categories:
                    for cat in result.categories:
                        if isinstance(cat, str):
                            categories.append(cat)
                        elif hasattr(cat, 'term'):
                            categories.append(cat.term)
                        else:
                            categories.append(str(cat))
                
                # Format abstract
                abstract = ""
                if result.summary:
                    abstract = result.summary.strip()[:300]
                
                paper = {
                    'id': paper_id,
                    'title': result.title.strip() if result.title else "",
                    'abstract': abstract,
                    'year': year,
                    'categories': categories,
                    'pdf_url': f"https://arxiv.org/pdf/{paper_id}.pdf",
                }
                
                papers.append(paper)
                existing_ids.add(paper_id)
                papers_collected += 1
                papers_in_last_100 += 1
                
                # Write immediately if file handle provided
                if output_file_handle:
                    output_file_handle.write(json.dumps(paper, ensure_ascii=False) + '\n')
                    output_file_handle.flush()
                
                # Detect pagination truncation: if we get very few papers when expecting more
                # Check every 100 papers
                if papers_collected % 100 == 0:
                    if papers_in_last_100 < 10 and papers_collected < max_papers_to_retrieve:
                        print(f"   ⚠️  Warning: Only {papers_in_last_100} papers in last 100 iterations (possible pagination truncation)")
                        hit_pagination_limit = True
                    papers_in_last_100 = 0  # Reset counter
            
            # Check if we stopped early due to low results
            # If we collected fewer papers than expected and the last batch was small, likely truncation
            if papers_collected < max_papers_to_retrieve:
                if papers_collected > 0 and papers_in_last_100 < 10:
                    # Small final batch suggests truncation
                    print(f"   ⚠️  Warning: Collection stopped early with only {papers_collected} papers (expected up to {max_papers_to_retrieve})")
                    print(f"   ⚠️  Final batch had only {papers_in_last_100} papers - possible ArXiv pagination truncation")
                    hit_pagination_limit = True
                elif papers_collected == 0:
                    # No papers at all - might be query issue or pagination
                    print(f"   ⚠️  Warning: No papers collected (possible query issue or pagination limit)")
                    hit_pagination_limit = True
            
            # Success - break out of retry loop
            break
            
        except Exception as e:
            # Check for UnexpectedEmptyPageError (ArXiv pagination limit)
            if UnexpectedEmptyPageError and isinstance(e, UnexpectedEmptyPageError):
                print(f"   ⚠️  Hit ArXiv pagination limit (UnexpectedEmptyPageError) at {papers_collected} papers")
                print(f"   ⚠️  ArXiv silently truncates results beyond ~1000 papers per query")
                hit_pagination_limit = True
                break  # Don't retry on pagination limit
            
            # Check for arxiv.HTTPError (rate limiting, etc.)
            error_type_name = type(e).__name__
            if 'HTTPError' in error_type_name or hasattr(e, 'status_code'):
                status_code = getattr(e, 'status_code', None) or (str(e).split()[-1] if '429' in str(e) else None)
                if status_code == 429 or '429' in str(e):
                    print(f"   ⚠️  Rate limited (HTTP 429), waiting 30 seconds...")
                    if retry_attempt < max_retries:
                        delay = 30  # Longer delay for rate limits
                        print(f"   Retrying in {delay}s (attempt {retry_attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"   ⚠️  Rate limit error after {max_retries} retries, stopping collection for this query")
                        break
                else:
                    # Other HTTP errors
                    print(f"   ⚠️  HTTP error: {e}")
                    if retry_attempt < max_retries:
                        delay = retry_delays[retry_attempt]
                        print(f"   Retrying in {delay}s (attempt {retry_attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"   ⚠️  HTTP error after {max_retries} retries, stopping collection")
                        break
            
            # Check for other rate limit indicators
            error_str = str(e).lower()
            is_rate_limit = (
                "429" in str(e) or 
                "rate limit" in error_str or 
                "too many requests" in error_str
            )
            
            if is_rate_limit and retry_attempt < max_retries:
                delay = 30  # Longer delay for rate limits
                print(f"   ⚠️  Rate limit error, retrying in {delay}s (attempt {retry_attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            else:
                # Not a rate limit error, or max retries reached
                if is_rate_limit:
                    print(f"   ⚠️  Rate limit error after {max_retries} retries, stopping collection for this query")
                else:
                    print(f"   ⚠️  Stopped at {papers_collected} papers: {e}")
                break
    
    # Final pagination status check
    if hit_pagination_limit:
        print(f"   ⚠️  Pagination limit detected: Collected {papers_collected} papers (may be truncated)")
    
    # Log deduplication summary
    total_processed = papers_collected + duplicates_skipped
    if duplicates_skipped > 0:
        duplicate_pct = (duplicates_skipped / total_processed * 100) if total_processed > 0 else 0
        print(f"   📊 Deduplication: {papers_collected} new papers, {duplicates_skipped} duplicates skipped ({duplicate_pct:.1f}% duplicate rate)")
        if duplicate_pct >= 80:
            print(f"   ⚠️  Very high duplicate rate - query may be redundant")
    
    return papers, papers_collected, hit_pagination_limit


def execute_optimized_queries(
    optimized_queries: List[Tuple[str, str]],
    output_jsonl: str,
    checkpoint_jsonl: str = None,
    max_papers_per_query: int = 1000,
    rate_limit_delay: float = 2.0
) -> int:
    """
    Execute all optimized queries in order.
    
    Args:
        optimized_queries: List of (query, strategy) tuples from build_optimized_queries()
        output_jsonl: Where to write papers (JSONL file)
        checkpoint_jsonl: Optional checkpoint file path (for resuming)
        max_papers_per_query: Maximum papers to retrieve per sub-query (default: 1000)
        rate_limit_delay: Delay between requests in seconds (default: 2.0)
    
    Returns: Total papers collected
    """
    if not ARXIV_AVAILABLE:
        error_msg = "Error: arxiv package not available. Install with: pip install arxiv"
        print(error_msg)
        raise ImportError(error_msg)
    
    # Load checkpoint if exists
    collected_ids = set()
    total_collected = 0
    
    if checkpoint_jsonl and os.path.exists(checkpoint_jsonl):
        print(f"Loading checkpoint from {checkpoint_jsonl}...")
        try:
            with open(checkpoint_jsonl, 'r') as f:
                checkpoint = json.load(f)
                total_collected = checkpoint.get('total', 0)
                checkpoint_ids = checkpoint.get('collected_ids', [])
                collected_ids = set(checkpoint_ids)
            print(f"   Found {len(collected_ids)} existing papers in checkpoint")
        except Exception as e:
            print(f"   Could not load checkpoint: {e}")
            print("   Starting fresh...")
    
    # Also load from output file if it exists (for deduplication)
    if os.path.exists(output_jsonl):
        print(f"Loading existing papers from {output_jsonl}...")
        try:
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            paper = json.loads(line)
                            if 'id' in paper:
                                collected_ids.add(paper['id'])
                        except json.JSONDecodeError:
                            continue
            print(f"   Found {len(collected_ids)} existing papers in output file")
        except Exception as e:
            print(f"   Could not read output file: {e}")
    
    # Create ArXiv client with ArXiv-recommended 3 second delay
    # Use 3.0 seconds regardless of rate_limit_delay parameter to comply with ArXiv API
    client = arxiv.Client(
        delay_seconds=3.0,  # ArXiv recommends 3 seconds between requests
        num_retries=3,
        page_size=100
    )
    
    print("\n" + "="*70)
    print("EXECUTING OPTIMIZED QUERIES")
    print("="*70)
    print(f"Total queries: {len(optimized_queries)}")
    print(f"Starting from: {len(collected_ids)} existing papers")
    print(f"Max papers per sub-query: {max_papers_per_query}")
    print()
    
    # Open output file in append mode
    with open(output_jsonl, 'a', encoding='utf-8') as out_f:
        for query_idx, (query, strategy) in enumerate(optimized_queries, 1):
            if total_collected >= 100000:  # Safety limit
                print(f"\nReached safety limit of 100,000 papers")
                break
            
            print(f"\n[{query_idx}/{len(optimized_queries)}] {query[:65]}...")
            print(f"Strategy: {strategy}")
            
            # Generate year-split sub-queries if needed
            if strategy.startswith("year_split"):
                sub_queries = generate_year_split_queries(query)
                print(f"   Split into {len(sub_queries)} year-specific queries")
            else:
                sub_queries = [query]
                print(f"   Using direct query (no year split)")
            
            papers_from_query = 0
            total_papers_processed = 0  # Track total papers from all sub-queries (before deduplication)
            
            for year_idx, sub_query in enumerate(sub_queries, 1):
                # Extract year for display
                year_str = "all"
                if "submittedDate:[" in sub_query:
                    try:
                        year_match = re.search(r'(\d{4})01010000', sub_query)
                        if year_match:
                            year_str = year_match.group(1)
                    except:
                        pass
                
                print(f"   [{year_idx}/{len(sub_queries)}] Year {year_str}: ", end="", flush=True)
                
                # Retry logic with exponential backoff for rate limiting
                max_retries = 3
                retry_delays = [5, 10, 20]  # Exponential backoff: 5s, 10s, 20s
                papers = []
                count = 0
                
                for retry_attempt in range(max_retries + 1):
                    try:
                        # Collect papers with pagination safety
                        papers, count, hit_limit = collect_with_pagination_safety(
                            client=client,
                            query=sub_query,
                            max_papers_to_retrieve=max_papers_per_query,
                            existing_ids=collected_ids,
                            output_file_handle=None  # We'll write manually to filter duplicates
                        )
                        
                        if hit_limit:
                            print(f"   ⚠️  Query hit pagination limit: {sub_query[:60]}...")
                        
                        # Success - break out of retry loop
                        break
                        
                    except KeyboardInterrupt:
                        print("\n\nInterrupted by user")
                        raise
                    except Exception as e:
                        error_str = str(e).lower()
                        is_rate_limit = (
                            "429" in str(e) or 
                            "rate limit" in error_str or 
                            "too many requests" in error_str or
                            "http error" in error_str and "429" in str(e)
                        )
                        
                        if is_rate_limit and retry_attempt < max_retries:
                            delay = retry_delays[retry_attempt]
                            print(f"   Rate limit error (HTTP 429), retrying in {delay}s (attempt {retry_attempt + 1}/{max_retries})...")
                            time.sleep(delay)
                            continue
                        else:
                            # Not a rate limit error, or max retries reached
                            if is_rate_limit:
                                print(f"   Rate limit error after {max_retries} retries, skipping this query")
                            else:
                                print(f"⚠️  Error: {str(e)[:60]}")
                                import traceback
                                print(f"   Traceback: {traceback.format_exc()[:200]}")
                            papers = []
                            count = 0
                            break
                
                # Track total papers processed (before deduplication)
                total_papers_processed += count
                
                # Filter out already-collected papers
                new_papers = []
                duplicates_in_batch = 0
                for paper in papers:
                    if paper['id'] not in collected_ids:
                        new_papers.append(paper)
                        collected_ids.add(paper['id'])
                    else:
                        duplicates_in_batch += 1
                
                # Write new papers to file
                for paper in new_papers:
                    out_f.write(json.dumps(paper, ensure_ascii=False) + '\n')
                    out_f.flush()
                    total_collected += 1
                
                # Log deduplication statistics
                if duplicates_in_batch > 0:
                    duplicate_pct = (duplicates_in_batch / count * 100) if count > 0 else 0
                    if duplicate_pct >= 90:
                        print(f"⚠️  {len(new_papers)} new papers, {duplicates_in_batch} duplicates ({duplicate_pct:.1f}% duplicate rate)")
                        if duplicate_pct >= 95:
                            print(f"   ⚠️  High duplication rate - query may be overlapping with previous queries")
                    elif duplicate_pct >= 50:
                        print(f"✓ {len(new_papers)} new papers, {duplicates_in_batch} duplicates ({duplicate_pct:.1f}% duplicate rate)")
                    else:
                        print(f"✓ {len(new_papers)} new papers, {duplicates_in_batch} duplicates")
                else:
                    print(f"✓ {len(new_papers)} papers")
                
                if len(new_papers) == 0 and count > 0:
                    print(f"   ⚠️  All {count} papers were duplicates - query returned no new papers")
                    print(f"   💡 This query may be redundant or overlapping with previous queries")
                
                papers_from_query += len(new_papers)
                
                # Rate limiting between sub-queries
                if year_idx < len(sub_queries):
                    time.sleep(1)
            
            # Calculate deduplication statistics for this query
            total_duplicates = total_papers_processed - papers_from_query
            if total_duplicates > 0 and total_papers_processed > 0:
                duplicate_pct = (total_duplicates / total_papers_processed * 100)
                print(f"   Query subtotal: {papers_from_query} new papers (skipped {total_duplicates} duplicates, {duplicate_pct:.1f}% duplicate rate)")
                if duplicate_pct >= 80:
                    print(f"   ⚠️  Very high duplicate rate for this query - may be redundant")
            elif len(sub_queries) > 1 and papers_from_query < len(sub_queries) * 10:
                # Low yield across multiple sub-queries suggests high overlap
                print(f"   Query subtotal: {papers_from_query} new papers from {len(sub_queries)} sub-queries")
                print(f"   💡 Low yield suggests high overlap between sub-queries")
            else:
                print(f"   Query subtotal: {papers_from_query} papers")
            print(f"   Total collected so far: {total_collected} papers")
            
            # Save checkpoint after each query
            if checkpoint_jsonl:
                try:
                    checkpoint_data = {
                        'total': total_collected,
                        'collected_ids': list(collected_ids),
                        'last_query_idx': query_idx,
                        'last_query': query,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(checkpoint_jsonl, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                except Exception as e:
                    print(f"   Warning: Could not save checkpoint: {e}")
    
    # Calculate final deduplication statistics
    print("\n" + "="*70)
    print(f"✅ Total collected: {total_collected} papers")
    print(f"✅ Unique papers: {len(collected_ids)}")
    
    # Log deduplication summary if we have checkpoint data
    if checkpoint_jsonl and os.path.exists(checkpoint_jsonl):
        try:
            with open(checkpoint_jsonl, 'r') as f:
                checkpoint_data = json.load(f)
                initial_count = checkpoint_data.get('total', 0)
                if initial_count > 0:
                    new_papers = total_collected - initial_count
                    print(f"📊 Collection session: {new_papers} new papers added (started with {initial_count})")
        except:
            pass
    
    print(f"✅ Output file: {output_jsonl}")
    if checkpoint_jsonl:
        print(f"✅ Checkpoint: {checkpoint_jsonl}")
    print("="*70)
    
    return total_collected


def optimize_and_execute_queries(
    base_queries: List[str],
    output_dir: str = "./data/arxiv",
    run_diagnostic: bool = True,
    run_collection: bool = True,
    max_papers_per_query: int = 1000,
    rate_limit_delay: float = 2.0
) -> List[Tuple[str, str]]:
    """
    Complete query optimization pipeline:
    
    1. Analyze all queries (diagnostic)
    2. Refine/optimize queries (build optimized list)
    3. Generate year-split variants (handled automatically)
    4. Execute with pagination safety
    
    Args:
        base_queries: List of base query strings to optimize
        output_dir: Output directory for papers and reports
        run_diagnostic: If True, run query analysis diagnostics
        run_collection: If True, execute optimized queries and collect papers
        max_papers_per_query: Maximum papers per sub-query (default: 1000)
        rate_limit_delay: Delay between requests in seconds (default: 2.0)
    
    Returns:
        List of (query, strategy) tuples from optimization
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_jsonl = os.path.join(output_dir, "arxiv_papers.jsonl")
    checkpoint_file = os.path.join(output_dir, "collection_checkpoint.json")
    optimized_file = os.path.join(output_dir, "optimized_queries.json")
    diagnostic_file = os.path.join(output_dir, "diagnostic_report.json")
    
    print("\n" + "="*70)
    print("QUERY OPTIMIZATION AND EXECUTION PIPELINE")
    print("="*70)
    print(f"Base queries: {len(base_queries)}")
    print(f"Output directory: {output_dir}")
    print(f"Run diagnostic: {run_diagnostic}")
    print(f"Run collection: {run_collection}")
    print()
    
    # Step 1: Analyze queries (diagnostic)
    if run_diagnostic:
        print("\n" + "="*70)
        print("STEP 1: ANALYZING QUERIES")
        print("="*70)
        print("Running diagnostic analysis on all queries...")
        
        report = analyze_all_queries(base_queries)
        
        # Save diagnostic report
        try:
            with open(diagnostic_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n✅ Diagnostic complete. Report saved to: {diagnostic_file}")
        except Exception as e:
            print(f"⚠️  Could not save diagnostic report: {e}")
        
        # Print summary
        if report.get('total_available'):
            print(f"\n📊 Summary:")
            print(f"  Total available papers: {report['total_available']:,}")
            print(f"  Estimated retrievable: {report['total_retrievable']:,}")
            print(f"  Estimated time: {report['estimated_time_hours']:.1f} hours")
            if report.get('risky_queries'):
                print(f"  Risky queries (pagination issues): {len(report['risky_queries'])}")
    else:
        print("\n⏭️  Skipping diagnostic analysis")
    
    # Step 2: Optimize queries
    print("\n" + "="*70)
    print("STEP 2: OPTIMIZING QUERIES")
    print("="*70)
    print("Building optimized query list with strategies...")
    
    optimized = build_optimized_queries(base_queries)
    
    # Save optimized queries
    try:
        output_data = {
            'optimized_queries': [{'query': q, 'strategy': s} for q, s in optimized],
            'total_queries': len(optimized),
            'strategies': {
                'direct': len([q for q, s in optimized if s == 'direct']),
                'year_split': len([q for q, s in optimized if s == 'year_split']),
                'year_split_truncated': len([q for q, s in optimized if s == 'year_split_truncated']),
            },
            'base_queries': base_queries,
            'timestamp': datetime.now().isoformat()
        }
        with open(optimized_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ Optimized queries saved to: {optimized_file}")
    except Exception as e:
        print(f"⚠️  Could not save optimized queries: {e}")
    
    # Print optimization summary
    print(f"\n📋 Query Optimization Summary:")
    print(f"  Original queries: {len(base_queries)}")
    print(f"  Optimized queries: {len(optimized)}")
    strategies = set([s for _, s in optimized])
    print(f"  Strategies used: {', '.join(sorted(strategies))}")
    
    strategy_counts = {}
    for _, s in optimized:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    for strategy, count in sorted(strategy_counts.items()):
        print(f"    - {strategy}: {count} queries")
    
    # Step 3: Execute optimized queries
    if run_collection:
        print("\n" + "="*70)
        print("STEP 3: EXECUTING OPTIMIZED QUERIES")
        print("="*70)
        print("Starting collection with pagination-safe execution...")
        
        total = execute_optimized_queries(
            optimized_queries=optimized,
            output_jsonl=output_jsonl,
            checkpoint_jsonl=checkpoint_file,
            max_papers_per_query=max_papers_per_query,
            rate_limit_delay=rate_limit_delay
        )
        
        print(f"\n✅ Collection complete: {total:,} papers")
        print(f"   Output file: {output_jsonl}")
        print(f"   Checkpoint: {checkpoint_file}")
    else:
        print("\n⏭️  Skipping collection (run_collection=False)")
        print(f"   To collect papers, run:")
        print(f"   python data_pipeline.py collect --use-optimized --output-dir {output_dir}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    return optimized


# Output fields (minimal metadata)
OUTPUT_FIELDS = ['id', 'title', 'abstract', 'year', 'categories', 'pdf_url']

# ============================================================================
# Google Drive Support (for Colab)
# ============================================================================

def get_drive_output_dir(local_output_dir: str = "./data/arxiv", drive_base: str = "/content/drive/MyDrive/neuroMOE_results") -> str:
    """
    Get output directory, preferring Google Drive if available (Colab).
    
    This function ensures that ALL pipeline outputs (metadata, text files, curated datasets, etc.)
    are saved directly to Google Drive for persistence.
    
    Args:
        local_output_dir: Local output directory (fallback)
        drive_base: Base path for Google Drive (default Colab path)
        
    Returns:
        Path to output directory (Drive if available, local otherwise)
    """
    # Check if we're in Colab and Drive is mounted
    try:
        import os
        drive_path = Path(drive_base)
        
        # Check if Drive is mounted (exists and is accessible)
        if drive_path.exists() and os.access(drive_path, os.W_OK):
            # Use Drive path - this will be the base for all outputs
            # Structure: /content/drive/MyDrive/neuroMOE_results/data/arxiv/
            #   - arxiv_papers.jsonl (metadata)
            #   - texts/ (extracted .txt files)
            #   - curated_dataset.jsonl
            #   - processed_dataset.jsonl
            #   - etc.
            drive_output = drive_path / "data" / "arxiv"
            drive_output.mkdir(parents=True, exist_ok=True)
            
            # Also create texts subdirectory on Drive
            texts_dir = drive_output / "texts"
            texts_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Using Google Drive for output: {drive_output}")
            print(f"  Text files directory: {texts_dir}")
            print(f"  All data will persist even if runtime is interrupted")
            return str(drive_output)
    except Exception as e:
        # If Drive not available, fall back to local
        pass
    
    # Fall back to local directory
    os.makedirs(local_output_dir, exist_ok=True)
    # Also create texts subdirectory locally
    texts_dir = os.path.join(local_output_dir, "texts")
    os.makedirs(texts_dir, exist_ok=True)
    return local_output_dir


def is_colab_environment() -> bool:
    """Check if running in Google Colab."""
    try:
        import os
        return 'COLAB_GPU' in os.environ or 'COLAB_JUPYTER_TOKEN' in os.environ
    except:
        return False


def is_drive_mounted(drive_path: str = "/content/drive/MyDrive") -> bool:
    """Check if Google Drive is mounted."""
    try:
        import os
        return os.path.exists(drive_path) and os.access(drive_path, os.W_OK)
    except:
        return False

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
            'retry_max': 5,
            'batch_size': 25,  # Optimized for time efficiency (24% faster than 10, minimal RAM impact)
            'ram_target': 50.0  # Target RAM percentage to stay below
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
            'min_relevance_score': 0.3
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
        print("YAML not available, using defaults")
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
            
            print(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return defaults
    else:
        # Create default config file
        try:
            with open(config_path, 'w') as f:
                yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)
            print(f"📝 Created default config file: {config_path}")
        except Exception as e:
            print(f"Could not create config file: {e}")
        return defaults


def save_config(config: Dict, config_path: str = "config.yaml"):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to config.yaml file
    """
    if not YAML_AVAILABLE:
        print("YAML not available, cannot save config")
        return
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Saved config to {config_path}")
    except Exception as e:
        print(f"Error saving config: {e}")


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
            print(f"Memory usage: {mem_stats['percent']:.1f}% (threshold: {warning_threshold}%)")
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
        print(f"   Memory: {mem_stats['percent']:.1f}% used, {mem_stats['available'] / (1024**3):.2f} GB available")


# ============================================================================
# Validation & Diagnostics
# ============================================================================

def check_arxiv_connection() -> bool:
    """Check if ArXiv API is accessible.
    
    Returns:
        True if accessible, False otherwise
    """
    if not ARXIV_AVAILABLE:
        print("ArXiv package not available")
        return False
    
    try:
        client = arxiv.Client(page_size=1, delay_seconds=0.5, num_retries=1)
        search = arxiv.Search(query="cat:cs.LG", max_results=1)
        result = next(client.results(search), None)
        if result:
            print("ArXiv API is accessible")
            return True
        else:
            print("ArXiv API returned no results (might be temporary)")
            return False
    except Exception as e:
        print(f"ArXiv API connection failed: {e}")
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
            print(f"Disk space: {available_gb:.2f} GB available (required: {required_gb:.2f} GB)")
            if available_gb < required_gb:
                print(f"Warning: Insufficient disk space")
                return False
            return True
        else:
            print("psutil not available, cannot check disk space")
            return True  # Assume OK
    except Exception as e:
        print(f"Could not check disk space: {e}")
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
        print(f"Output directory is writable: {path}")
        return True
    except Exception as e:
        print(f"Output directory not writable: {path} - {e}")
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
    print("Pipeline Diagnostics")
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
    print(f"   ArXiv: {'' if ARXIV_AVAILABLE else ''}")
    print(f"   PDF: {'' if PDF_AVAILABLE else ''}")
    print(f"   NeMo Curator: {'' if NEMO_CURATOR_AVAILABLE else ''}")
    print(f"   SentencePiece: {'' if SENTENCEPIECE_AVAILABLE else ''}")
    print(f"   YAML: {'' if YAML_AVAILABLE else ''}")
    print(f"   psutil: {'' if PSUTIL_AVAILABLE else ''}")
    
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
        print(f"Loading existing cache from {cache_file}...")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            paper = json.loads(line)
                            if 'id' in paper:
                                existing_ids.add(paper['id'])
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                            continue
            print(f"   Found {len(existing_ids)} existing papers in cache")
        except Exception as e:
            print(f"Warning: Error reading cache file: {e}")
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
        print(f"   Error: Empty query")
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
        print("   Signal-based timeout not available (Windows), using time-based checks only")
    
    print(f"\nQuery: {query}")
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
        # Create ArXiv client with ArXiv-recommended 3 second delay
        client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,  # ArXiv recommends 3 seconds between requests
            num_retries=2
        )
        
        # Search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        print(f"   Fetching results from ArXiv API...")
        results_iter = client.results(search)
        
        # Set query-level timeout
        if use_signal_timeout:
            signal.alarm(query_timeout)
        
        result_idx = -1
        for result_idx, result in enumerate(results_iter):
            # Check query timeout (time-based, works on all platforms)
            elapsed = time.time() - query_start
            if elapsed > query_timeout:
                print(f"   Query timeout ({query_timeout}s), aborting")
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
                    print(f"   {papers_found} papers, {elapsed:.0f}s elapsed, {rate:.2f} papers/sec")
                    last_progress_time = time.time()
                
                # Memory check every 100 results
                if papers_found % 100 == 0:
                    if PSUTIL_AVAILABLE:
                        mem_stats = get_memory_usage()
                        if mem_stats['percent'] is not None and mem_stats['percent'] > 80:
                            print(f"   RAM at {mem_stats['percent']:.0f}%, stopping to prevent OOM")
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
                print(f"   Result #{papers_found + 1} took >{per_result_timeout}s, skipping...")
                papers_skipped += 1
                if use_signal_timeout:
                    signal.alarm(0)
                continue
            except Exception as e:
                print(f"   Error processing result: {e}")
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
        total_processed = papers_found + skipped_duplicate + skipped_no_abstract + skipped_date_range
        
        print(f"   Query complete: {papers_found} papers ({rate:.2f} papers/sec)")
        
        # Log deduplication statistics with warnings for high rates
        if skipped_duplicate > 0:
            duplicate_pct = (skipped_duplicate / total_processed * 100) if total_processed > 0 else 0
            if duplicate_pct >= 80:
                print(f"   ⚠️  Skipped {skipped_duplicate} duplicates ({duplicate_pct:.1f}% duplicate rate - very high!)")
                print(f"   💡 Query may be redundant or overlapping with previous queries")
            elif duplicate_pct >= 50:
                print(f"   ⚠️  Skipped {skipped_duplicate} duplicates ({duplicate_pct:.1f}% duplicate rate)")
            else:
                print(f"   Skipped {skipped_duplicate} duplicates ({duplicate_pct:.1f}% duplicate rate)")
        if skipped_no_abstract > 0:
            print(f"   Skipped {skipped_no_abstract} papers without abstracts")
        if skipped_date_range > 0:
            if MIN_YEAR is not None and MAX_YEAR is not None:
                date_range_str = f"{MIN_YEAR}-{MAX_YEAR}"
            elif MIN_YEAR is not None:
                date_range_str = f">={MIN_YEAR}"
            elif MAX_YEAR is not None:
                date_range_str = f"<={MAX_YEAR}"
            else:
                date_range_str = "all years"
            print(f"   Skipped {skipped_date_range} papers outside date range ({date_range_str})")
        
        if papers_found == 0:
            if result_idx == -1:
                print(f"   Warning: No results found for query")
                print(f"      - Query: {query}")
                print(f"      - This might indicate query syntax issue or API problem")
            else:
                print(f"   Warning: Processed {result_idx + 1} results but none matched criteria")
        
    except TimeoutError:
        elapsed = time.time() - query_start
        print(f"   Query timed out after {elapsed:.0f}s")
    except KeyboardInterrupt:
        print(f"\n   Query interrupted by user")
        raise
    except Exception as e:
        print(f"   Query error: {e}")
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
    print("Warning: Using legacy search_arxiv_query() which accumulates results in memory.")
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
            print(f"   {prefix} RAM: {used_gb:.1f}GB/{total_gb:.1f}GB ({percent:.0f}%)")
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
        print(f"   Checkpoint saved: {self.total_collected} papers")
    
    def _load_checkpoint(self):
        """Load checkpoint to resume collection.
        
        Always loads ALL IDs from the output file to ensure accurate deduplication.
        The checkpoint file is only used for the count, but we verify against the actual file.
        """
        try:
            # Always load from output file first (most accurate source of truth)
            if os.path.exists(self.output_file):
                print(f"Loading existing papers from: {self.output_file}")
                existing_ids = load_existing_ids(self.output_file)
                self.collected_ids = existing_ids
                self.total_collected = len(existing_ids)
                if self.total_collected > 0:
                    print(f"Found {self.total_collected} existing papers in output file")
                else:
                    print(f"Output file exists but is empty or has no valid papers")
            else:
                print(f"Output file not found at: {self.output_file}")
                print(f"  Starting fresh collection")
            
            # Also check checkpoint file for count verification
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    checkpoint_count = checkpoint.get('total_collected', 0)
                    if checkpoint_count != self.total_collected:
                        print(f"Checkpoint count ({checkpoint_count}) differs from file count ({self.total_collected}), using file count")
                    # Note: We don't use checkpoint's collected_ids since we loaded all from file
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            if self.total_collected == 0:
                print("Starting fresh")
    
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
            print("   ArXiv package not available")
            return 0
        
        try:
            # Use ArXiv-recommended 3 seconds between requests
            client = arxiv.Client(
                delay_seconds=3.0,  # ArXiv recommends 3 seconds between requests
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
                    
                    # Check RAM less frequently for speed (every 10 papers instead of 5)
                    if papers_in_batch % 10 == 0:
                        ram_percent = self._get_ram_percent()
                        if ram_percent > self.ram_target_percent:
                            print(f"   RAM at {ram_percent:.0f}%, stopping batch early")
                            break
                    
                    # Rate limiting - optimized for Colab speed
                    # Use minimum 2 seconds (faster than 3, still safe)
                    time.sleep(max(self.rate_limit, 2.0))
        
        except Exception as e:
            print(f"   Batch error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        
        elapsed = time.time() - batch_start
        rate = papers_in_batch / elapsed if elapsed > 0 else 0
        
        print(f"   Batch {batch_num}: {papers_in_batch} papers ({rate:.1f} papers/sec)")
        self._log_memory("After batch:")
        
        return papers_in_batch
    
    def collect_query(
        self,
        query: str,
        max_papers: int,
        query_num: int,
        total_queries: int,
        strategy: str = "year_split"
    ) -> int:
        """
        Collect papers for one query, using multiple batches.
        
        Supports different strategies:
        - "direct": Query directly (for small result sets)
        - "year_split": Split query by year to avoid pagination
        - "year_split_truncated": Year split but may lose some papers
        
        Uses a single large search and processes results incrementally to avoid
        getting the same papers in each batch.
        
        Returns: Total papers collected for this query
        """
        print(f"\n{'='*70}")
        print(f"Query {query_num}/{total_queries}: {query[:60]}...")
        print(f"Strategy: {strategy}")
        print(f"{'='*70}")
        
        if not ARXIV_AVAILABLE:
            print("   ArXiv package not available")
            return 0
        
        papers_in_query = 0
        
        # If strategy is year_split, split the query by year
        if strategy in ["year_split", "year_split_truncated"]:
            year_queries = generate_year_split_queries(query)
            print(f"   Split into {len(year_queries)} year-specific queries")
            
            # Collect from each year query
            # Use ArXiv-recommended 3 seconds between requests
            client = arxiv.Client(
                delay_seconds=3.0,  # ArXiv recommends 3 seconds between requests
                num_retries=3,
                page_size=100
            )
            
            papers_per_year = max_papers // len(year_queries) if len(year_queries) > 0 else max_papers
            max_per_year = max(100, papers_per_year)  # At least 100 per year
            
            for year_query in year_queries:
                if self.total_collected >= max_papers:
                    break
                
                # Extract year from query for display
                year_match = re.search(r'(\d{4})01010000', year_query)
                year_display = year_match.group(1) if year_match else "unknown"
                
                remaining = max_papers - papers_in_query
                max_for_year = min(max_per_year, remaining)
                
                print(f"\n   Year {year_display}: Collecting up to {max_for_year} papers...")
                
                # Use pagination-safe collection with 1000 limit per year
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    year_papers, count, hit_limit = collect_with_pagination_safety(
                        client=client,
                        query=year_query,
                        max_papers_to_retrieve=min(1000, max_for_year),  # ArXiv pagination limit
                        existing_ids=self.collected_ids,
                        output_file_handle=f
                    )
                    
                    if hit_limit:
                        print(f"   ⚠️  Year {year_display} query hit pagination limit")
                
                papers_in_query += count
                self.total_collected += count
                
                # Track papers returned from ArXiv (before deduplication)
                papers_from_arxiv = len(year_papers)  # This is papers returned before deduplication check
                
                # Update collected_ids
                for paper in year_papers:
                    self.collected_ids.add(paper['id'])
                
                # Save checkpoint after each year
                self._save_checkpoint()
                self._clear_memory()
                
                if count > 0:
                    print(f"   Year {year_display}: Collected {count} papers (total: {papers_in_query})")
                else:
                    # Explain why no papers were collected
                    if papers_from_arxiv == 0:
                        print(f"   Year {year_display}: No papers returned from ArXiv (query may have no results for this year)")
                    else:
                        duplicates = papers_from_arxiv - count
                        duplicate_pct = (duplicates / papers_from_arxiv * 100) if papers_from_arxiv > 0 else 0
                        print(f"   Year {year_display}: No new papers (ArXiv returned {papers_from_arxiv} papers, all were duplicates)")
                        if duplicate_pct >= 80:
                            print(f"      ⚠️  Very high duplicate rate ({duplicate_pct:.1f}%) - this year's papers already collected")
            
            # Query-level summary with deduplication info
            if papers_in_query == 0:
                print(f"\n⚠️  Query complete: 0 papers collected")
                print(f"   Possible reasons:")
                print(f"   1. All papers from this query were already collected (duplicates)")
                print(f"   2. Query returned no results from ArXiv")
                print(f"   3. All results were filtered out (no abstract, date range, etc.)")
            else:
                print(f"\nQuery complete: {papers_in_query} papers")
            return papers_in_query
        
        # Direct strategy (for small queries)
        batch_num = 0
        total_results_checked = 0  # Track across all batches
        consecutive_empty_batches = 0  # Track consecutive batches with 0 papers
        
        # Fetch many more results to account for filtering (no abstract, date range, etc.)
        # ArXiv API allows up to 300,000 results, so we can request a large number
        # Use a much larger window to ensure we can find enough papers
        # For large targets (30k+), we need to fetch even more results to account for filtering
        max_search_results = max(max_papers * 30, 200000)  # Increased: 30x target, min 200k results
        
        try:
            # ArXiv API rate limiting: Use ArXiv-recommended 3 seconds between requests
            # This prevents rate limiting failures and ensures reliable collection
            client = arxiv.Client(
                delay_seconds=3.0,  # ArXiv recommends 3 seconds between requests
                num_retries=3,  # Increased retries for better reliability
                page_size=100  # Standard page size
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
                    print(f"   High RAM ({ram_percent:.0f}%), reducing batch size to {current_batch_size}")
                elif ram_percent > 50:
                    current_batch_size = max(5, int(self.batch_size * 0.75))
                    print(f"   Moderate RAM ({ram_percent:.0f}%), reducing batch size to {current_batch_size}")
                else:
                    current_batch_size = self.batch_size
                
                # Collect batch from the same iterator
                papers, results_checked, skip_reasons = self._collect_batch_from_iterator(
                    results_iter, query, batch_num, current_batch_size
                )
                
                total_results_checked += results_checked
                
                if papers == 0:
                    consecutive_empty_batches += 1
                    # Check if results were returned but filtered (duplicates, no abstract, etc.)
                    if results_checked > 0:
                        total_skipped = sum(skip_reasons.values())
                        if total_skipped > 0:
                            # Already logged in _collect_batch_from_iterator, just summarize
                            if skip_reasons['duplicates'] > 0:
                                dup_pct = (skip_reasons['duplicates'] / results_checked * 100) if results_checked > 0 else 0
                                if dup_pct >= 80:
                                    print(f"      Total checked so far: {total_results_checked} results")
                        else:
                            print(f"      Total results checked so far: {total_results_checked}")
                    else:
                        print(f"      Total results checked so far: {total_results_checked}")
                    
                    # If we've checked many results with no new papers, move to next query
                    if consecutive_empty_batches >= 3:
                        print(f"   ⚠️  {consecutive_empty_batches} consecutive empty batches, moving to next query")
                        if total_results_checked > 0:
                            print(f"   💡 Query may be exhausted or all results are duplicates")
                        break
                    elif total_results_checked > 5000 and papers_in_query == 0:
                        print(f"   ⚠️  Query returned no papers after checking {total_results_checked} results, moving to next query")
                        print(f"   💡 All results may be duplicates or filtered out")
                        break
                    elif total_results_checked > 200000:  # Increased from 50000 to allow more results (200k = ~4x increase)
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
                
                # Avoid hitting rate limits - brief delay after each batch
                # Reduced delay for speed (batch already has per-request delays)
                time.sleep(0.5)  # Reduced from 2 to 0.5 seconds for faster collection
        
        except StopIteration:
            print("   Reached end of search results")
            # Don't break - continue to next query if target not reached
        except Exception as e:
            print(f"   Query error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            # Continue to next query even on error
        
        # Query-level summary with deduplication info
        if papers_in_query == 0:
            print(f"\n⚠️  Query complete: 0 papers collected")
            if total_results_checked > 0:
                print(f"   Checked {total_results_checked} results from ArXiv")
                print(f"   Possible reasons:")
                print(f"   1. All papers were duplicates (already collected)")
                print(f"   2. All results filtered out (no abstract, date range, etc.)")
                print(f"   3. Query syntax issue or no matching papers")
            else:
                print(f"   No results returned from ArXiv API")
                print(f"   Possible reasons:")
                print(f"   1. Query syntax issue")
                print(f"   2. ArXiv API error or rate limiting")
                print(f"   3. No papers match this query")
        else:
            print(f"\nQuery complete: {papers_in_query} papers")
        return papers_in_query
    
    def _collect_batch_from_iterator(
        self,
        results_iter,
        query: str,
        batch_num: int,
        batch_size: int
    ) -> Tuple[int, int, Dict[str, int]]:
        """
        Collect one batch of papers from an existing results iterator.
        
        This avoids restarting the search and getting duplicate results.
        
        Returns: Tuple of (number of papers collected, number of results checked, skip_reasons dict)
        """
        print(f"\n   Batch {batch_num}: Collecting up to {batch_size} papers...")
        self._log_memory("Before batch:")
        
        papers_in_batch = 0
        batch_start = time.time()
        results_checked = 0
        skip_reasons = {
            'duplicates': 0,
            'no_abstract': 0,
            'date_range': 0,
            'other': 0
        }
        
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for result in results_iter:
                    results_checked += 1
                    paper_id = result.entry_id.split('/')[-1]
                    
                    # Skip if already collected
                    if paper_id in self.collected_ids:
                        skip_reasons['duplicates'] += 1
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
                        skip_reasons['date_range'] += 1
                        continue
                    if MAX_YEAR is not None and year is not None and year > MAX_YEAR:
                        skip_reasons['date_range'] += 1
                        continue
                    
                    # Skip if no abstract
                    if not result.summary or not result.summary.strip():
                        skip_reasons['no_abstract'] += 1
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
                    
                    # Check RAM less frequently for speed (every 10 papers instead of 5)
                    if papers_in_batch % 10 == 0:
                        ram_percent = self._get_ram_percent()
                        if ram_percent > self.ram_target_percent:
                            print(f"   RAM at {ram_percent:.0f}%, stopping batch early")
                            break
                    
                    # Rate limiting - optimized for Colab speed
                    # Use minimum 2 seconds (faster than 3, still safe)
                    time.sleep(max(self.rate_limit, 2.0))
        
        except StopIteration:
            # End of results
            pass
        except Exception as e:
            # Check if it's an ArXiv API error (empty page, rate limiting, etc.)
            is_empty_page_error = False
            if UnexpectedEmptyPageError and isinstance(e, UnexpectedEmptyPageError):
                is_empty_page_error = True
            elif "UnexpectedEmptyPageError" in str(type(e).__name__) or "empty" in str(e).lower():
                is_empty_page_error = True
            
            if is_empty_page_error:
                print(f"   ArXiv API returned empty page (likely rate limiting or pagination issue)")
                print(f"   Collected {papers_in_batch} papers before error, continuing to next query...")
                # Return what we have - the iterator is broken, so we'll move to next query
            else:
                print(f"   Batch error: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
        
        elapsed = time.time() - batch_start
        rate = papers_in_batch / elapsed if elapsed > 0 else 0
        
        # Log batch results with skip reasons
        if papers_in_batch > 0:
            print(f"   Batch {batch_num}: {papers_in_batch} papers ({rate:.1f} papers/sec, checked {results_checked} results)")
        else:
            # Explain why batch returned 0 papers
            total_skipped = sum(skip_reasons.values())
            if total_skipped > 0:
                skip_details = []
                if skip_reasons['duplicates'] > 0:
                    skip_details.append(f"{skip_reasons['duplicates']} duplicates")
                if skip_reasons['no_abstract'] > 0:
                    skip_details.append(f"{skip_reasons['no_abstract']} no abstract")
                if skip_reasons['date_range'] > 0:
                    skip_details.append(f"{skip_reasons['date_range']} date range")
                if skip_reasons['other'] > 0:
                    skip_details.append(f"{skip_reasons['other']} other")
                
                print(f"   Batch {batch_num}: 0 papers (checked {results_checked} results)")
                print(f"      Skipped: {', '.join(skip_details)}")
                if skip_reasons['duplicates'] > 0:
                    dup_pct = (skip_reasons['duplicates'] / results_checked * 100) if results_checked > 0 else 0
                    if dup_pct >= 80:
                        print(f"      ⚠️  Very high duplicate rate ({dup_pct:.1f}%)")
            else:
                print(f"   Batch {batch_num}: 0 papers (checked {results_checked} results, no results from ArXiv)")
        
        self._log_memory("After batch:")
        
        return papers_in_batch, results_checked, skip_reasons
    
    def collect_all(
        self,
        queries: List[Tuple[str, int]],
        total_target: int,
        query_strategies: Optional[Dict[str, str]] = None
    ):
        """
        Collect papers from multiple queries.
        
        Args:
            queries: List of (query_str, max_papers_per_query) tuples
            total_target: Total papers to collect
            query_strategies: Optional dict mapping query strings to strategies
                (e.g., {"query": "year_split"}). If None, uses default "year_split" for all.
        """
        print("\n" + "="*70)
        print("RAM-Efficient ArXiv Collection")
        print("="*70)
        print(f"Starting from: {self.total_collected} papers")
        print(f"Target: {total_target} papers")
        print(f"Batch size: {self.batch_size} papers/batch")
        print(f"RAM target: <{self.ram_target_percent:.0f}%")
        if MIN_YEAR is not None or MAX_YEAR is not None:
            date_range = f"{MIN_YEAR or 'any'}-{MAX_YEAR or 'any'}"
            print(f"Date range: {date_range}")
        else:
            print(f"Date range: All years (no filtering)")
        print()
        
        try:
            for query_num, (query, max_per_query) in enumerate(queries, 1):
                if self.total_collected >= total_target:
                    print(f"\nReached target of {total_target} papers")
                    break
                
                # Adjust max for this query
                remaining = total_target - self.total_collected
                max_for_query = min(max_per_query, remaining)
                
                print(f"\nQuery {query_num}/{len(queries)}: {query}")
                print(f"   Target: {max_for_query} papers (remaining: {remaining})")
                
                # Get strategy for this query
                strategy = "year_split"  # Default
                if query_strategies is not None:
                    if query in query_strategies:
                        strategy = query_strategies[query]
                    else:
                        # Try to match by query prefix (in case of slight variations)
                        for q, s in query_strategies.items():
                            if query.startswith(q[:30]) or q.startswith(query[:30]):
                                strategy = s
                                break
                
                # Collect query
                papers = self.collect_query(
                    query=query,
                    max_papers=max_for_query,
                    query_num=query_num,
                    total_queries=len(queries),
                    strategy=strategy
                )
                
                # If we got papers but haven't reached target, continue to next query
                if papers > 0 and self.total_collected < total_target:
                    print(f"   Collected {papers} papers from this query, continuing...")
                elif papers == 0 and self.total_collected < total_target:
                    print(f"   ⚠️  Query returned 0 papers, but target not reached ({self.total_collected}/{total_target})")
                    print(f"   💡 This query may have no new results (all duplicates) or no matching papers")
                    print(f"   Continuing to next query...")
            
            # After all queries, check if we need more papers
            if self.total_collected < total_target:
                remaining = total_target - self.total_collected
                print(f"\nTarget not reached: {self.total_collected}/{total_target} papers ({remaining} remaining)")
                print("All queries exhausted. Consider:")
                print("  1. Adding more diverse queries")
                print("  2. Relaxing date filtering (if enabled)")
                print("  3. Reducing quality filters")
                print("  4. Increasing max_search_results per query")
        
        except KeyboardInterrupt:
            print("\nCollection paused by user")
        
        finally:
            self._save_checkpoint()
            print(f"\nTotal collected: {self.total_collected} papers")
            if self.total_collected < total_target:
                print(f"Target: {total_target} papers (shortfall: {total_target - self.total_collected})")
            self._log_memory("Final:")


def collect_arxiv_efficient(
    output_dir: str = "./data/arxiv",
    total_target: int = 10000,
    batch_size: int = 10,
    ram_target: float = 50.0,
    rate_limit: float = 0.33,
    use_drive: bool = True
):
    """
    Collect ArXiv papers efficiently without OOM.
    
    Args:
        output_dir: Output directory (local fallback)
        total_target: Total papers to collect
        batch_size: Papers per batch (adjust for RAM)
        ram_target: Target RAM percentage to stay below
        rate_limit: Delay between requests (seconds)
        use_drive: If True, use Google Drive if available (default: True)
    """
    if not ARXIV_AVAILABLE:
        error_msg = "Error: arxiv package not available. Install with: pip install arxiv"
        print(error_msg)
        raise ImportError(error_msg)
    
    # Use Google Drive if available and requested
    if use_drive:
        output_dir = get_drive_output_dir(local_output_dir=output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "arxiv_papers.jsonl")
    checkpoint_file = os.path.join(output_dir, "collection_checkpoint.json")
    
    # Debug: Print the exact paths being used
    print(f"\nCollection paths:")
    print(f"  Output directory: {output_dir}")
    print(f"  Output file: {output_file}")
    print(f"  Checkpoint file: {checkpoint_file}")
    print(f"  Output file exists: {os.path.exists(output_file)}")
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"  Output file size: {file_size:,} bytes")
    
    # Initialize collector
    collector = RAMEfficientArxivCollector(
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        batch_size=batch_size,
        ram_target_percent=ram_target,
        rate_limit=rate_limit
    )
    
    # Option 1: Use optimized query list with strategies (recommended)
    # Check if optimized queries file exists
    optimized_file = os.path.join(output_dir, "optimized_queries.json")
    if os.path.exists(optimized_file):
        print(f"Loading optimized queries from {optimized_file}...")
        try:
            with open(optimized_file, 'r') as f:
                optimized_data = json.load(f)
            queries = []
            query_strategies = {}
            for item in optimized_data.get('optimized_queries', []):
                query = item['query']
                strategy = item.get('strategy', 'year_split')
                # Use a reasonable max_papers per query
                max_papers = min(total_target // len(optimized_data.get('optimized_queries', [])), 5000)
                queries.append((query, max_papers))
                query_strategies[query] = strategy
            print(f"Loaded {len(queries)} optimized queries with strategies")
        except Exception as e:
            print(f"Could not load optimized queries: {e}")
            print("Falling back to default queries...")
            queries = []
            query_strategies = None
            for tier_name, tier_queries in OPTIMIZED_QUERIES.items():
                for query, max_papers in tier_queries:
                    adjusted_limit = min(max_papers, total_target)
                    queries.append((query, adjusted_limit))
    else:
        # Option 2: Use default tiered queries
        queries = []
        query_strategies = None
        for tier_name, tier_queries in OPTIMIZED_QUERIES.items():
            for query, max_papers in tier_queries:
                adjusted_limit = min(max_papers, total_target)
                queries.append((query, adjusted_limit))
    
    print(f"Using {len(queries)} queries")
    query_strategies_dict = query_strategies if query_strategies is not None else None
    if query_strategies_dict:
        strategy_counts = {}
        for q, s in query_strategies_dict.items():
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        print(f"Strategies: {strategy_counts}")
    else:
        print("Using default year_split strategy for all queries")
    
    # Collect
    collector.collect_all(queries, total_target, query_strategies=query_strategies_dict)
    
    print(f"\nPapers saved to: {output_file}")
    print(f"Checkpoint saved to: {checkpoint_file}")
    print(f"To resume: Run collect_arxiv_efficient() again")


def collect_arxiv_papers(
    output_dir: str = "./data/arxiv",
    max_papers: int = 40000,
    cache_file: str = None,
    rate_limit_delay: float = RATE_LIMIT_DELAY,
    batch_size: int = 10,
    ram_target: float = 50.0,
    use_drive: bool = True
):
    """Main function to collect ArXiv papers using RAM-efficient batch collection.
    
    Args:
        output_dir: Directory to save output files (local fallback)
        max_papers: Maximum total papers to collect
        cache_file: Path to cache file (default: output_dir/arxiv_papers.jsonl)
        rate_limit_delay: Delay between API requests (seconds)
        batch_size: Papers per batch (default: 10 for RAM efficiency)
        ram_target: Target RAM percentage to stay below (default: 50%)
        use_drive: If True, use Google Drive if available (default: True)
    """
    if cache_file is None:
        cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
    
    # Use the efficient collector
    collect_arxiv_efficient(
        output_dir=output_dir,
        total_target=max_papers,
        batch_size=batch_size,
        ram_target=ram_target,
        rate_limit=rate_limit_delay,
        use_drive=use_drive
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
            # Suppress harmless PyPDF2 warnings about unknown widths/formatting
            import warnings
            import logging
            # Suppress PyPDF2 warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Also suppress logging warnings from PyPDF2
                pypdf2_logger = logging.getLogger("PyPDF2")
                old_level = pypdf2_logger.level
                pypdf2_logger.setLevel(logging.ERROR)
                try:
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
                finally:
                    # Restore original logging level
                    pypdf2_logger.setLevel(old_level)
        
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
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load already processed IDs (from files and checkpoint)
    processed_ids = load_processed_ids(output_dir)
    print(f"Found {len(processed_ids)} already processed papers")
    
    # Load checkpoint if exists
    checkpoint_file = os.path.join(output_dir, 'pdf_extraction_checkpoint.json')
    checkpoint_stats = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                checkpoint_stats = checkpoint_data.get('stats', {})
                print(f"Resuming from checkpoint: {checkpoint_stats.get('success', 0)} successful, {checkpoint_stats.get('failed', 0)} failed")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    # Load papers from JSONL (stream, don't load all in memory)
    print(f"Loading papers from {input_jsonl}...")
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
                print(f"Warning: Invalid JSON on line {line_num}")
                continue
    
    print(f"Total papers in file: {total_papers}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining to process: {len(papers_to_process)}")
    print()
    
    if not papers_to_process:
        print("All papers already processed!")
        return
    
    # Statistics
    stats = {
        'total': len(papers_to_process),
        'success': 0,
        'failed': 0,
        'errors': defaultdict(int)
    }
    
    # Process papers with thread pool
    print(f"Starting extraction with {num_workers} workers...")
    print(f"Rate limit: {1.0/rate_limit_delay:.1f} requests/sec")
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
                    print(f"   Progress: {completed}/{len(papers_to_process)} "
                          f"(OK {stats['success']} success, FAIL {stats['failed']} failed)")
                
                # Checkpoint
                if completed % CHECKPOINT_INTERVAL_PDF == 0:
                    save_checkpoint(output_dir, processed_ids, stats)
                    print(f"   Checkpoint saved: {completed} papers processed")
            
            except Exception as e:
                stats['failed'] += 1
                stats['errors'][f"Exception: {str(e)}"] += 1
                print(f"   Unexpected error processing paper: {e}")
    
    # Final checkpoint
    save_checkpoint(output_dir, processed_ids, stats)
    
    # Print summary
    print()
    print("=" * 60)
    print("PDF Extraction Complete!")
    print("=" * 60)
    print(f"Total processed: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output directory: {output_dir}")
    
    if stats['errors']:
        print(f"\nError breakdown:")
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
    print("Domain Classifier & Text Preprocessor")
    print("=" * 60)
    print(f"Metadata file: {metadata_jsonl}")
    print(f"Text directory: {text_dir}")
    print(f"Output file: {output_jsonl}")
    print(f"Workers: {num_workers}")
    print()
    
    # Load metadata
    print("Loading metadata...")
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
                print(f"Warning: Invalid JSON on line {line_num}")
                continue
    
    print(f"   Loaded {total_papers} papers from metadata")
    
    # Find text files
    print(f"\n📂 Scanning text directory...")
    text_files = {}
    if os.path.exists(text_dir):
        for filename in os.listdir(text_dir):
            if filename.endswith('.txt'):
                arxiv_id = filename[:-4]  # Remove .txt extension
                text_files[arxiv_id] = os.path.join(text_dir, filename)
    
    print(f"   Found {len(text_files)} text files")
    
    # Match metadata with text files
    papers_to_process = []
    for arxiv_id, metadata in metadata_by_id.items():
        if arxiv_id in text_files:
            papers_to_process.append((arxiv_id, text_files[arxiv_id], metadata))
    
    print(f"   {len(papers_to_process)} papers ready to process")
    print()
    
    if not papers_to_process:
        print("No papers to process!")
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
    print(f"Starting preprocessing and classification...")
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
                    print(f"   Progress: {completed}/{len(papers_to_process)} "
                          f"(OK {stats['success']} success, FAIL {stats['failed']} failed)")
            
            except Exception as e:
                stats['failed'] += 1
                stats['errors'][f"Exception: {str(e)}"] += 1
                print(f"   Unexpected error processing {arxiv_id}: {e}")
    
    # Final statistics pass (read output file to get domain counts)
    print(f"\nComputing final statistics...")
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
    print("Preprocessing & Classification Complete!")
    print("=" * 60)
    print(f"Total processed: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output file: {output_jsonl}")
    
    print(f"\nDomain distribution:")
    for domain, count in sorted(stats['domain_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {domain}: {count} papers")
    
    print(f"\nNeurodegeneration papers: {stats['neurodegeneration_count']}")
    
    if stats['errors']:
        print(f"\nError breakdown:")
        for error, count in sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {error}: {count}")
    
    # Estimate file size
    if os.path.exists(output_jsonl):
        file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
        print(f"\nOutput file size: {file_size_mb:.2f} MB")


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
            # Score: more lenient - use sqrt to give credit for partial matches
            # If 1/3 of keywords match, score is ~0.58 instead of 0.33
            if matches > 0:
                score = min((matches / len(keywords)) ** 0.7, 1.0)  # More lenient scoring
            else:
                score = 0.0
            domain_scores[domain] = score
        
        # Calculate overall relevance
        # Use MAX domain score as base (more lenient - if paper matches any domain well, it's relevant)
        max_domain_score = max(domain_scores.values()) if domain_scores else 0.0
        
        # Also consider average for multi-domain papers
        avg_domain_score = sum(domain_scores.values()) / len(domain_scores) if domain_scores else 0.0
        
        # Base score: weighted combination of max and average (favor max for single-domain papers)
        base_score = (max_domain_score * 0.7 + avg_domain_score * 0.3)
        
        # Boost for multiple domains (multi-domain papers are more relevant)
        active_domains = sum(1 for score in domain_scores.values() if score > 0.1)
        if active_domains > 1:
            base_score *= (1.0 + 0.15 * (active_domains - 1))  # 15% boost per additional domain
        
        # Boost for paper length (longer papers tend to be more substantial)
        word_count = len(text.split())
        if word_count > 500:
            base_score *= 1.15  # 15% boost for longer papers
        elif word_count < 200:
            base_score *= 0.85  # Penalty for very short papers
        
        # Count medical terms presence (more lenient)
        medical_term_count = sum(1 for term in self.medical_terms if term in text)
        medical_boost = min(medical_term_count / 15.0, 0.25)  # Up to 25% boost, more lenient threshold
        base_score += medical_boost
        
        # ML keywords boost (papers with ML terms are more relevant for our use case)
        ml_keywords = ['machine learning', 'deep learning', 'neural network', 'transformer', 
                      'lstm', 'cnn', 'classification', 'prediction', 'model', 'algorithm',
                      'training', 'supervised', 'unsupervised', 'reinforcement learning']
        ml_matches = sum(1 for keyword in ml_keywords if keyword in text)
        ml_boost = min(ml_matches / 10.0, 0.2)  # Up to 20% boost for ML terms
        base_score += ml_boost
        
        # Minimum boost if ANY domain has matches (ensures healthcare papers aren't filtered out)
        if max_domain_score > 0:
            base_score = max(base_score, 0.3)  # At least 0.3 if any domain matches
        
        # Normalize to 0-1 range
        relevance = min(base_score, 1.0)
        
        # Add relevance to scores
        domain_scores['relevance'] = relevance
        
        return domain_scores
    
    def filter_document(self, document: Dict, min_relevance: float = 0.3) -> bool:
        """Filter document based on relevance score.
        
        Args:
            document: Document dictionary
            min_relevance: Minimum relevance score to keep (default: 0.3)
            
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


def create_domain_relevance_filter(min_score: float = 0.3):
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
        
        # Check requirements (relaxed for better coverage):
        # - At least 1 healthcare domain keyword OR at least 1 ML method keyword
        # - Domain + ML relevance score > 0.4 (lowered from 0.6)
        healthcare_keywords_found = len(domains) > 0
        ml_keywords_found = sum(1 for keyword in self.ml_keywords if keyword in cleaned_text.lower()) >= 1  # Lowered from 2
        relevance_score = (domain_score * 0.6 + ml_score * 0.4)
        
        # Accept if: (healthcare OR ML) AND relevance > 0.4
        if not ((healthcare_keywords_found or ml_keywords_found) and relevance_score > 0.4):
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
        print(f"HealthcareFilterStage Statistics:")
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
        print(f"HealthcareQualityFilterStage Statistics:")
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
        print("HealthcareJsonlWriter Final Statistics")
        print("=" * 60)
        
        print(f"\nTotal documents processed: {self.stats['total_processed']}")
        
        # Domain distribution
        if self.stats['domain_counts']:
            print(f"\nDomain distribution:")
            for domain, count in sorted(
                self.stats['domain_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / self.stats['total_processed']) * 100
                print(f"   {domain}: {count} ({percentage:.1f}%)")
        
        # Year distribution
        if self.stats['year_counts']:
            print(f"\nYear distribution:")
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
        
        print(f"\nOutput file: {self.output_path}")
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
            print(f"Could not load checkpoint: {e}")
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
            "Error: NeMo Curator not available.\n"
            "   Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'\n"
            "   Note: NeMo Curator only supports Linux systems"
        )
        print(error_msg)
        raise RuntimeError("NeMo Curator not available. Install with: pip install 'nemo-curator[text]'")
    
    if not Pipeline_AVAILABLE:
        error_msg = "Error: Pipeline class not available. Check NeMo Curator installation."
        print(error_msg)
        raise RuntimeError(error_msg)
    
    if not download_arxiv_AVAILABLE:
        print("Error: download_arxiv() function not available.")
        print("   This requires NeMo Curator with download support")
        return None
    
    print("=" * 60)
    print("NeMo Curator Healthcare Pipeline (FREE - No AWS Required)")
    print("=" * 60)
    print(f"Raw data directory: {raw_data_path}")
    print(f"Raw output file: {raw_output_path}")
    print(f"Final curated output: {output_path}")
    print(f"Filter query: {filter_query}")
    print(f"Max workers: {max_workers}")
    print(f"Batch size: {batch_size} papers per batch")
    print(f"Checkpoint interval: {checkpoint_interval} papers")
    print(f"Max papers: {max_papers}")
    print(f"Cost: FREE (direct ArXiv access, no AWS charges)")
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
            print(f"Resuming from checkpoint: {total_downloaded} papers already downloaded")
            print(f"   Found {len(downloaded_ids)} unique paper IDs in checkpoint")
    
    try:
        # Initialize Dask client
        print("Initializing Dask client...")
        try:
            client = get_client(cluster_type="cpu")
            print(f"   Dask client initialized: {client}")
        except:
            # Fallback: create local client
            from dask.distributed import Client
            client = Client(processes=False, threads_per_worker=max_workers)
            print(f"   Created local Dask client: {client}")
        
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
            print(f"   Starting download (target: {max_papers} papers)...")
            
            dataset = download_arxiv(
                output_path=raw_data_path,
                max_workers=max_workers,
                filter_query=filter_query
            )
            
            # Process dataset in batches with checkpointing
            print(f"   Processing downloaded dataset in batches of {batch_size}...")
            
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
                    print(f"\n   Reached target of {max_papers} papers")
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
                    print(f"   Progress: {total_downloaded}/{max_papers} papers downloaded...")
                
                # Checkpoint every N papers
                if total_downloaded % checkpoint_interval == 0:
                    save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
                    print(f"   Checkpoint saved: {total_downloaded} papers downloaded")
                
                # Process batch if full
                if papers_in_batch >= batch_size:
                    batch_count += 1
                    print(f"   Batch {batch_count} complete: {papers_in_batch} papers")
                    papers_in_batch = 0
                    
                    # Save checkpoint after each batch
                    save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
            
            # Final checkpoint
            save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
            raw_output_handle.close()
            
            print(f"\n   Download complete: {total_downloaded} papers downloaded")
            print(f"   Final checkpoint saved")
            
        except Exception as e:
            raw_output_handle.close()
            print(f"   Download interrupted: {e}")
            print(f"   Saving checkpoint with {total_downloaded} papers...")
            save_download_checkpoint(checkpoint_file, downloaded_ids, total_downloaded)
            raise
        
        # Stage 2: Process downloaded JSONL using Pipeline API
        print("\nStage 2: Processing with NeMo Curator Pipeline API")
        print("   Creating pipeline with custom healthcare stages...")
        
        if JsonlReader_AVAILABLE and ProcessingStage_AVAILABLE:
            # Use Pipeline API (preferred method)
            print("   Using NeMo Curator Pipeline API")
            
            # Create pipeline
            pipeline = Pipeline(name="healthcare_curation_pipeline")
            
            # Step 1: Read JSONL files
            print("   Adding JsonlReader stage...")
            reader = JsonlReader(
                file_paths=raw_output_path,
                files_per_partition=4,
                fields=["text", "file_name"]  # Read text and filename fields
            )
            pipeline.add_stage(reader)
            
            # Step 2: Healthcare filtering and classification
            print("   Adding HealthcareFilterStage...")
            healthcare_filter = HealthcareFilterStage()
            pipeline.add_stage(healthcare_filter)
            
            # Step 3: Quality filtering and deduplication
            print("   Adding HealthcareQualityFilterStage...")
            quality_filter = HealthcareQualityFilterStage()
            pipeline.add_stage(quality_filter)
            
            # Step 4: Write curated dataset
            print("   Adding JsonlWriter stage...")
            if JsonlWriter_AVAILABLE:
                writer = JsonlWriter(path=output_path)
                pipeline.add_stage(writer)
            else:
                # Fallback: use custom writer
                print("   JsonlWriter not available, using custom writer...")
                custom_writer = HealthcareJsonlWriter(output_path=output_path)
                pipeline.add_stage(custom_writer)
            
            # Execute pipeline
            print("\n   Executing pipeline...")
            results = pipeline.run()
            print("   Pipeline execution complete!")
            
        else:
            # Fallback: manual processing (if Pipeline API not available)
            print("   Pipeline API not fully available, falling back to manual processing...")
            raw_documents = []
            with open(raw_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            doc = json.loads(line)
                            raw_documents.append(doc)
                        except json.JSONDecodeError:
                            continue
            
            print(f"   Loaded {len(raw_documents)} documents for processing")
            
            # Process with custom stages
            filtered_dataset = HealthcareFilterStage()(raw_documents)
            
            # Quality filtering and deduplication
            print("\nStage 3: Quality Filtering & Deduplication")
            print("   Using HealthcareQualityFilterStage...")
            quality_filtered_dataset = HealthcareQualityFilterStage()(filtered_dataset)
            
            # Write curated dataset
            print("\nStage 4: Writing Curated Dataset")
            print("   Using HealthcareJsonlWriter...")
            final_output = HealthcareJsonlWriter(output_path=output_path)(quality_filtered_dataset)
        
        print("\nPipeline completed successfully!")
        print(f"Final curated dataset: {output_path}")
        print(f"Raw data (for reference): {raw_output_path}")
        print(f"Total papers downloaded: {total_downloaded}")
        
        return output_path
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        print(traceback.format_exc())
        print(f"\nCheckpoint saved - you can resume by running again with resume=True")
        return None


def curate_with_nemo(
    text_dir: str,
    metadata_jsonl: str,
    output_jsonl: str,
    use_gpu: bool = False,
    skip_dedup: bool = False,
    min_relevance_score: float = 0.3
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
            "Error: NeMo Curator not available.\n"
            "   Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'\n"
            "   Note: NeMo Curator only supports Linux systems"
        )
        print(error_msg)
        raise RuntimeError("NeMo Curator not available. Install with: pip install 'nemo-curator[text]'")
    
    print("=" * 60)
    print("NeMo Curator Text Curation Pipeline")
    print("=" * 60)
    print(f"Text directory: {text_dir}")
    print(f"Metadata file: {metadata_jsonl}")
    print(f"Output file: {output_jsonl}")
    print(f"Min relevance score: {min_relevance_score}")
    print(f"GPU deduplication: {use_gpu}")
    print(f"Skip deduplication: {skip_dedup}")
    print(f"NEMO_CURATOR_AVAILABLE: {NEMO_CURATOR_AVAILABLE}")
    print()
    
    # Verify inputs exist
    if not os.path.exists(text_dir):
        error_msg = f"Text directory does not exist: {text_dir}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not os.path.exists(metadata_jsonl):
        error_msg = f"Metadata file does not exist: {metadata_jsonl}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    print(f"Found {len(text_files)} text files in {text_dir}")
    
    if len(text_files) == 0:
        error_msg = f"No text files found in {text_dir}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Initialize Dask client for parallelization
    try:
        if get_client is not None:
            try:
                client = get_client()
                print(f"Dask client initialized: {client}")
            except:
                # Create local Dask client
                from dask.distributed import Client
                client = Client(processes=False, threads_per_worker=2)
                print(f"Created local Dask client: {client}")
        else:
            # Create local Dask client manually
            from dask.distributed import Client
            client = Client(processes=False, threads_per_worker=2)
            print(f"Created local Dask client: {client}")
    except Exception as e:
        print(f"Dask client creation failed: {e}")
        print("   Continuing without Dask (will use sequential processing)")
        client = None
    
    # Load metadata
    print("Loading metadata...")
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
    print(f"   Loaded {len(metadata_map)} metadata entries")
    
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
            print(f"   Error loading {filename}: {e}")
            continue
    
    print(f"   Loaded {len(documents)} documents")
    initial_count = len(documents)
    
    # Create DocumentDataset (NeMo Curator format)
    # For compatibility, use list if NeMo Curator not available
    if NEMO_CURATOR_AVAILABLE and DocumentDataset_AVAILABLE and DocumentDataset is not None:
        try:
            dataset = DocumentDataset(documents)
            print("   Using NeMo Curator DocumentDataset")
        except Exception as e:
            print(f"   DocumentDataset creation failed: {e}, using list")
            dataset = documents
    else:
        dataset = documents
        if not NEMO_CURATOR_AVAILABLE:
            print("   Using list-based dataset (NeMo Curator not available)")
        elif not DocumentDataset_AVAILABLE or DocumentDataset is None:
            print("   Using list-based dataset (DocumentDataset not available)")
    
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
            print("   Text cleaning applied (NeMo Curator)")
        except Exception as e:
            print(f"   NeMo Curator cleaning failed: {e}, using fallback")
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
            print("   Simple text cleaning applied")
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
        print("   Simple text cleaning applied (NeMo Curator not available)")
    
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
        print(f"   Word count filter: {len(dataset)}/{before_quality} documents")
        
        dataset = [doc for doc in dataset if alphanumeric_filter(doc)]
        after_alnum = len(dataset)
        print(f"   Alphanumeric filter: {after_alnum} documents")
        
        dataset = [doc for doc in dataset if language_filter(doc)]
        after_lang = len(dataset)
        print(f"   Language filter: {after_lang} documents")
    else:
        dataset = dataset.filter(word_count_filter)
        print(f"   Word count filter: {len(dataset)}/{before_quality} documents")
        
        dataset = dataset.filter(alphanumeric_filter)
        after_alnum = len(dataset)
        print(f"   Alphanumeric filter: {after_alnum} documents")
        
        dataset = dataset.filter(language_filter)
        after_lang = len(dataset)
        print(f"   Language filter: {after_lang} documents")
    
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
                print(f"   NeMo Curator ScoreFilter failed: {e}, using fallback")
                dataset = [doc for doc in dataset if domain_filter.filter_document(doc, min_relevance=min_relevance_score)]
        else:
            dataset = [doc for doc in dataset if domain_filter.filter_document(doc, min_relevance=min_relevance_score)]
    
    after_domain = len(dataset)
    print(f"   Domain relevance filter: {after_domain}/{before_domain} documents")
    print(f"   Relevance threshold: {min_relevance_score}")
    
    # Count domain distribution
    domain_counts = defaultdict(int)
    for doc in (dataset if isinstance(dataset, list) else list(dataset)):
        domains = doc.get('domains', [])
        for domain in domains:
            domain_counts[domain] += 1
    
    if domain_counts:
        print(f"   Domain distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {domain}: {count} documents")
    
    # Stage 5: Deduplication (optional)
    if not skip_dedup:
        print("\n" + "=" * 60)
        print("Stage 5: Deduplication")
        print("=" * 60)
        
        before_dedup = len(dataset)
        
        try:
            if NEMO_CURATOR_AVAILABLE and FuzzyDedup_AVAILABLE and FuzzyDedup is not None:
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
                    if DocumentDataset_AVAILABLE and DocumentDataset is not None:
                        try:
                            temp_dataset = DocumentDataset(dataset)
                            temp_dataset = deduplicator(temp_dataset)
                            dataset = list(temp_dataset)
                        except Exception as e2:
                            print(f"   FuzzyDedup on DocumentDataset failed: {e2}, using simple deduplication")
                            # Fallback: simple deduplication by text hash
                            seen_texts = set()
                            unique_docs = []
                            for doc in dataset:
                                text_hash = hash(doc.get('text', ''))
                                if text_hash not in seen_texts:
                                    seen_texts.add(text_hash)
                                    unique_docs.append(doc)
                            dataset = unique_docs
                    else:
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
                print(f"   Deduplication: {after_dedup}/{before_dedup} documents")
            else:
                # FuzzyDedup not available - use simple deduplication
                if not NEMO_CURATOR_AVAILABLE:
                    print("   NeMo Curator not available, using simple deduplication")
                elif not FuzzyDedup_AVAILABLE or FuzzyDedup is None:
                    print("   FuzzyDedup not available, using simple deduplication")
                else:
                    print("   NeMo Curator deduplication not available, using simple hash-based dedup")
                # Simple deduplication by text hash
                seen_texts = set()
                unique_docs = []
                for doc in (dataset if isinstance(dataset, list) else list(dataset)):
                    text_hash = hash(doc.get('text', ''))
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        unique_docs.append(doc)
                dataset = unique_docs
                print(f"   Simple deduplication: {len(dataset)}/{before_dedup} documents")
        except Exception as e:
            print(f"   Deduplication failed: {e}")
            print("   Continuing without deduplication...")
    else:
        print("\n" + "=" * 60)
        print("Stage 5: Deduplication (Skipped)")
        print("=" * 60)
        print("   Deduplication skipped as requested")
    
    # Stage 6: Format & Export
    print("\n" + "=" * 60)
    print("Stage 6: Format & Export")
    print("=" * 60)
    
    # Domains are already assigned by HealthcareDomainFilter in Stage 4
    # No need to add them again
    
    # Export to JSONL
    print(f"   Exporting to {output_jsonl}...")
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
    print("Curation Complete!")
    print("=" * 60)
    print(f"Initial documents: {initial_count}")
    print(f"After quality filtering: {after_lang}")
    print(f"After domain filtering: {after_domain}")
    print(f"Final curated documents: {final_count}")
    print(f"Retention rate: {final_count/initial_count*100:.1f}%")
    print(f"Output file: {output_jsonl}")
    
    # Quality score distribution
    if final_count > 0:
        docs_list = dataset if isinstance(dataset, list) else list(dataset)
        scores = [doc.get('relevance_score', 0.0) for doc in docs_list]
        if scores:
            import numpy as np
            print(f"\nQuality Score Distribution:")
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
        
        print(f"   Sampling {sample_size} documents for validation...")
        
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
        
        print(f"   Documents with detected domains: {validation_results['with_domains']}/{sample_size}")
        print(f"   Average relevance score: {validation_results['avg_relevance']:.3f}")
        print(f"   High relevance documents (>0.7): {validation_results['high_relevance']}/{sample_size}")
        print(f"   Domain distribution in sample:")
        for domain, count in sorted(validation_results['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"      {domain}: {count} documents")
    
    # Close Dask client
    try:
        if client is not None:
            client.close()
            print("Dask client closed")
    except Exception as e:
        print(f"Error closing Dask client: {e}")
    
    # Final confirmation
    print("\n" + "=" * 60)
    print("curate_with_nemo() completed successfully!")
    print(f"Output file: {output_jsonl}")
    if os.path.exists(output_jsonl):
        file_size = os.path.getsize(output_jsonl) / (1024 * 1024)  # MB
        count = sum(1 for _ in open(output_jsonl))
        print(f"Output file size: {file_size:.2f} MB")
        print(f"Documents in output: {count}")
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
    print("Healthcare Text Processing Pipeline")
    print("=" * 60)
    print(f"Input file: {input_jsonl}")
    print(f"Output file: {output_jsonl}")
    print(f"Workers: {num_workers}")
    print()
    
    # Initialize modifier
    modifier = HealthcareTextModifier()
    
    # Load documents
    print("Loading curated dataset...")
    documents = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"   Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    total_docs = len(documents)
    print(f"   Loaded {total_docs} documents")
    print()
    
    # Process documents in parallel
    print("Processing documents...")
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
                    print(f"   Error processing document: {error}")
            else:
                processed_docs.append(result)
            
            # Progress update
            if completed % 500 == 0:
                print(f"   Progress: {completed}/{total_docs} documents processed...")
    
    print(f"   Processed {len(processed_docs)} documents")
    if errors > 0:
        print(f"   Errors: {errors} documents failed")
    print()
    
    # Write output
    print("Writing processed dataset...")
    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else '.', exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for doc in processed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Statistics
    print()
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total documents: {total_docs}")
    print(f"Successfully processed: {len(processed_docs)}")
    print(f"Failed: {errors}")
    print(f"Output file: {output_jsonl}")
    
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
        
        print(f"\nStatistics:")
        print(f"   Average tokens per document: {avg_tokens:.0f}")
        print(f"   Average medical terms per document: {avg_medical_terms:.1f}")
        print(f"   Sections detected:")
        for section, count in sorted(sections_found.items(), key=lambda x: x[1], reverse=True):
            print(f"      {section}: {count} documents")
        
        # File size
        file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
        print(f"\nOutput file size: {file_size_mb:.2f} MB")


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
    print(f"Extracting texts from {input_jsonl}...")
    
    total_papers = 0
    total_chars = 0
    null_chars_removed = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
         open(output_txt, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                text = record.get('text', '')
                
                if text and text.strip():
                    # Remove null characters (0x00) which cause SentencePiece warnings
                    # Replace with space to preserve text structure
                    null_count = text.count('\x00')
                    if null_count > 0:
                        null_chars_removed += null_count
                        text = text.replace('\x00', ' ')
                    
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
                print(f"Warning: Invalid JSON on line {line_num}")
                continue
    
    file_size_mb = os.path.getsize(output_txt) / (1024 * 1024)
    print(f"Extracted {total_papers} papers, {total_chars:,} characters")
    if null_chars_removed > 0:
        print(f"   Removed {null_chars_removed:,} null characters (to prevent SentencePiece warnings)")
    print(f"Output file size: {file_size_mb:.2f} MB")
    print(f"Output file: {output_txt}")
    
    return total_papers, total_chars


def train_tokenizer(
    input_txt: str,
    model_prefix: str = 'healthcare_tokenizer',
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    model_type: str = TOKENIZER_MODEL_TYPE,
    char_coverage: float = TOKENIZER_CHAR_COVERAGE,
    special_tokens: List[str] = None,
    normalization: str = TOKENIZER_NORMALIZATION,
    skip_if_exists: bool = True,
    force_retrain: bool = False
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
        skip_if_exists: If True, skip training if model files already exist (default: True)
        force_retrain: If True, retrain even if model files exist (default: False)
    """
    if not SENTENCEPIECE_AVAILABLE:
        print("Error: sentencepiece package not available.")
        print("   Install with: pip install sentencepiece")
        return None
    
    if special_tokens is None:
        special_tokens = TOKENIZER_SPECIAL_TOKENS
    
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    
    # Check if tokenizer already exists (resume capability)
    if not force_retrain and skip_if_exists:
        if os.path.exists(model_file) and os.path.exists(vocab_file):
            # Verify files are non-empty
            model_size = os.path.getsize(model_file)
            vocab_size_check = os.path.getsize(vocab_file)
            
            if model_size > 0 and vocab_size_check > 0:
                print("=" * 60)
                print("🔤 Tokenizer Training (Resume Check)")
                print("=" * 60)
                print(f"✅ Found existing tokenizer files:")
                print(f"   Model: {model_file} ({model_size:,} bytes)")
                print(f"   Vocab: {vocab_file} ({vocab_size_check:,} bytes)")
                print()
                print("   Skipping training (tokenizer already exists)")
                print("   To force retraining, set force_retrain=True")
                print("=" * 60)
                return model_prefix
    
    print("=" * 60)
    print("🔤 Training SentencePiece BPE Tokenizer")
    print("=" * 60)
    print(f"Input file: {input_txt}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model type: {model_type}")
    print(f"Character coverage: {char_coverage}")
    print(f"🔤 Special tokens: {special_tokens}")
    print(f"📝 Normalization: {normalization}")
    if force_retrain:
        print("⚠️  Force retrain: True (will overwrite existing tokenizer)")
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
    
    print("Starting tokenizer training...")
    print("   This may take several minutes for large datasets...")
    print()
    
    try:
        spm.SentencePieceTrainer.train(**train_args)
        print("Tokenizer training complete!")
        print(f"Model file: {model_prefix}.model")
        print(f"Vocab file: {model_prefix}.vocab")
        return model_prefix
    except Exception as e:
        print(f"Error training tokenizer: {e}")
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
        print("Error: sentencepiece package not available.")
        return {}
    
    if medical_terms is None:
        medical_terms = MEDICAL_TERMS
    
    print("=" * 60)
    print("Tokenizer Validation")
    print("=" * 60)
    
    # Load tokenizer
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        print(f"Loaded tokenizer from {model_path}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return {}
    
    # Validate medical terms
    print(f"\nValidating {len(medical_terms)} medical terms...")
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
    
    print(f"Single-token coverage: {single_token_count}/{len(medical_terms)} ({efficiency:.1f}%)")
    print(f"Multi-token terms: {multi_token_count}/{len(medical_terms)}")
    
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
    
    print(f"\nValidation report saved to: {output_file}")


def train_healthcare_tokenizer(
    input_jsonl: str,
    output_dir: str = './data/arxiv',
    model_prefix: str = 'healthcare_tokenizer',
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    skip_if_exists: bool = True,
    force_retrain: bool = False
):
    """Complete pipeline: extract texts, train tokenizer, validate.
    
    Args:
        input_jsonl: Input JSONL file with processed papers
        output_dir: Output directory for tokenizer files
        model_prefix: Prefix for tokenizer model files
        vocab_size: Vocabulary size for tokenizer
        skip_if_exists: If True, skip training if tokenizer already exists (default: True)
        force_retrain: If True, retrain even if tokenizer exists (default: False)
    """
    print("=" * 60)
    print("🔤 Healthcare Tokenizer Training Pipeline")
    print("=" * 60)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, model_prefix)
    model_file = f"{model_path}.model"
    vocab_file = f"{model_path}.vocab"
    metadata_file = os.path.join(output_dir, 'tokenizer_metadata.json')
    
    # Check if tokenizer already exists and if dataset has changed
    should_retrain_due_to_dataset_change = False
    
    if not force_retrain and skip_if_exists:
        if os.path.exists(model_file) and os.path.exists(vocab_file):
            model_size = os.path.getsize(model_file)
            vocab_size_check = os.path.getsize(vocab_file)
            
            if model_size > 0 and vocab_size_check > 0:
                # Check if dataset has changed since last training
                if os.path.exists(input_jsonl) and os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            tokenizer_metadata = json.load(f)
                        
                        # Get current dataset stats
                        current_file_size = os.path.getsize(input_jsonl)
                        current_mtime = os.path.getmtime(input_jsonl)
                        current_paper_count = sum(1 for line in open(input_jsonl) if line.strip())
                        
                        # Get stored dataset stats
                        stored_file_size = tokenizer_metadata.get('input_file_size', 0)
                        stored_mtime = tokenizer_metadata.get('input_file_mtime', 0)
                        stored_paper_count = tokenizer_metadata.get('input_paper_count', 0)
                        stored_input_file = tokenizer_metadata.get('input_file', '')
                        
                        # Check if dataset has changed
                        dataset_changed = False
                        change_reasons = []
                        
                        if stored_input_file != input_jsonl:
                            dataset_changed = True
                            change_reasons.append("input file path changed")
                        
                        if current_file_size != stored_file_size:
                            dataset_changed = True
                            change_reasons.append(f"file size changed ({stored_file_size:,} → {current_file_size:,} bytes)")
                        
                        if abs(current_mtime - stored_mtime) > 1.0:  # Allow 1 second tolerance
                            dataset_changed = True
                            change_reasons.append("file modification time changed")
                        
                        if current_paper_count != stored_paper_count:
                            dataset_changed = True
                            change_reasons.append(f"paper count changed ({stored_paper_count} → {current_paper_count} papers)")
                        
                        if dataset_changed:
                            should_retrain_due_to_dataset_change = True
                            print("⚠️  Dataset has changed since last tokenizer training:")
                            for reason in change_reasons:
                                print(f"   - {reason}")
                            print()
                            print("   Retraining tokenizer to include new vocabulary...")
                            print()
                        else:
                            print("✅ Found existing tokenizer files - skipping training")
                            print(f"   Model: {model_file} ({model_size:,} bytes)")
                            print(f"   Vocab: {vocab_file} ({vocab_size_check:,} bytes)")
                            print()
                            print(f"   Dataset unchanged: {current_paper_count} papers, {current_file_size:,} bytes")
                            print("   To force retraining, set force_retrain=True")
                            print("=" * 60)
                            return model_path
                    except Exception as e:
                        print(f"⚠️  Could not read tokenizer metadata: {e}")
                        print("   Will retrain to be safe...")
                        print()
                        should_retrain_due_to_dataset_change = True
                else:
                    # No metadata file - assume dataset might be different, but check if files exist
                    if os.path.exists(metadata_file):
                        # Metadata exists but couldn't read it - retrain to be safe
                        should_retrain_due_to_dataset_change = True
                    else:
                        # No metadata - this is first training or metadata was deleted
                        # Check if input file exists and has content
                        if os.path.exists(input_jsonl):
                            current_paper_count = sum(1 for line in open(input_jsonl) if line.strip())
                            if current_paper_count > 0:
                                print("✅ Found existing tokenizer files - skipping training")
                                print(f"   Model: {model_file} ({model_size:,} bytes)")
                                print(f"   Vocab: {vocab_file} ({vocab_size_check:,} bytes)")
                                print()
                                print("   ⚠️  No tokenizer metadata found - cannot verify dataset hasn't changed")
                                print("   To force retraining, set force_retrain=True")
                                print("=" * 60)
                                return model_path
    
    # If we get here, we need to train (either files don't exist, or dataset changed, or force_retrain)
    if should_retrain_due_to_dataset_change:
        print("🔄 Retraining tokenizer due to dataset changes...")
        print()
    
    # Step 1: Extract texts
    temp_txt_file = os.path.join(output_dir, 'training_texts.txt')
    print("Step 1: Extracting texts from JSONL...")
    total_papers, total_chars = extract_texts_from_jsonl(input_jsonl, temp_txt_file)
    print()
    
    if total_papers == 0:
        error_msg = f"No papers found in input file: {input_jsonl}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Step 2: Train tokenizer
    print("Step 2: Training SentencePiece tokenizer...")
    trained_model = train_tokenizer(
        input_txt=temp_txt_file,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=TOKENIZER_MODEL_TYPE,
        char_coverage=TOKENIZER_CHAR_COVERAGE,
        special_tokens=TOKENIZER_SPECIAL_TOKENS,
        normalization=TOKENIZER_NORMALIZATION,
        skip_if_exists=skip_if_exists,
        force_retrain=force_retrain
    )
    print()
    
    if not trained_model:
        error_msg = "Tokenizer training failed!"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Step 3: Validate tokenizer
    print("Step 3: Validating tokenizer...")
    model_file = f"{model_path}.model"
    validation_report = validate_tokenizer(model_file, medical_terms=MEDICAL_TERMS)
    print()
    
    # Step 4: Save validation report and metadata
    if validation_report:
        report_file = os.path.join(output_dir, 'tokenizer_validation_report.json')
        save_validation_report(validation_report, report_file)
        print()
        
        # Save tokenizer metadata (for future dataset change detection)
        if os.path.exists(input_jsonl):
            current_file_size = os.path.getsize(input_jsonl)
            current_mtime = os.path.getmtime(input_jsonl)
            current_paper_count = sum(1 for line in open(input_jsonl) if line.strip())
            
            tokenizer_metadata = {
                'input_file': input_jsonl,
                'input_file_size': current_file_size,
                'input_file_mtime': current_mtime,
                'input_paper_count': current_paper_count,
                'vocab_size': vocab_size,
                'model_type': TOKENIZER_MODEL_TYPE,
                'char_coverage': TOKENIZER_CHAR_COVERAGE,
                'trained_at': datetime.now().isoformat(),
                'model_file': model_file,
                'vocab_file': f"{model_path}.vocab"
            }
            
            metadata_file = os.path.join(output_dir, 'tokenizer_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(tokenizer_metadata, f, indent=2)
            print(f"✅ Saved tokenizer metadata: {metadata_file}")
            print()
        
        # Print summary
        print("=" * 60)
        print("Tokenizer Training Complete!")
        print("=" * 60)
        print(f"Model file: {model_file}")
        print(f"Vocab file: {model_path}.vocab")
        print(f"Validation report: {report_file}")
        print(f"Vocabulary size: {validation_report.get('vocab_size', vocab_size)}")
        print(f"Medical term efficiency: {validation_report.get('efficiency_percent', 0):.1f}%")
        print()
        
        # Print term results
        print("Medical term tokenization results:")
        term_results = validation_report.get('term_results', {})
        for term, result in sorted(term_results.items()):
            status = "OK" if result['is_single'] else "FAIL"
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
    print("Healthcare MoE Data Pipeline - Full Orchestration")
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
    print("Running diagnostics...")
    arxiv_ok = print_diagnostics(config)
    
    if not arxiv_ok:
        print("\nArXiv connection check failed. Continuing anyway...")
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
            print(f"Stage 1: {stages[stage_name]}")
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
                        print("Warning: 0 papers collected")
                        print("   This might be due to:")
                        print("   - ArXiv API rate limiting")
                        print("   - Network issues")
                        print("   - Query syntax problems")
                        print("   - All papers filtered out")
                        stage_results[stage_name] = {'success': False, 'papers': 0}
                    else:
                        stage_results[stage_name] = {'success': True, 'papers': count}
                        print(f"Collected {count} papers")
                else:
                    stage_results[stage_name] = {'success': False, 'papers': 0}
                    
            except Exception as e:
                print(f"Stage 1 failed: {e}")
                import traceback
                print(traceback.format_exc())
                stage_results[stage_name] = {'success': False, 'error': str(e)}
                if not resume:
                    raise
            
            stage_elapsed = time.time() - stage_start
            print(f"Stage 1 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\nSkipping stage: {stages['collect']}")
            stage_results['collect'] = {'success': True, 'skipped': True}
        
        # Stage 2: Extract PDFs
        if 'extract' not in skip_stages:
            stage_name = 'extract'
            print("\n" + "=" * 80)
            print(f"Stage 2: {stages[stage_name]}")
            print("=" * 80)
            
            stage_start = time.time()
            extraction_config = config.get('extraction', {})
            text_dir = os.path.join(output_dir, "texts")
            cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
            
            if not os.path.exists(cache_file):
                print(f"Metadata file not found: {cache_file}")
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
                    print(f"Extracted {len(text_files)} text files")
                    
                except Exception as e:
                    print(f"Stage 2 failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    stage_results[stage_name] = {'success': False, 'error': str(e)}
                    if not resume:
                        raise
            
            stage_elapsed = time.time() - stage_start
            print(f"Stage 2 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\nSkipping stage: {stages['extract']}")
            stage_results['extract'] = {'success': True, 'skipped': True}
        
        # Stage 3: NeMo Curator curation (optional)
        if 'curate' not in skip_stages:
            curation_config = config.get('curation', {})
            use_nemo = curation_config.get('use_nemo_curator', True) and NEMO_CURATOR_AVAILABLE
            
            if use_nemo:
                stage_name = 'curate'
                print("\n" + "=" * 80)
                print(f"Stage 3: {stages[stage_name]}")
                print("=" * 80)
                
                stage_start = time.time()
                text_dir = os.path.join(output_dir, "texts")
                cache_file = os.path.join(output_dir, "arxiv_papers.jsonl")
                curated_file = os.path.join(output_dir, "curated_dataset.jsonl")
                
                if not os.path.exists(text_dir) or not os.listdir(text_dir):
                    print(f"Text directory empty or missing: {text_dir}")
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
                            min_relevance_score=curation_config.get('min_relevance_score', 0.3)
                        )
                        
                        if os.path.exists(curated_file):
                            count = sum(1 for _ in open(curated_file))
                            stage_results[stage_name] = {'success': True, 'papers': count}
                            print(f"Curated {count} papers")
                        else:
                            stage_results[stage_name] = {'success': False, 'error': 'Output file not created'}
                            
                    except Exception as e:
                        print(f"Stage 3 failed: {e}")
                        import traceback
                        print(traceback.format_exc())
                        stage_results[stage_name] = {'success': False, 'error': str(e)}
                        if not resume:
                            raise
                
                stage_elapsed = time.time() - stage_start
                print(f"Stage 3 completed in {stage_elapsed:.1f}s")
            else:
                print(f"\nSkipping NeMo Curator curation (not available or disabled)")
                stage_results['curate'] = {'success': True, 'skipped': True}
        else:
            print(f"\nSkipping stage: {stages['curate']}")
            stage_results['curate'] = {'success': True, 'skipped': True}
        
        # Stage 4: Preprocess and classify
        if 'preprocess' not in skip_stages:
            stage_name = 'preprocess'
            print("\n" + "=" * 80)
            print(f"Stage 4: {stages[stage_name]}")
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
                print(f"No input file found")
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
                        print(f"Processed {count} papers")
                    else:
                        stage_results[stage_name] = {'success': False, 'error': 'Output file not created'}
                        
                except Exception as e:
                    print(f"Stage 4 failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    stage_results[stage_name] = {'success': False, 'error': str(e)}
                    if not resume:
                        raise
            
            stage_elapsed = time.time() - stage_start
            print(f"Stage 4 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\nSkipping stage: {stages['preprocess']}")
            stage_results['preprocess'] = {'success': True, 'skipped': True}
        
        # Stage 5: Train tokenizer
        if 'tokenize' not in skip_stages:
            stage_name = 'tokenize'
            print("\n" + "=" * 80)
            print(f"Stage 5: {stages[stage_name]}")
            print("=" * 80)
            
            stage_start = time.time()
            tokenizer_config = config.get('tokenizer', {})
            processed_file = os.path.join(output_dir, "processed_dataset.jsonl")
            
            if not os.path.exists(processed_file):
                print(f"Processed dataset not found: {processed_file}")
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
                        print(f"Tokenizer trained: {model_file}")
                    else:
                        stage_results[stage_name] = {'success': False, 'error': 'Model file not created'}
                        
                except Exception as e:
                    print(f"Stage 5 failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    stage_results[stage_name] = {'success': False, 'error': str(e)}
                    if not resume:
                        raise
            
            stage_elapsed = time.time() - stage_start
            print(f"Stage 5 completed in {stage_elapsed:.1f}s")
        else:
            print(f"\nSkipping stage: {stages['tokenize']}")
            stage_results['tokenize'] = {'success': True, 'skipped': True}
        
        # Generate final report
        total_elapsed = time.time() - start_time
        generate_final_report(stage_results, output_dir, total_elapsed)
        
        print("\n" + "=" * 80)
        print("Pipeline Complete!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
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
                status = "Success" if result.get('success') else "Failed"
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
        
        print(f"\nFinal report saved:")
        print(f"   JSON: {report_file}")
        print(f"   Text: {report_text_file}")
        
    except Exception as e:
        print(f"Could not save report: {e}")


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
    collect_parser.add_argument(
        '--use-optimized',
        action='store_true',
        help='Use optimized queries from optimized_queries.json if available'
    )
    collect_parser.add_argument(
        '--optimized-file',
        type=str,
        default=None,
        help='Path to optimized_queries.json file (default: output_dir/optimized_queries.json)'
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
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze query feasibility and recommend optimal strategies')
    analyze_parser.add_argument('--query', type=str, help='Analyze a single query')
    analyze_parser.add_argument('--all', action='store_true', help='Analyze all queries from OPTIMIZED_QUERIES')
    analyze_parser.add_argument('--optimize', action='store_true', help='Build optimized query list with refined queries')
    analyze_parser.add_argument('--output', type=str, help='Save report to JSON file')
    
    # Optimize and execute command (master script)
    optimize_execute_parser = subparsers.add_parser('optimize-execute', help='Complete query optimization and execution pipeline')
    optimize_execute_parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/arxiv',
        help='Output directory for papers and reports (default: ./data/arxiv)'
    )
    optimize_execute_parser.add_argument(
        '--skip-diagnostic',
        action='store_true',
        help='Skip diagnostic analysis step'
    )
    optimize_execute_parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip collection step (only optimize queries)'
    )
    optimize_execute_parser.add_argument(
        '--max-papers-per-query',
        type=int,
        default=1000,
        help='Maximum papers per sub-query (default: 1000)'
    )
    optimize_execute_parser.add_argument(
        '--rate-limit',
        type=float,
        default=2.0,
        help='Rate limit delay in seconds (default: 2.0)'
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
        # Check if user wants to use optimized queries
        if args.use_optimized or args.optimized_file:
            # Use optimized query execution
            optimized_file = args.optimized_file
            if not optimized_file:
                optimized_file = os.path.join(args.output_dir, "optimized_queries.json")
            
            if not os.path.exists(optimized_file):
                print(f"⚠️  Optimized queries file not found: {optimized_file}")
                print("   Run 'python data_pipeline.py analyze --optimize --output <file>' first")
                print("   Falling back to standard collection...")
                rate_limit_delay = 1.0 / args.rate_limit
                collect_arxiv_papers(
                    output_dir=args.output_dir,
                    max_papers=args.max_papers,
                    cache_file=args.cache_file,
                    rate_limit_delay=rate_limit_delay,
                    batch_size=args.batch_size,
                    ram_target=args.ram_target
                )
            else:
                # Load optimized queries
                print(f"Loading optimized queries from {optimized_file}...")
                with open(optimized_file, 'r') as f:
                    optimized_data = json.load(f)
                
                optimized_queries = []
                for item in optimized_data.get('optimized_queries', []):
                    query = item['query']
                    strategy = item.get('strategy', 'year_split')
                    optimized_queries.append((query, strategy))
                
                print(f"Loaded {len(optimized_queries)} optimized queries")
                
                # Set up output files
                os.makedirs(args.output_dir, exist_ok=True)
                output_jsonl = args.cache_file or os.path.join(args.output_dir, "arxiv_papers.jsonl")
                checkpoint_jsonl = os.path.join(args.output_dir, "collection_checkpoint.json")
                
                # Execute optimized queries
                total_collected = execute_optimized_queries(
                    optimized_queries=optimized_queries,
                    output_jsonl=output_jsonl,
                    checkpoint_jsonl=checkpoint_jsonl,
                    max_papers_per_query=1000,
                    rate_limit_delay=1.0 / args.rate_limit
                )
                
                print(f"\n✅ Collection complete: {total_collected} papers")
                print(f"   Output: {output_jsonl}")
        else:
            # Use standard collection
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
    elif args.command == 'analyze':
        if args.query:
            # Analyze single query
            report = analyze_query_feasibility(args.query)
            print("\n" + "="*70)
            print("QUERY ANALYSIS RESULT")
            print("="*70)
            if report.get('error'):
                print(f"ERROR: {report['error']}")
            else:
                print(f"Query: {report['query']}")
                print(f"Total results: {report['total_results']:,}")
                print(f"Estimated retrievable: {report['estimated_papers']:,} ({report['retrieval_rate']*100:.0f}%)")
                print(f"Strategy: {report['strategy']}")
                print(f"Time estimate: {report['time_estimate_seconds']} seconds")
                print(f"Risky years: {report['risky_years'] if report['risky_years'] else 'None'}")
                if report['by_year']:
                    print("\nYear breakdown:")
                    for year in range(2024, 2014, -1):
                        count = report['by_year'].get(year, 0)
                        marker = "*" if year in report['risky_years'] else " "
                        print(f"  {marker} {year}: {count:,} papers")
            
            # Save report if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"\nReport saved to: {args.output}")
        elif args.all:
            # Analyze all optimized queries
            report = analyze_optimized_queries()
            # Save report if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"\nReport saved to: {args.output}")
        elif args.optimize:
            # Build optimized query list
            # Flatten OPTIMIZED_QUERIES to get base queries
            base_queries = []
            for tier_name, tier_queries in OPTIMIZED_QUERIES.items():
                for query, max_papers in tier_queries:
                    base_queries.append(query)
            
            optimized = build_optimized_queries(base_queries)
            
            # Print optimized list
            print("\n" + "="*70)
            print("OPTIMIZED QUERY LIST")
            print("="*70)
            for i, (query, strategy) in enumerate(optimized, 1):
                print(f"{i}. [{strategy:20s}] {query}")
            
            # Save if requested
            if args.output:
                output_data = {
                    'optimized_queries': [{'query': q, 'strategy': s} for q, s in optimized],
                    'total_queries': len(optimized),
                    'strategies': {
                        'direct': len([q for q, s in optimized if s == 'direct']),
                        'year_split': len([q for q, s in optimized if s == 'year_split']),
                        'year_split_truncated': len([q for q, s in optimized if s == 'year_split_truncated']),
                    }
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nOptimized query list saved to: {args.output}")
        else:
            analyze_parser.print_help()
            return
    elif args.command == 'optimize-execute':
        # Get base queries from OPTIMIZED_QUERIES
        base_queries = []
        for tier_name, tier_queries in OPTIMIZED_QUERIES.items():
            for query, max_papers in tier_queries:
                base_queries.append(query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in base_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        base_queries = unique_queries
        
        print(f"Using {len(base_queries)} base queries from OPTIMIZED_QUERIES")
        
        # Run optimization and execution pipeline
        optimize_and_execute_queries(
            base_queries=base_queries,
            output_dir=args.output_dir,
            run_diagnostic=not args.skip_diagnostic,
            run_collection=not args.skip_collection,
            max_papers_per_query=args.max_papers_per_query,
            rate_limit_delay=args.rate_limit
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
