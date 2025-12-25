#!/usr/bin/env python3
"""
Collect a NEW test set from ArXiv to evaluate on.
This ensures no data leakage - test papers are never seen during training.
"""

import argparse
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set

import requests


def fetch_arxiv_papers(
    query: str,
    max_results: int = 1000,
    days_back: int = 30,
    rate_limit: float = 0.5
) -> List[Dict]:
    """Fetch papers from ArXiv API.

    Args:
        query: Search query
        max_results: Maximum number of papers to fetch
        days_back: Number of days back to search (default: 30)
        rate_limit: Seconds between requests

    Returns:
        List of paper metadata
    """
    base_url = "http://export.arxiv.org/api/query"

    # Build date range for recent papers
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    papers = []
    batch_size = 100  # ArXiv API max per request
    start = 0

    print(f"Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Query: {query}")
    print(f"Max results: {max_results}")

    while start < max_results:
        # Build search query with date filter
        search_query = f"{query} AND submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO {end_date.strftime('%Y%m%d')}2359]"

        params = {
            'search_query': search_query,
            'start': start,
            'max_results': min(batch_size, max_results - start),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse ArXiv API response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            # Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('atom:entry', ns)
            if not entries:
                print(f"No more papers found")
                break

            for entry in entries:
                # Extract paper info
                arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]

                # Check if already in training set
                if arxiv_id in seen_papers:
                    continue

                title = entry.find('atom:title', ns).text.strip()
                summary = entry.find('atom:summary', ns).text.strip()

                # Get categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    cat_id = category.get('term')
                    if cat_id:
                        categories.append(cat_id)

                # Get authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)

                paper = {
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'abstract': summary,
                    'categories': categories,
                    'authors': authors,
                    'published': entry.find('atom:published', ns).text,
                    'domains': classify_domains(categories, title, summary),  # Will be used for stratified split
                    'relevance_score': 1.0,  # All test papers are relevant
                    'has_neurodegeneration': False  # Not needed for test set
                }

                papers.append(paper)
                seen_papers.add(arxiv_id)

            print(f"  Fetched {len(entries)} papers ({len(papers)} total unique)")

            start += len(entries)

            # Rate limiting
            time.sleep(rate_limit)

        except Exception as e:
            print(f"Error fetching papers: {e}")
            break

    return papers


def classify_domains(
    categories: List[str],
    title: str,
    abstract: str
) -> List[str]:
    """Classify paper into domains based on categories and text.

    Args:
        categories: ArXiv categories
        title: Paper title
        abstract: Paper abstract

    Returns:
        List of domain labels
    """
    domains = []
    text = f"{title} {abstract}".lower()

    # Domain keywords
    domain_keywords = {
        'neurodegeneration': ['alzheimer', 'parkinson', 'dementia', 'als', 'huntington', 'prion', 'lewy'],
        'neuroscience': ['brain', 'neural', 'neuron', 'synapse', 'cortex', 'cognitive', 'motor'],
        'medical_imaging': ['mri', 'ct', 'ultrasound', 'x-ray', 'imaging', 'tomography'],
        'clinical': ['clinical', 'trial', 'patient', 'diagnosis', 'treatment', 'therapy'],
        'drug_discovery': ['drug', 'compound', 'molecule', 'screening', 'pharmaceutical'],
    }

    # Check against keywords
    for domain, keywords in domain_keywords.items():
        if any(keyword in text for keyword in keywords):
            domains.append(domain)

    # Also classify by ArXiv categories
    for cat in categories:
        if cat.startswith('q-bio'):  # Quantitative Biology
            if 'neuroscience' not in domains:
                domains.append('neuroscience')
        elif cat.startswith('cs.CV'):  # Computer Vision
            if 'medical_imaging' not in domains:
                domains.append('medical_imaging')
        elif cat in ['cs.HC', 'cs.CY']:  # Human-Computer, Cybernetics
            if 'clinical' not in domains:
                domains.append('clinical')

    # Default to general_ml_health if no specific domain
    if not domains:
        domains.append('general_ml_health')

    return domains


def load_training_papers(metadata_path: str) -> Set[str]:
    """Load ArXiv IDs from training set to exclude from test set.

    Args:
        metadata_path: Path to training metadata JSONL

    Returns:
        Set of ArXiv IDs
    """
    seen_papers = set()

    if not os.path.exists(metadata_path):
        print(f"Warning: Training metadata not found at {metadata_path}")
        print(f"Will not filter out training papers from test set")
        return seen_papers

    print(f"Loading training papers from: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as open_f:
        for line in open_f:
            try:
                paper = json.loads(line)
                arxiv_id = paper.get('arxiv_id') or paper.get('id')
                if arxiv_id:
                    seen_papers.add(arxiv_id)
            except Exception as e:
                continue

    print(f"  Loaded {len(seen_papers)} training papers to exclude")
    return seen_papers


def save_test_set(
    papers: List[Dict],
    output_dir: str,
    metadata_file: str,
    split_ratio: float = 0.5
):
    """Save test set with stratified split.

    Args:
        papers: List of paper metadata
        output_dir: Output directory
        metadata_file: Metadata filename
        split_ratio: Ratio for test/validation split
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stratified split by domain
    domain_groups = {}
    for paper in papers:
        for domain in paper.get('domains', ['general_ml_health']):
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(paper)

    print(f"\nDomain distribution in collected papers:")
    for domain, group in domain_groups.items():
        print(f"  {domain}: {len(group)} papers")

    # Split into test and validation
    test_papers = []
    val_papers = []

    random.seed(42)  # Reproducible split

    for domain, group in domain_groups.items():
        n_val = max(1, int(len(group) * split_ratio))
        random.shuffle(group)

        val_papers.extend(group[:n_val])
        test_papers.extend(group[n_val:])

    print(f"\nFinal split:")
    print(f"  Test: {len(test_papers)} papers")
    print(f"  Validation: {len(val_papers)} papers")

    # Save metadata
    test_metadata_path = os.path.join(output_dir, metadata_file.replace('.jsonl', '_test.jsonl'))
    val_metadata_path = os.path.join(output_dir, metadata_file.replace('.jsonl', '_val.jsonl'))

    with open(test_metadata_path, 'w', encoding='utf-8') as f:
        for paper in test_papers:
            f.write(json.dumps(paper) + '\n')

    with open(val_metadata_path, 'w', encoding='utf-8') as f:
        for paper in val_papers:
            f.write(json.dumps(paper) + '\n')

    print(f"\n✅ Test set saved to: {test_metadata_path}")
    print(f"✅ Validation set saved to: {val_metadata_path}")

    # Save summary stats
    stats = {
        'collected_date': datetime.now().isoformat(),
        'total_test_papers': len(test_papers),
        'total_val_papers': len(val_papers),
        'domain_distribution': {
            domain: len([p for p in test_papers if domain in p.get('domains', [])])
            for domain in domain_groups.keys()
        }
    }

    stats_path = os.path.join(output_dir, 'test_set_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Stats saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect new test set from ArXiv (no data leakage)"
    )
    parser.add_argument(
        '--training-metadata',
        type=str,
        required=True,
        help='Path to training metadata JSONL (to exclude those papers)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/test_set',
        help='Output directory for test set'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        default=2000,
        help='Maximum number of papers to collect'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Number of days back to search (default: 30)'
    )
    parser.add_argument(
        '--metadata-file',
        type=str,
        default='test_metadata.jsonl',
        help='Output metadata filename'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.5,
        help='Test/validation split ratio (default: 0.5 = 50/50)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ArXiv Test Set Collection")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Max papers: {args.max_papers}")
    print(f"Days back: {args.days_back}")
    print()

    # Load training papers to exclude
    global seen_papers
    seen_papers = load_training_papers(args.training_metadata)

    # Define search queries for healthcare+ML
    queries = [
        # Neurodegeneration
        "cat:cs.LG OR cat:cs.AI OR cat:q-bio.NC alzheimer OR parkinson OR dementia OR ALS OR huntington",

        # Neuroscience
        "cat:cs.LG OR cat:cs.AI OR cat:q-bio.NC (brain OR neural OR neuron) AND (learning OR model OR network)",

        # Medical Imaging
        "cat:cs.CV OR cat:cs.LG (mri OR ct OR ultrasound OR x-ray) AND (deep OR learning OR CNN)",

        # Clinical NLP
        "cat:cs.CL OR cat:cs.LG (clinical OR patient OR diagnosis) AND (language OR NLP OR text)",

        # Drug Discovery
        "cat:cs.LG OR cat:cs.AI OR cat:q-bio (drug OR molecule) AND (learning OR prediction OR screening)",

        # General Health ML
        "cat:cs.LG OR cat:cs.AI (health OR medical) AND (learning OR prediction OR model)"
    ]

    all_papers = []
    papers_per_query = args.max_papers // len(queries)

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Running query: {query[:100]}...")
        papers = fetch_arxiv_papers(
            query=query,
            max_results=papers_per_query,
            days_back=args.days_back,
            rate_limit=0.5
        )
        all_papers.extend(papers)
        print(f"  Collected {len(papers)} papers")

    # Remove duplicates
    seen = set()
    unique_papers = []
    for paper in all_papers:
        if paper['arxiv_id'] not in seen:
            seen.add(paper['arxiv_id'])
            unique_papers.append(paper)

    print(f"\nTotal unique papers collected: {len(unique_papers)}")

    if len(unique_papers) == 0:
        print("❌ No papers collected. Please check your query or try increasing --days-back")
        return

    # Save test set
    save_test_set(
        papers=unique_papers,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        split_ratio=args.split_ratio
    )

    print("\n" + "=" * 60)
    print("✅ Test set collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
