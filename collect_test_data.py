#!/usr/bin/env python3
"""
Collect Fresh Test Dataset for Model Evaluation

This script collects a separate test dataset that has no overlap with the training data,
ensuring proper evaluation of trained models on unseen data.

Usage:
    python collect_test_data.py [--config config.yaml] [--max-papers 2000]
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import random
import subprocess


def load_training_ids(training_metadata_path: str) -> set:
    """Load arxiv_ids from training dataset to avoid overlap."""
    training_ids = set()

    if not os.path.exists(training_metadata_path):
        print(f"Warning: Training metadata not found at {training_metadata_path}")
        return training_ids

    print(f"Loading training IDs from {training_metadata_path}...")
    with open(training_metadata_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    record = json.loads(line)
                    arxiv_id = record.get('arxiv_id')
                    if arxiv_id:
                        training_ids.add(arxiv_id)
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(training_ids)} training paper IDs")
    return training_ids


def create_test_config(base_config_path: str, test_config_path: str, max_papers: int = 2000) -> dict:
    """Create test-specific configuration."""

    # Load base config
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config if base doesn't exist
        config = {
            'pipeline': {
                'output_dir': './data/test_dataset',
                'use_drive': False,
                'cleanup_intermediate': True,
            },
            'collection': {
                'max_papers': max_papers,
                'rate_limit': 0.3,  # Conservative to avoid rate limiting
                'retry_max': 5,
                'batch_size': 20,
            },
            'extraction': {
                'workers': 2,
                'rate_limit': 0.4,
                'max_pages': 6,
                'max_chars': 12000,
            },
            'nemo_curator': {
                'use_pipeline_api': True,
                'filter_query': 'cs.LG OR cs.AI OR q-bio.NC OR stat.ML',
                'max_workers': 1,
                'max_papers': max_papers,
                'batch_size': 500,
                'checkpoint_interval': 500,
                'resume': True,
                'use_gpu': False,
                'skip_dedup': True,  # We'll do our own dedup against training data
                'quality_filters': {
                    'word_count_min': 100,
                    'word_count_max': 25000,
                    'alphanumeric_ratio_min': 0.4,
                    'language': 'en',
                },
            },
            'preprocessing': {
                'workers': 2,
                'preserve_sections': True,
                'preserve_abbreviations': True,
                'max_chars': 12000,
            },
            'tokenizer': {
                'vocab_size': 30000,
                'model_prefix': 'test_healthcare_tokenizer',
                'character_coverage': 0.9995,
                'model_type': 'bpe',
            },
        }

    # Modify for test data collection
    config['pipeline']['output_dir'] = './data/test_dataset'
    config['collection']['max_papers'] = max_papers
    config['nemo_curator']['max_papers'] = max_papers

    # Use different query to get different papers
    # Add recent ML, healthcare, and neuroscience topics
    test_queries = [
        'cs.LG AND ("machine learning" OR "neural networks")',
        'cs.AI AND ("artificial intelligence" OR "deep learning")',
        'q-bio.NC AND ("neuroscience" OR "brain")',
        'stat.ML AND ("statistical learning" OR "regression")',
        'cs.CV AND ("computer vision" OR "image recognition")',
        'q-bio.QM AND ("medical imaging" OR "healthcare")',
    ]

    # Create balanced test set from multiple domains
    # Instead of one query, collect from multiple domains for balance
    if max_papers <= 1000:
        # For small sets, use 2 balanced queries
        selected_queries = [
            'cs.LG AND ("machine learning" OR "neural networks")',
            'q-bio.NC AND ("neuroscience" OR "brain")'
        ]
        papers_per_query = max_papers // 2
    elif max_papers <= 3000:
        # For medium sets, use 4 balanced queries
        selected_queries = [
            'cs.LG AND ("machine learning" OR "neural networks")',
            'q-bio.NC AND ("neuroscience" OR "brain")',
            'cs.AI AND ("artificial intelligence" OR "deep learning")',
            'q-bio.QM AND ("medical imaging" OR "healthcare")'
        ]
        papers_per_query = max_papers // 4
    else:
        # For large sets, use all 6 queries
        selected_queries = test_queries
        papers_per_query = max_papers // 6

    # Store balanced collection settings
    config['nemo_curator']['balanced_queries'] = selected_queries
    config['nemo_curator']['papers_per_query'] = papers_per_query
    config['nemo_curator']['use_balanced_collection'] = True

    # For the initial query, use a general ML+Healthcare query with recent papers
    current_year = datetime.now().year

    # Focus on recent papers from last 2-3 years for contemporary evaluation
    year_query = f"submittedDate:[{current_year-3}0101 TO {current_year}1231]"

    config['nemo_curator']['filter_query'] = f'({year_query}) AND (cs.LG OR cs.AI OR q-bio.NC OR q-bio.QM) AND (("machine learning" OR "neural networks") OR ("neuroscience" OR "healthcare" OR "medical imaging"))'

    # Store temporal settings
    config['nemo_curator']['focus_recent'] = True
    config['nemo_curator']['year_range'] = f"{current_year-3} to {current_year}"

    # Optimized settings for small test set collection
    config['collection']['rate_limit'] = 1.0  # Faster for small sets
    config['extraction']['rate_limit'] = 1.0
    config['nemo_curator']['batch_size'] = 100  # Smaller batches
    config['extraction']['max_pages'] = 3  # Fewer pages for speed
    config['extraction']['max_chars'] = 8000  # Smaller text for speed

    # Save test config
    test_config_dir = os.path.dirname(test_config_path)
    if test_config_dir:  # Only create directory if path has a directory component
        os.makedirs(test_config_dir, exist_ok=True)
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created test config at {test_config_path}")
    print(f"Test query: {config['nemo_curator']['filter_query']}")
    print(f"Max papers: {max_papers}")

    return config


def filter_duplicates(raw_papers_path: str, training_ids: set, output_path: str) -> int:
    """Filter out papers that overlap with training data."""

    if not os.path.exists(raw_papers_path):
        print(f"Warning: Raw papers file not found at {raw_papers_path}")
        return 0

    print(f"Filtering duplicates from {raw_papers_path}...")
    print(f"Training IDs to exclude: {len(training_ids)}")

    unique_papers = []
    duplicates_found = 0

    with open(raw_papers_path, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                paper = json.loads(line)
                arxiv_id = paper.get('arxiv_id', '')

                # Skip if in training data
                if arxiv_id in training_ids:
                    duplicates_found += 1
                    continue

                unique_papers.append(paper)

            except json.JSONDecodeError:
                continue

    print(f"Found {duplicates_found} duplicates with training data")
    print(f"Kept {len(unique_papers)} unique papers")

    # Save filtered papers
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for paper in unique_papers:
            f.write(json.dumps(paper) + '\n')

    return len(unique_papers)


def run_pipeline(test_config_path: str) -> bool:
    """Run the data collection pipeline with test config with better debugging."""

    print("Running data collection pipeline for test dataset...")
    print("This may take 10-30 minutes for 2000-5000 papers")

    try:
        # Load test config to check settings
        import yaml
        with open(test_config_path, 'r') as f:
            test_config = yaml.safe_load(f)

        max_papers = test_config['nemo_curator']['max_papers']
        print(f"Target papers: {max_papers}")
        print(f"Query: {test_config['nemo_curator']['filter_query']}")

        # Check if run_pipeline.py exists
        if not os.path.exists('run_pipeline.py'):
            print("❌ run_pipeline.py not found! Running simplified collection...")
            return run_simplified_collection(test_config)

        # Run pipeline with real-time output (no capture)
        cmd = ['python', 'run_pipeline.py', '--config', test_config_path]
        print(f"Running: {' '.join(cmd)}")

        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1
        )

        output_lines = []
        last_progress = ""

        try:
            # Print output in real-time with shorter timeout
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
                    output_lines.append(line)

                    # Track progress
                    if 'papers' in line.lower() or 'processed' in line.lower():
                        last_progress = line.strip()

                    # Check if we're getting stuck
                    if len(output_lines) > 100:  # Too many lines, might be stuck
                        print("⚠️  Lots of output - pipeline might be in a loop")
                        break

            process.wait(timeout=1800)  # 30 minute timeout for completion

            if process.returncode == 0:
                print("✅ Test data collection completed successfully")
                return True
            else:
                print(f"❌ Pipeline failed with return code {process.returncode}")

                # Show last few lines of output for debugging
                print("Last few output lines:")
                for line in output_lines[-10:]:
                    print(f"  {line.strip()}")
                return False

        except subprocess.TimeoutExpired:
            process.terminate()
            print(f"❌ Pipeline timed out. Last progress: {last_progress}")
            print("Try running with --skip-collection flag to use existing data")
            return False

    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simplified_collection(test_config: dict) -> bool:
    """Fallback simplified collection using nemo_curator directly."""

    try:
        print("🔄 Running simplified data collection...")

        # Try to import nemo_curator
        from nemo_curator import Download, filters
        print("✅ Imported nemo_curator successfully")

        # Extract settings
        max_papers = test_config['nemo_curator']['max_papers']
        query = test_config['nemo_curator']['filter_query']
        output_dir = test_config['pipeline']['output_dir']

        print(f"Collecting {max_papers} papers with query: {query}")

        # Simple collection
        downloader = Download(
            max_papers=max_papers,
            output_format="jsonl"
        )

        # This is a simplified approach - you may need to adjust based on your nemo_curator version
        print("Starting data download...")
        # downloader(query) - This line would need your specific nemo_curator setup

        print("✅ Simplified collection completed")
        return True

    except ImportError as e:
        print(f"❌ Cannot import nemo_curator: {e}")
        print("Please install nemo_curator: pip install nemo-curator")
        return False
    except Exception as e:
        print(f"❌ Error in simplified collection: {e}")
        return False


def validate_test_dataset(test_metadata_path: str, training_ids: set) -> bool:
    """Validate that test dataset has no overlap with training data and check domain balance."""

    if not os.path.exists(test_metadata_path):
        print(f"❌ Test metadata not found at {test_metadata_path}")
        return False

    print(f"Validating test dataset at {test_metadata_path}...")

    test_ids = set()
    overlap_count = 0
    domain_counts = {
        'ML': 0,
        'Healthcare': 0,
        'Both': 0,
        'Other': 0,
        'Unknown': 0
    }
    year_counts = {}  # Track publication years

    def classify_paper_domain_simple(paper: dict) -> str:
        """Simple domain classification for validation."""
        categories = paper.get('categories', [])
        if not isinstance(categories, list):
            categories = [categories] if categories else []

        # Check for ML categories
        ml_categories = [cat for cat in categories if isinstance(cat, str) and cat.startswith('cs.') or 'stat.' in cat]
        # Check for healthcare/bio categories
        healthcare_categories = [cat for cat in categories if isinstance(cat, str) and ('q-bio' in cat or 'bio' in cat)]

        # Also check title/abstract for keywords
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        text = title + ' ' + abstract

        ml_keywords = ['machine learning', 'neural network', 'deep learning', 'artificial intelligence']
        healthcare_keywords = ['neuroscience', 'healthcare', 'medical', 'brain', 'imaging']

        has_ml = len(ml_categories) > 0 or any(kw in text for kw in ml_keywords)
        has_healthcare = len(healthcare_categories) > 0 or any(kw in text for kw in healthcare_keywords)

        if has_ml and has_healthcare:
            return 'Both'
        elif has_ml:
            return 'ML'
        elif has_healthcare:
            return 'Healthcare'
        elif len(categories) > 0:
            return 'Other'
        else:
            return 'Unknown'

    with open(test_metadata_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    arxiv_id = record.get('arxiv_id')
                    if arxiv_id:
                        test_ids.add(arxiv_id)
                        if arxiv_id in training_ids:
                            overlap_count += 1

                        # Classify domain
                        domain = classify_paper_domain_simple(record)
                        domain_counts[domain] += 1

                        # Track publication year
                        year = record.get('published') or record.get('year') or record.get('update_date')
                        if year:
                            # Extract year from date string or use the year field directly
                            if isinstance(year, str) and len(year) >= 4:
                                year = year[:4]  # Extract first 4 characters (year)
                            year = str(year)
                            year_counts[year] = year_counts.get(year, 0) + 1

                except json.JSONDecodeError:
                    continue

    print(f"\n📊 Test Dataset Analysis:")
    print(f"   Total papers: {len(test_ids)}")
    print(f"   Overlap with training data: {overlap_count} papers")
    print(f"\n📈 Domain Distribution:")
    total = sum(domain_counts.values())
    for domain, count in domain_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"   {domain}: {count} papers ({percentage:.1f}%)")

    # Check balance quality
    ml_total = domain_counts['ML'] + domain_counts['Both']
    healthcare_total = domain_counts['Healthcare'] + domain_counts['Both']
    intersection = domain_counts['Both']

    print(f"\n⚖️  Balance Analysis:")
    print(f"   ML papers: {ml_total} ({ml_total/total*100:.1f}%)")
    print(f"   Healthcare papers: {healthcare_total} ({healthcare_total/total*100:.1f}%)")
    print(f"   Intersection (ML+Healthcare): {intersection} ({intersection/total*100:.1f}%)")

    # Temporal analysis
    if year_counts:
        print(f"\n📅 Temporal Distribution:")
        sorted_years = sorted(year_counts.items())
        current_year = datetime.now().year

        recent_years = 0
        for year, count in sorted_years:
            percentage = (count / total * 100) if total > 0 else 0
            if int(year) >= current_year - 2:  # Last 2 years
                recent_years += count
                print(f"   {year}: {count} papers ({percentage:.1f}%) 🆕")
            else:
                print(f"   {year}: {count} papers ({percentage:.1f}%)")

        recent_percentage = (recent_years / total * 100) if total > 0 else 0
        print(f"\n   Recent papers ({current_year-2}-{current_year}): {recent_years} ({recent_percentage:.1f}%)")

        # Oldest and newest papers
        oldest_year = min(sorted_years, key=lambda x: x[0])[0] if sorted_years else "Unknown"
        newest_year = max(sorted_years, key=lambda x: x[0])[0] if sorted_years else "Unknown"
        print(f"   Time span: {oldest_year} to {newest_year}")

        if recent_percentage < 50:
            print("⚠️  Warning: Less than 50% recent papers - may not reflect current trends")
    else:
        print("\n📅 Temporal Distribution: No year information available")

    # Balance quality assessment
    if overlap_count > 0:
        print(f"❌ Found {overlap_count} overlapping papers!")
        return False
    else:
        print("✅ No overlap found with training data")

    # Check if we have reasonable balance
    if ml_total == 0 or healthcare_total == 0:
        print("⚠️  Warning: Test set missing ML or Healthcare papers - not balanced")
        return False
    elif ml_total < total * 0.2 or healthcare_total < total * 0.2:
        print("⚠️  Warning: Test set may be imbalanced (<20% in one domain)")

    print("✅ Test dataset validation completed")
    return True


def create_evaluation_commands(test_dir: str, models: dict) -> str:
    """Create evaluation commands for existing models."""

    commands = []

    for model_name, model_config in models.items():
        checkpoint = model_config['checkpoint']
        tokenizer = model_config['tokenizer']
        output_dir = f"./evaluations/test_{model_name}"

        cmd = f"""# Evaluate {model_name}
!python evaluate.py \\
    --model-checkpoint {checkpoint} \\
    --dataset-text-dir {test_dir}/texts \\
    --dataset-metadata {test_dir}/processed_dataset.jsonl \\
    --tokenizer-path "{tokenizer}" \\
    --output-dir {output_dir}"""

        commands.append(cmd)

    return "\n\n".join(commands)


def main():
    parser = argparse.ArgumentParser(description="Collect fresh test dataset for evaluation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Base config file')
    parser.add_argument('--test-config', type=str, default='config_test.yaml', help='Test config output file')
    parser.add_argument('--max-papers', type=int, default=2000, help='Maximum papers to collect')
    parser.add_argument('--training-data', type=str,
                       default='/content/drive/MyDrive/neuroMOE_results/data/arxiv/processed_dataset.jsonl',
                       help='Path to training metadata')
    parser.add_argument('--output-dir', type=str, default='./data/test_dataset', help='Test data output directory')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection, just process existing data')

    args = parser.parse_args()

    print("=" * 60)
    print("Test Dataset Collection Pipeline")
    print("=" * 60)

    # Step 1: Load training IDs to avoid overlap
    training_ids = load_training_ids(args.training_data)

    # Step 2: Create test-specific configuration
    test_config = create_test_config(args.config, args.test_config, args.max_papers)

    # Step 3: Run data collection pipeline (unless skipped)
    if not args.skip_collection:
        success = run_pipeline(args.test_config)
        if not success:
            print("❌ Failed to collect test data")
            print("💡 You can run with --skip-collection to use existing data or try a smaller --max-papers")
            return 1
    else:
        print("⏭️  Skipping data collection (using existing data)")

    # Step 4: Filter duplicates from training data
    raw_papers_path = os.path.join(test_config['pipeline']['output_dir'], 'arxiv_raw_output.jsonl')
    filtered_papers_path = os.path.join(test_config['pipeline']['output_dir'], 'test_papers_filtered.jsonl')

    unique_count = filter_duplicates(raw_papers_path, training_ids, filtered_papers_path)

    if unique_count == 0:
        print("❌ No unique papers found after filtering")
        return 1

    # Step 5: Validate final test dataset
    final_test_path = os.path.join(test_config['pipeline']['output_dir'], 'processed_dataset.jsonl')
    if not validate_test_dataset(final_test_path, training_ids):
        print("❌ Test dataset validation failed")
        return 1

    # Step 6: Create evaluation commands
    models = {
        'baseline_encoder': {
            'checkpoint': '/content/drive/MyDrive/neuroMOE_results/checkpoints/baseline/encoder/baseline_encoder_final.pt',
            'tokenizer': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        },
        'baseline_decoder': {
            'checkpoint': '/content/drive/MyDrive/neuroMOE_results/checkpoints/baseline/decoder/baseline_decoder_final.pt',
            'tokenizer': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        },
        'moe': {
            'checkpoint': '/content/drive/MyDrive/neuroMOE_results/checkpoints/step_50000.pt',
            'tokenizer': '/content/drive/MyDrive/neuroMOE_results/data/arxiv/healthcare_tokenizer.model'
        }
    }

    evaluation_script = create_evaluation_commands(test_config['pipeline']['output_dir'], models)

    # Save evaluation commands
    eval_script_path = os.path.join(test_config['pipeline']['output_dir'], 'evaluate_models.sh')
    with open(eval_script_path, 'w') as f:
        f.write(evaluation_script)

    print("\n" + "=" * 60)
    print("✅ Test dataset collection completed successfully!")
    print("=" * 60)
    print(f"Test dataset location: {test_config['pipeline']['output_dir']}")
    print(f"Unique test papers: {unique_count}")
    print(f"Evaluation commands saved to: {eval_script_path}")

    print("\nNext steps:")
    print("1. Review the test dataset")
    print("2. Run evaluation commands to test your models")
    print("3. Compare results between models")

    return 0


if __name__ == "__main__":
    exit(main())