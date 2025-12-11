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

    # Pick a different query from what was used for training
    config['nemo_curator']['filter_query'] = random.choice(test_queries)

    # More conservative settings for test data
    config['collection']['rate_limit'] = 0.5  # Slower requests
    config['extraction']['rate_limit'] = 0.5

    # Save test config
    os.makedirs(os.path.dirname(test_config_path), exist_ok=True)
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
    """Run the data collection pipeline with test config."""

    print("Running data collection pipeline for test dataset...")

    try:
        # Use run_pipeline.py with the test config
        cmd = ['python', 'run_pipeline.py', '--config', test_config_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

        if result.returncode == 0:
            print("✅ Test data collection completed successfully")
            return True
        else:
            print(f"❌ Pipeline failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Pipeline timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False


def validate_test_dataset(test_metadata_path: str, training_ids: set) -> bool:
    """Validate that test dataset has no overlap with training data."""

    if not os.path.exists(test_metadata_path):
        print(f"❌ Test metadata not found at {test_metadata_path}")
        return False

    print(f"Validating test dataset at {test_metadata_path}...")

    test_ids = set()
    overlap_count = 0

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
                except json.JSONDecodeError:
                    continue

    print(f"Test dataset size: {len(test_ids)} papers")
    print(f"Overlap with training data: {overlap_count} papers")

    if overlap_count > 0:
        print(f"❌ Found {overlap_count} overlapping papers!")
        return False
    else:
        print("✅ No overlap found with training data")
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
            return 1

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