#!/usr/bin/env python3
"""
Quick script to update domain classification in existing processed_dataset.jsonl
from generic domains to the new 3-domain system.
"""

import json
import sys
from pathlib import Path

def classify_healthcare_domains_six(paper: dict) -> list:
    """6-domain classification for healthcare expert specialization."""

    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    text = (paper.get('text', '') or '').lower()

    # Use first 2000 chars for keyword matching to save memory
    combined_text = (title + ' ' + abstract + ' ' + text[:2000]).lower()

    # Specific 6-domain keywords for healthcare expert specialization
    domain_keywords = {
        'neurodegeneration': [
            'alzheimer', 'parkinson', 'dementia', 'neurodegenerative', 'cognitive decline',
            'memory loss', 'neurodegeneration', 'amyloid', 'tau protein', 'lewy body',
            'frontotemporal dementia', 'cognitive impairment', 'brain atrophy', 'mild cognitive impairment'
        ],
        'neuroscience': [
            'neuron', 'neural', 'brain', 'cortical', 'synapse', 'synaptic', 'neurotransmitter',
            'dopamine', 'serotonin', 'gaba', 'glutamate', 'neuroscience', 'cognitive',
            'motor cortex', 'prefrontal', 'hippocampus', 'cerebellum', 'brain imaging',
            'fmri', 'eeg', 'neural activity', 'brain function', 'neural circuit', 'neuroimaging'
        ],
        'medical_imaging': [
            'mri', 'ct scan', 'pet scan', 'ultrasound', 'x-ray', 'radiology', 'imaging',
            'medical image', 'scan', 'tomography', 'mammography', 'angiography', 'fluoroscopy',
            'medical imaging', 'image analysis', 'computer vision', 'segmentation',
            'image registration', 'dicom', 'pixel', 'radiograph', 'medical image processing'
        ],
        'clinical': [
            'patient', 'clinical trial', 'treatment', 'therapy', 'diagnosis', 'symptom',
            'hospital', 'physician', 'medical', 'clinical', 'patient care', 'therapeutic',
            'medical treatment', 'clinical study', 'intervention', 'prognosis', 'diagnostic',
            'medical procedure', 'clinical outcome', 'patient outcome', 'clinical practice'
        ],
        'drug_discovery': [
            'drug', 'pharmaceutical', 'medication', 'compound', 'drug discovery', 'clinical trial',
            'fda approval', 'drug development', 'pharmacology', 'drug target', 'lead compound',
            'drug screening', 'medicinal chemistry', 'pharmacokinetic', 'pharmacodynamic',
            'bioavailability', 'drug interaction', 'adverse drug reaction', 'drug design'
        ],
        'general_ml_health': [
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'algorithm', 'model', 'prediction', 'classification', 'regression', 'clustering',
            'data mining', 'feature extraction', 'training', 'validation', 'cross-validation',
            'supervised learning', 'unsupervised learning', 'reinforcement learning', 'healthcare ai'
        ]
    }

    domains = []

    # Keyword-based classification
    for domain, keywords in domain_keywords.items():
        keyword_count = sum(1 for keyword in keywords if keyword in combined_text)
        if keyword_count >= 1:  # Lower threshold for specific domains
            domains.append(domain)

    # Fallback classification
    if not domains:
        healthcare_indicators = ['patient', 'medical', 'clinical', 'health', 'disease', 'treatment']
        ml_indicators = ['machine learning', 'neural network', 'deep learning', 'algorithm', 'model', 'prediction']

        has_healthcare = any(indicator in combined_text for indicator in healthcare_indicators)
        has_ml = any(indicator in combined_text for indicator in ml_indicators)

        if has_healthcare and has_ml:
            domains.append('general_ml_health')
        elif has_healthcare:
            # Try to be more specific with healthcare content
            if any(term in combined_text for term in ['alzheimer', 'parkinson', 'dementia', 'neurodegenerative']):
                domains.append('neurodegeneration')
            elif any(term in combined_text for term in ['brain', 'neural', 'cognitive', 'fmri', 'eeg']):
                domains.append('neuroscience')
            elif any(term in combined_text for term in ['imaging', 'scan', 'radiology', 'mri', 'ct']):
                domains.append('medical_imaging')
            elif any(term in combined_text for term in ['drug', 'pharmaceutical', 'medication']):
                domains.append('drug_discovery')
            else:
                domains.append('clinical')
        else:
            domains.append('general_ml_health')

    return domains

def update_domain_classification(input_file: str, output_file: str = None):
    """Update domain classification in processed_dataset.jsonl."""

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return False

    if output_file is None:
        output_file = str(input_path).replace('.jsonl', '_updated_domains.jsonl')

    output_path = Path(output_file)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    domain_counts = {
        'neurodegeneration': 0, 'neuroscience': 0, 'medical_imaging': 0,
        'clinical': 0, 'drug_discovery': 0, 'general_ml_health': 0
    }
    papers_updated = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in):
                if not line.strip():
                    continue

                try:
                    paper = json.loads(line)

                    # Update domain classification
                    old_domains = paper.get('domains', [])
                    new_domains = classify_healthcare_domains_six(paper)

                    paper['domains'] = new_domains
                    # Check for neurodegeneration content
                    paper['has_neurodegeneration'] = 'neurodegeneration' in new_domains

                    # Count domains
                    for domain in new_domains:
                        if domain in domain_counts:
                            domain_counts[domain] += 1

                    papers_updated += 1

                    # Write updated paper
                    f_out.write(json.dumps(paper, ensure_ascii=False) + '\n')

                    # Progress indicator
                    if papers_updated % 1000 == 0:
                        print(f"Processed {papers_updated} papers...", flush=True)

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num + 1} due to JSON error: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num + 1}: {e}")
                    continue

    except Exception as e:
        print(f"Error: {e}")
        return False

    print(f"\n✅ Domain classification updated successfully!")
    print(f"📊 Processed {papers_updated} papers")
    print(f"📈 Domain distribution:")
    for domain, count in domain_counts.items():
        percentage = (count / papers_updated * 100) if papers_updated > 0 else 0
        print(f"   {domain}: {count} ({percentage:.1f}%)")

    return True

def main():
    """Main function."""
    # Default paths for Drive usage
    default_input = "/content/drive/MyDrive/neuroMOE_results/data/arxiv/processed_dataset.jsonl"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input

    print("🔄 Updating Domain Classification to 6-Domain System")
    print("=" * 60)

    success = update_domain_classification(input_file)

    if success:
        print("\n🎉 Domain classification complete!")
        print("\nTo use the updated dataset:")
        print("1. Replace the original processed_dataset.jsonl with the new file")
        print("2. Or update your training script to point to the new file")
    else:
        print("\n❌ Domain classification failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()