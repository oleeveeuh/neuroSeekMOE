#!/usr/bin/env python3
"""
Complete Pipeline Orchestration Script

Runs the full NeMo Curator + training pipeline:
1. Collect ArXiv papers
2. Extract PDF texts
3. NeMo Curator curation (cleaning, filtering, deduplication)
4. Healthcare-specific processing
5. Train tokenizer
6. Train model
7. Evaluate model
8. Export inference pipeline

Features:
- Resume from any step (detects completed steps)
- Configurable via config.yaml
- Cleanup intermediate files
- Comprehensive logging and reporting
- Error handling with checkpoints

Usage:
    python run_pipeline.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Import pipeline components
from data_pipeline import (
    collect_arxiv_papers,
    extract_pdf_texts,
    curate_with_nemo,
    run_nemo_curator_pipeline,  # New NeMo Curator Pipeline API
    process_curated_dataset,
    train_healthcare_tokenizer
)

try:
    from train_colab import train, find_latest_checkpoint
    TRAIN_COLAB_AVAILABLE = True
except ImportError:
    TRAIN_COLAB_AVAILABLE = False
    print("train_colab.py not available")

try:
    from evaluate import evaluate_model
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    print("evaluate.py not available")

try:
    from inference import InferencePipeline
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("inference.py not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete pipeline from paper collection to inference export."""
    
    def __init__(self, config_path: str):
        """Initialize orchestrator with configuration.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['pipeline']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.step_status = {}
        self.start_time = time.time()
        self.step_times = {}
        
        # File paths
        self.metadata_jsonl = self.output_dir / "arxiv_papers.jsonl"
        self.text_dir = self.output_dir / "texts"
        self.curated_jsonl = self.output_dir / "curated_dataset.jsonl"
        self.processed_jsonl = self.output_dir / "processed_dataset.jsonl"
        self.tokenizer_model = self.output_dir / f"{self.config['tokenizer']['model_prefix']}.model"
        self.tokenizer_vocab = self.output_dir / f"{self.config['tokenizer']['model_prefix']}.vocab"
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.eval_dir = Path(self.config['evaluation']['output_dir'])
        self.inference_dir = Path(self.config['inference']['output_dir'])
        
        logger.info("=" * 80)
        logger.info("Pipeline Orchestrator Initialized")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Config file: {config_path}")
        logger.info(f"Resume mode: {self.config['pipeline']['resume']}")
        logger.info(f"🧹 Cleanup intermediate: {self.config['pipeline']['cleanup_intermediate']}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _check_step_complete(self, step_name: str, output_files: List[Path], check_non_empty: bool = True) -> bool:
        """Check if a step is complete by verifying output files exist and are non-empty.
        
        Args:
            step_name: Name of the step
            output_files: List of output file paths to check
            check_non_empty: If True, also verify files are non-empty (default: True)
            
        Returns:
            True if all output files exist (and are non-empty if check_non_empty=True), False otherwise
        """
        if not self.config['pipeline']['resume']:
            return False
        
        all_valid = True
        for f in output_files:
            if not f.exists():
                all_valid = False
                break
            
            # Check if file is non-empty
            if check_non_empty:
                if f.is_file():
                    # For files, check if they have content
                    try:
                        if f.stat().st_size == 0:
                            logger.info(f"Step '{step_name}' output file exists but is empty: {f}")
                            all_valid = False
                            break
                        # For JSONL files, check if they have at least one valid line
                        if f.suffix == '.jsonl':
                            with open(f, 'r', encoding='utf-8') as file_handle:
                                has_content = any(line.strip() for line in file_handle)
                                if not has_content:
                                    logger.info(f"Step '{step_name}' JSONL file exists but has no valid lines: {f}")
                                    all_valid = False
                                    break
                    except Exception as e:
                        logger.warning(f"Could not check file {f}: {e}")
                        all_valid = False
                        break
                elif f.is_dir():
                    # For directories, check if they have at least one file
                    try:
                        if not any(f.iterdir()):
                            logger.info(f"Step '{step_name}' output directory exists but is empty: {f}")
                            all_valid = False
                            break
                    except Exception as e:
                        logger.warning(f"Could not check directory {f}: {e}")
                        all_valid = False
                        break
        
        if all_valid:
            logger.info(f"Step '{step_name}' already complete (output files exist and are valid)")
        return all_valid
    
    def _log_step_start(self, step_name: str, step_num: int, total_steps: int):
        """Log step start.
        
        Args:
            step_name: Name of the step
            step_num: Step number (1-indexed)
            total_steps: Total number of steps
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Step {step_num}/{total_steps}: {step_name}")
        logger.info("=" * 80)
        self.step_times[step_name] = time.time()
    
    def _log_step_end(self, step_name: str, success: bool = True):
        """Log step completion.
        
        Args:
            step_name: Name of the step
            success: Whether step completed successfully
        """
        elapsed = time.time() - self.step_times.get(step_name, time.time())
        status = "" if success else ""
        logger.info(f"{status} Step '{step_name}' completed in {elapsed:.2f}s")
        self.step_status[step_name] = {
            'success': success,
            'elapsed': elapsed,
            'timestamp': datetime.now().isoformat()
        }
    
    def step1_collect_papers(self) -> bool:
        """Step 1: Collect ArXiv papers.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Collect Papers"
        self._log_step_start(step_name, 1, 8)
        
        # Check if already complete - must have reached target count
        collection_config = self.config['collection']
        target_papers = collection_config['max_papers']
        
        if self.metadata_jsonl.exists():
            count = sum(1 for line in open(self.metadata_jsonl) if line.strip())
            if count >= target_papers:
                logger.info(f"Found {count} papers (target: {target_papers}) - collection already complete")
                self._log_step_end(step_name, True)
                return True
            elif count > 0:
                logger.info(f"Found {count} papers (target: {target_papers}) - continuing collection...")
                # Continue to collection below
            else:
                logger.warning(f"Metadata file exists but is empty. Re-running collection...")
        
        try:
            # If file exists but is empty, delete it to force re-collection
            if self.metadata_jsonl.exists():
                count = sum(1 for line in open(self.metadata_jsonl) if line.strip())
                if count == 0:
                    logger.warning(f"Existing metadata file is empty. Deleting to force re-collection...")
                    self.metadata_jsonl.unlink()
            
            collection_config = self.config['collection']
            rate_limit_delay = 1.0 / collection_config['rate_limit']
            
            logger.info(f"Starting ArXiv paper collection...")
            logger.info(f"   Target: {collection_config['max_papers']} papers")
            logger.info(f"   Rate limit: {collection_config['rate_limit']} requests/sec")
            
            # Get batch collection parameters
            batch_size = collection_config.get('batch_size', 10)
            ram_target = collection_config.get('ram_target', 50.0)
            logger.info(f"   Batch size: {batch_size} papers/batch")
            logger.info(f"   RAM target: <{ram_target}%")
            
            collect_arxiv_papers(
                output_dir=str(self.output_dir),
                max_papers=collection_config['max_papers'],
                cache_file=collection_config.get('cache_file'),
                rate_limit_delay=rate_limit_delay,
                batch_size=batch_size,
                ram_target=ram_target
            )
            
            if not self.metadata_jsonl.exists():
                raise FileNotFoundError(f"Metadata file not created: {self.metadata_jsonl}")
            
            # Count collected papers
            count = sum(1 for line in open(self.metadata_jsonl) if line.strip())
            logger.info(f"Collected {count} papers")
            
            if count == 0:
                error_msg = (
                    "Error: 0 papers collected. This indicates a problem:\n"
                    "   1. ArXiv API issues or rate limiting\n"
                    "   2. Network connectivity problems\n"
                    "   3. Query parameters too restrictive\n"
                    "   4. Collection function returned without collecting\n"
                    "   Check the collection logs above for detailed error messages."
                )
                logger.error(error_msg)
                raise RuntimeError("Paper collection failed: 0 papers collected")
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"Step 1 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step2_extract_pdfs(self) -> bool:
        """Step 2: Extract PDF texts.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Extract PDFs"
        self._log_step_start(step_name, 2, 8)
        
        # Check if already complete (verify directory has files)
        if self._check_step_complete(step_name, [self.text_dir], check_non_empty=True):
            # Double-check: count text files
            if self.text_dir.exists():
                text_files = list(self.text_dir.glob("*.txt"))
                if len(text_files) > 0:
                    logger.info(f"Found {len(text_files)} text files in existing directory")
                    self._log_step_end(step_name, True)
                    return True
                else:
                    logger.warning(f"Text directory exists but has no .txt files. Re-running extraction...")
            else:
                logger.warning(f"Text directory check passed but doesn't exist. Re-running extraction...")
        
        try:
            if not self.metadata_jsonl.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_jsonl}")
            
            extraction_config = self.config['extraction']
            
            # Map config keys to function parameters
            num_workers = extraction_config.get('workers', extraction_config.get('num_workers', 2))
            rate_limit_delay = extraction_config.get('rate_limit', extraction_config.get('rate_limit_delay', 0.4))
            
            logger.info(f"Starting PDF extraction...")
            logger.info(f"   Workers: {num_workers}")
            logger.info(f"   Rate limit: {1.0/rate_limit_delay:.1f} requests/sec")
            
            extract_pdf_texts(
                input_jsonl=str(self.metadata_jsonl),
                output_dir=str(self.text_dir),
                num_workers=num_workers,
                rate_limit_delay=rate_limit_delay
            )
            
            if not self.text_dir.exists():
                raise FileNotFoundError(f"Text directory not created: {self.text_dir}")
            
            # Count extracted texts
            text_files = list(self.text_dir.glob("*.txt"))
            logger.info(f"Extracted {len(text_files)} text files")
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step3_nemo_curator(self) -> bool:
        """Step 3: NeMo Curator curation.
        
        Uses the new NeMo Curator Pipeline API with custom healthcare stages:
        - ArxivDownloadExtractStage: Downloads from S3
        - HealthcareFilterStage: Text cleaning and domain classification
        - HealthcareQualityFilterStage: Deduplication and quality checks
        - HealthcareJsonlWriter: Formats and writes output
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "NeMo Curator Curation"
        
        # Force immediate output - CRITICAL for Colab visibility
        import sys
        print(f"\n{'='*80}", flush=True)
        print(f"Step 3: {step_name}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"⏰ Starting at: {datetime.now().isoformat()}", flush=True)
        sys.stdout.flush()
        
        try:
            self._log_step_start(step_name, 3, 8)
        except Exception as e:
            print(f"Error in _log_step_start: {e}", flush=True)
        
        # Check if already complete (verify file is non-empty)
        print(f"Checking if step already complete...", flush=True)
        try:
            is_complete = self._check_step_complete(step_name, [self.curated_jsonl], check_non_empty=True)
            print(f"   Step complete check: {is_complete}", flush=True)
        except Exception as e:
            print(f"Error checking step completion: {e}", flush=True)
            is_complete = False
        
        if is_complete:
            # Double-check: count curated papers
            print(f"   Checking existing curated file...", flush=True)
            if self.curated_jsonl.exists():
                try:
                    count = sum(1 for line in open(self.curated_jsonl) if line.strip())
                    print(f"   Found {count} papers in existing file", flush=True)
                    if count > 0:
                        print(f"Step already complete with {count} papers", flush=True)
                        logger.info(f"Found {count} papers in existing curated dataset")
                        self._log_step_end(step_name, True)
                        return True
                    else:
                        print(f"Curated file exists but is empty. Re-running curation...", flush=True)
                        logger.warning(f"Curated file exists but is empty. Re-running curation...")
                except Exception as e:
                    print(f"Error reading curated file: {e}", flush=True)
            else:
                print(f"Curated file check passed but doesn't exist. Re-running curation...", flush=True)
                logger.warning(f"Curated file check passed but doesn't exist. Re-running curation...")
        
        print(f"Starting NeMo Curator curation...", flush=True)
        sys.stdout.flush()
        
        try:
            print(f"Loading NeMo Curator config from config.yaml...", flush=True)
            nemo_config = self.config.get('nemo_curator', {})
            if not nemo_config:
                print(f"Warning: 'nemo_curator' section not found in config, using defaults", flush=True)
            print(f"   Config loaded: {list(nemo_config.keys())}", flush=True)
            sys.stdout.flush()
            
            # Check if text files already exist (from extraction step)
            # If so, use legacy function to process existing files
            # Otherwise, use Pipeline API to download from scratch
            text_files_exist = self.text_dir.exists() and any(self.text_dir.glob("*.txt"))
            
            print(f"\nDebug: Checking NeMo Curator step...", flush=True)
            print(f"   Text files exist: {text_files_exist}", flush=True)
            print(f"   Text directory: {self.text_dir}", flush=True)
            print(f"   Metadata file: {self.metadata_jsonl}", flush=True)
            print(f"   Output file: {self.curated_jsonl}", flush=True)
            sys.stdout.flush()
            
            if text_files_exist:
                # Use legacy curate_with_nemo function (processes existing text files)
                print("Using NeMo Curator to process existing extracted text files", flush=True)
                logger.info("Using NeMo Curator to process existing extracted text files")
                print(f"   Found text files in: {self.text_dir}", flush=True)
                logger.info(f"   Found text files in: {self.text_dir}")
                
                # Check if NeMo Curator is available
                print(f"   Checking NeMo Curator availability...", flush=True)
                try:
                    from data_pipeline import NEMO_CURATOR_AVAILABLE
                    print(f"   NeMo Curator available: {NEMO_CURATOR_AVAILABLE}", flush=True)
                except Exception as e:
                    print(f"Error importing NEMO_CURATOR_AVAILABLE: {e}", flush=True)
                    raise
                
                if not NEMO_CURATOR_AVAILABLE:
                    error_msg = (
                        "NeMo Curator not available.\n"
                        "   Install with: pip install 'nemo-curator[text]' or 'nemo-curator[text_cuda12]'\n"
                        "   Note: NeMo Curator only supports Linux systems\n"
                        "   On non-Linux systems, you can skip this step and use the preprocess command instead"
                    )
                    print(error_msg, flush=True)
                    logger.error(error_msg)
                    sys.stdout.flush()
                    raise RuntimeError("NeMo Curator not available")
                
                if not self.metadata_jsonl.exists():
                    error_msg = f"Metadata file not found: {self.metadata_jsonl}"
                    print(f"{error_msg}", flush=True)
                    logger.error(error_msg)
                    sys.stdout.flush()
                    raise FileNotFoundError(error_msg)
                
                # Count text files
                text_file_count = len(list(self.text_dir.glob("*.txt")))
                print(f"   Processing {text_file_count} text files", flush=True)
                logger.info(f"   Processing {text_file_count} text files")
                
                print(f"\nCalling curate_with_nemo()...", flush=True)
                sys.stdout.flush()
                try:
                    curate_with_nemo(
                        text_dir=str(self.text_dir),
                        metadata_jsonl=str(self.metadata_jsonl),
                        output_jsonl=str(self.curated_jsonl),
                        use_gpu=nemo_config.get('use_gpu', False),
                        skip_dedup=nemo_config.get('skip_dedup', False),
                        min_relevance_score=nemo_config.get('min_relevance_score', 0.5)
                    )
                    print(f"curate_with_nemo() completed", flush=True)
                    sys.stdout.flush()
                except Exception as e:
                    print(f"curate_with_nemo() failed with error: {e}", flush=True)
                    import traceback
                    print(f"   Full traceback:\n{traceback.format_exc()}", flush=True)
                    sys.stdout.flush()
                    raise
                
                # Verify curate_with_nemo actually created output (it returns None on error)
                if not self.curated_jsonl.exists():
                    error_msg = (
                        "NeMo Curator curation completed but no output file was created.\n"
                        "   This may indicate NeMo Curator is not properly installed or configured."
                    )
                    print(f"{error_msg}", flush=True)
                    logger.error(error_msg)
                    sys.stdout.flush()
                    raise RuntimeError(error_msg)
                else:
                    print(f"Output file created: {self.curated_jsonl}", flush=True)
                    sys.stdout.flush()
                
            else:
                # Use new NeMo Curator Pipeline API to download from ArXiv
                use_pipeline_api = nemo_config.get('use_pipeline_api', True)
                print(f"   use_pipeline_api: {use_pipeline_api}", flush=True)
                sys.stdout.flush()
                
                if use_pipeline_api:
                    print("Using NeMo Curator Pipeline API with FREE download_arxiv()", flush=True)
                    logger.info("Using NeMo Curator Pipeline API with FREE download_arxiv()")
                    print("   No existing text files found - will download from ArXiv", flush=True)
                    logger.info("   No existing text files found - will download from ArXiv")
                    sys.stdout.flush()
                    
                    raw_data_path = nemo_config.get('raw_data_path', str(self.output_dir / "arxiv_raw_data"))
                    raw_output_path = nemo_config.get('raw_output_path', str(self.output_dir / "arxiv_raw_output.jsonl"))
                    filter_query = nemo_config.get('filter_query', "cs.LG OR cs.AI OR q-bio.NC")
                    max_workers = nemo_config.get('max_workers', 1)  # Colab safe
                    use_gpu = nemo_config.get('use_gpu', False)
                    max_papers = nemo_config.get('max_papers', 40000)
                    batch_size = nemo_config.get('batch_size', 5000)
                    checkpoint_interval = nemo_config.get('checkpoint_interval', 1000)
                    resume = nemo_config.get('resume', True)
                    
                    print(f"\nCalling run_nemo_curator_pipeline()...", flush=True)
                    print(f"   Output path: {self.curated_jsonl}", flush=True)
                    print(f"   Raw data path: {raw_data_path}", flush=True)
                    print(f"   Filter query: {filter_query}", flush=True)
                    print(f"   Max workers: {max_workers}", flush=True)
                    print(f"   Max papers: {max_papers}", flush=True)
                    sys.stdout.flush()
                    
                    try:
                        result = run_nemo_curator_pipeline(
                            output_path=str(self.curated_jsonl),
                            raw_data_path=raw_data_path,
                            raw_output_path=raw_output_path,
                            filter_query=filter_query,
                            max_workers=max_workers,
                            use_gpu=use_gpu,
                            batch_size=batch_size,
                            checkpoint_interval=checkpoint_interval,
                            max_papers=max_papers,
                            resume=resume
                        )
                        print(f"run_nemo_curator_pipeline() returned: {result}", flush=True)
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"run_nemo_curator_pipeline() failed with error: {e}", flush=True)
                        import traceback
                        print(f"   Full traceback:\n{traceback.format_exc()}", flush=True)
                        sys.stdout.flush()
                        raise
                    
                    if result is None:
                        error_msg = "NeMo Curator pipeline failed (returned None)"
                        print(f"{error_msg}", flush=True)
                        logger.error(error_msg)
                        sys.stdout.flush()
                        raise RuntimeError(error_msg)
                else:
                    raise FileNotFoundError(
                        f"Text directory not found: {self.text_dir}\n"
                        f"   Either run extraction step first, or enable use_pipeline_api to download from ArXiv"
                    )
            
            if not self.curated_jsonl.exists():
                error_msg = f"Curated dataset not created: {self.curated_jsonl}"
                print(f"{error_msg}", flush=True)
                logger.error(error_msg)
                sys.stdout.flush()
                raise FileNotFoundError(error_msg)
            
            # Count curated papers
            print(f"Counting curated papers...", flush=True)
            try:
                count = sum(1 for _ in open(self.curated_jsonl))
                print(f"Curated {count} papers", flush=True)
                logger.info(f"Curated {count} papers")
            except Exception as e:
                print(f"Error counting papers: {e}", flush=True)
                count = 0
            
            print(f"Step 3 completed successfully!", flush=True)
            sys.stdout.flush()
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            import sys
            import traceback
            
            error_msg = f"Step 3 failed: {e}"
            print(f"\n{'='*80}", flush=True)
            print(f"{error_msg}", flush=True)
            print(f"{'='*80}", flush=True)
            
            full_traceback = traceback.format_exc()
            print(f"\nFull error traceback:", flush=True)
            print(full_traceback, flush=True)
            
            logger.error(error_msg, exc_info=True)
            sys.stdout.flush()
            
            self._log_step_end(step_name, False)
            return False
    
    def step4_process_curated(self) -> bool:
        """Step 4: Healthcare-specific processing.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Process Curated Dataset"
        self._log_step_start(step_name, 4, 8)
        
        # Check if already complete (verify file is non-empty)
        if self._check_step_complete(step_name, [self.processed_jsonl], check_non_empty=True):
            # Double-check: count processed papers
            if self.processed_jsonl.exists():
                count = sum(1 for line in open(self.processed_jsonl) if line.strip())
                if count > 0:
                    logger.info(f"Found {count} papers in existing processed dataset")
                    self._log_step_end(step_name, True)
                    return True
                else:
                    logger.warning(f"Processed file exists but is empty. Re-running processing...")
            else:
                logger.warning(f"Processed file check passed but doesn't exist. Re-running processing...")
        
        try:
            if not self.curated_jsonl.exists():
                raise FileNotFoundError(f"Curated dataset not found: {self.curated_jsonl}")
            
            # Get processing config (check both 'processing' and 'preprocessing' keys)
            processing_config = self.config.get('processing', self.config.get('preprocessing', {}))
            
            num_workers = processing_config.get('workers', processing_config.get('num_workers', 4))
            
            logger.info(f"Starting healthcare-specific processing...")
            logger.info(f"   Input: {self.curated_jsonl}")
            logger.info(f"   Output: {self.processed_jsonl}")
            logger.info(f"   Workers: {num_workers}")
            
            process_curated_dataset(
                input_jsonl=str(self.curated_jsonl),
                output_jsonl=str(self.processed_jsonl),
                num_workers=num_workers
            )
            
            if not self.processed_jsonl.exists():
                raise FileNotFoundError(f"Processed dataset not created: {self.processed_jsonl}")
            
            # Count processed papers
            count = sum(1 for _ in open(self.processed_jsonl))
            logger.info(f"Processed {count} papers")
            
            # Cleanup intermediate files if requested (but keep text_dir for training)
            if self.config['pipeline']['cleanup_intermediate']:
                if self.curated_jsonl.exists() and self.processed_jsonl.exists():
                    logger.info(f"🧹 Cleaning up intermediate file: {self.curated_jsonl}")
                    self.curated_jsonl.unlink()
                # Note: text_dir will be cleaned up after training (step 6)
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"Step 4 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step5_train_tokenizer(self) -> bool:
        """Step 5: Train SentencePiece tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Train Tokenizer"
        self._log_step_start(step_name, 5, 8)
        
        # Check if already complete (verify files are non-empty)
        if self._check_step_complete(step_name, [self.tokenizer_model, self.tokenizer_vocab], check_non_empty=True):
            # Double-check: verify tokenizer files are valid
            if self.tokenizer_model.exists() and self.tokenizer_vocab.exists():
                if self.tokenizer_model.stat().st_size > 0 and self.tokenizer_vocab.stat().st_size > 0:
                    logger.info(f"Found existing tokenizer files")
                    self._log_step_end(step_name, True)
                    return True
                else:
                    logger.warning(f"Tokenizer files exist but are empty. Re-running tokenizer training...")
            else:
                logger.warning(f"Tokenizer file check passed but files don't exist. Re-running tokenizer training...")
        
        try:
            import sys
            print(f"\n{'='*80}", flush=True)
            print(f"Step 5: {step_name}", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"⏰ Starting at: {datetime.now().isoformat()}", flush=True)
            sys.stdout.flush()
            
            if not self.processed_jsonl.exists():
                error_msg = f"Processed dataset not found: {self.processed_jsonl}"
                print(f"{error_msg}", flush=True)
                raise FileNotFoundError(error_msg)
            
            print(f"Input file: {self.processed_jsonl}", flush=True)
            print(f"Output directory: {self.output_dir}", flush=True)
            
            # Check file size
            file_size = self.processed_jsonl.stat().st_size
            print(f"Input file size: {file_size:,} bytes", flush=True)
            
            if file_size == 0:
                error_msg = f"Input file is empty: {self.processed_jsonl}"
                print(f"{error_msg}", flush=True)
                raise ValueError(error_msg)
            
            tokenizer_config = self.config['tokenizer']
            print(f"Tokenizer config:", flush=True)
            print(f"   Model prefix: {tokenizer_config['model_prefix']}", flush=True)
            print(f"   Vocab size: {tokenizer_config['vocab_size']}", flush=True)
            sys.stdout.flush()
            
            print(f"\nCalling train_healthcare_tokenizer()...", flush=True)
            sys.stdout.flush()
            
            train_healthcare_tokenizer(
                input_jsonl=str(self.processed_jsonl),
                output_dir=str(self.output_dir),
                model_prefix=tokenizer_config['model_prefix'],
                vocab_size=tokenizer_config['vocab_size']
            )
            
            print(f"\nVerifying tokenizer files were created...", flush=True)
            if not self.tokenizer_model.exists():
                error_msg = f"Tokenizer model file not created: {self.tokenizer_model}"
                print(f"{error_msg}", flush=True)
                raise FileNotFoundError(error_msg)
            
            if not self.tokenizer_vocab.exists():
                error_msg = f"Tokenizer vocab file not created: {self.tokenizer_vocab}"
                print(f"{error_msg}", flush=True)
                raise FileNotFoundError(error_msg)
            
            model_size = self.tokenizer_model.stat().st_size
            vocab_size = self.tokenizer_vocab.stat().st_size
            print(f"Tokenizer files created:", flush=True)
            print(f"   Model: {self.tokenizer_model} ({model_size:,} bytes)", flush=True)
            print(f"   Vocab: {self.tokenizer_vocab} ({vocab_size:,} bytes)", flush=True)
            
            logger.info(f"Tokenizer trained: {self.tokenizer_model}")
            
            print(f"Step 5 completed successfully!", flush=True)
            sys.stdout.flush()
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Step 5 failed: {e}"
            print(f"\n{'='*80}", flush=True)
            print(f"{error_msg}", flush=True)
            print(f"{'='*80}", flush=True)
            
            full_traceback = traceback.format_exc()
            print(f"\nFull error traceback:", flush=True)
            print(full_traceback, flush=True)
            
            logger.error(error_msg, exc_info=True)
            sys.stdout.flush()
            
            self._log_step_end(step_name, False)
            return False
    
    def step6_train_model(self) -> bool:
        """Step 6: Train model.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Train Model"
        self._log_step_start(step_name, 6, 8)
        
        if not TRAIN_COLAB_AVAILABLE:
            logger.error("train_colab.py not available")
            self._log_step_end(step_name, False)
            return False
        
        try:
            if not self.processed_jsonl.exists():
                raise FileNotFoundError(f"Processed dataset not found: {self.processed_jsonl}")
            if not self.tokenizer_model.exists():
                raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_model}")
            
            # Check if training already complete (has final checkpoint)
            training_config = self.config['training']
            checkpoint_dir = Path(training_config['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            latest_checkpoint = find_latest_checkpoint(str(checkpoint_dir))
            max_steps = training_config['max_steps']
            
            # Check if we've reached max_steps
            if latest_checkpoint:
                # Extract step number from checkpoint filename
                checkpoint_name = Path(latest_checkpoint).stem
                if 'step_' in checkpoint_name:
                    step_num = int(checkpoint_name.split('_')[1])
                    if step_num >= max_steps:
                        logger.info(f"Training already complete (step {step_num} >= {max_steps})")
                        self._log_step_end(step_name, True)
                        return True
            
            # Import model
            from train_real import SimpleMoEModel
            import torch.nn as nn
            import sentencepiece as spm
            from arxiv_dataset import ArXivStreamingDataset, create_dataloader
            from training_adapter import ModelAdapter
            
            # Load tokenizer
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(str(self.tokenizer_model))
            
            # Create model
            model = SimpleMoEModel(
                vocab_size=training_config['vocab_size'],
                embedding_dim=training_config['embedding_dim'],
                num_shared_experts=training_config['num_shared_experts'],
                num_routed_experts=training_config['num_routed_experts'],
                top_k=training_config['top_k']
            )
            
            # Wrap model to match expected signature
            class ModelWrapper(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                
                def forward(self, input_ids):
                    output = self.base_model(
                        input_ids,
                        image_features=None,
                        return_load_balance_loss=False,
                        return_gate_logits=False
                    )
                    if isinstance(output, tuple):
                        return output[0]
                    return output
            
            model = ModelWrapper(model)
            
            # Load pretrained model if specified
            if training_config.get('model_path') and os.path.exists(training_config['model_path']):
                checkpoint = torch.load(training_config['model_path'], map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded pretrained model from {training_config['model_path']}")
            
            # Create dataset
            # Note: dataset reads from text_dir, so we need to keep it until after training
            if not self.text_dir.exists():
                raise FileNotFoundError(
                    f"Text directory not found: {self.text_dir}. "
                    "Text directory is required for training (dataset reads from .txt files)."
                )
            
            dataset = ArXivStreamingDataset(
                text_dir=str(self.text_dir),
                metadata_jsonl=str(self.processed_jsonl),
                tokenizer=tokenizer,
                max_length=training_config['max_length'],
                min_length=training_config['min_length'],
                shuffle_buffer=training_config['shuffle_buffer']
            )
            
            # Create adapter
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            adapter = ModelAdapter(
                model=model,
                device=device,
                domain_weights=training_config.get('domain_weights', {})
            )
            
            # Run training
            train(
                model=model,
                dataset=dataset,
                adapter=adapter,
                checkpoint_dir=str(checkpoint_dir),
                batch_size=training_config['batch_size'],
                gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
                max_steps=training_config['max_steps'],
                learning_rate=training_config['learning_rate'],
                warmup_steps=training_config['warmup_steps'],
                save_interval=training_config.get('checkpoint_interval', 5000),
                log_interval=training_config.get('log_interval', 100),
                resume_from_checkpoint=None  # Auto-detect
            )
            
            # Cleanup text_dir after training if requested
            if self.config['pipeline']['cleanup_intermediate']:
                if self.text_dir.exists():
                    logger.info(f"🧹 Cleaning up intermediate files: {self.text_dir}")
                    shutil.rmtree(self.text_dir)
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"Step 6 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step7_evaluate(self) -> bool:
        """Step 7: Evaluate model.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Evaluate Model"
        self._log_step_start(step_name, 7, 8)
        
        if not EVALUATE_AVAILABLE:
            logger.error("evaluate.py not available")
            self._log_step_end(step_name, False)
            return False
        
        try:
            # Find latest checkpoint
            checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
            latest_checkpoint = find_latest_checkpoint(str(checkpoint_dir))
            
            if not latest_checkpoint:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
            
            if not self.processed_jsonl.exists():
                raise FileNotFoundError(f"Processed dataset not found: {self.processed_jsonl}")
            if not self.tokenizer_model.exists():
                raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_model}")
            
            eval_config = self.config['evaluation']
            eval_dir = Path(eval_config['output_dir'])
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Run evaluation
            # Note: evaluation may need text_dir if dataset requires it
            # For now, pass text_dir if it exists, otherwise None
            text_dir_for_eval = str(self.text_dir) if self.text_dir.exists() else None
            
            evaluate_model(
                model_checkpoint=latest_checkpoint,
                dataset_text_dir=text_dir_for_eval,
                dataset_metadata=str(self.processed_jsonl),
                tokenizer_path=str(self.tokenizer_model),
                output_dir=str(eval_dir),
                test_split=eval_config.get('test_split', 0.1),
                batch_size=eval_config.get('batch_size', 16)
            )
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"Step 7 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step8_export_inference(self) -> bool:
        """Step 8: Export inference pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Export Inference"
        self._log_step_start(step_name, 8, 8)
        
        if not INFERENCE_AVAILABLE:
            logger.error("inference.py not available")
            self._log_step_end(step_name, False)
            return False
        
        try:
            # Find latest checkpoint
            checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
            latest_checkpoint = find_latest_checkpoint(str(checkpoint_dir))
            
            if not latest_checkpoint:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
            
            if not self.tokenizer_model.exists():
                raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_model}")
            
            inference_config = self.config['inference']
            inference_dir = Path(inference_config['output_dir'])
            inference_dir.mkdir(parents=True, exist_ok=True)
            
            # Create inference pipeline
            pipeline = InferencePipeline(
                checkpoint_path=latest_checkpoint,
                tokenizer_path=str(self.tokenizer_model),
                device='cpu',  # Default to CPU for inference export
                quantize=inference_config.get('quantize_int8', False)  # Map quantize_int8 to quantize
            )
            
            # Precompute embeddings if requested
            if inference_config.get('precompute_embeddings', False) and self.processed_jsonl.exists():
                logger.info("Precomputing corpus embeddings...")
                embeddings_path = inference_dir / "corpus_embeddings.npz"
                
                # Read corpus texts from JSONL
                corpus_texts = []
                metadata_list = []
                with open(self.processed_jsonl, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            doc = json.loads(line)
                            text = doc.get('text', '')
                            if text:
                                corpus_texts.append(text)
                                metadata_list.append({
                                    'arxiv_id': doc.get('arxiv_id', ''),
                                    'domains': doc.get('domains', []),
                                    'year': doc.get('year', None)
                                })
                
                if corpus_texts:
                    pipeline.precompute_corpus_embeddings(
                        corpus_texts=corpus_texts,
                        output_path=str(embeddings_path),
                        metadata=metadata_list if metadata_list else None
                    )
                    logger.info(f"Precomputed embeddings for {len(corpus_texts)} documents")
                else:
                    logger.warning("No texts found in processed dataset, skipping embedding precomputation")
            
            # Export to ONNX if requested
            if inference_config.get('export_onnx', False):
                onnx_path = inference_dir / "model.onnx"
                pipeline.export_to_onnx(str(onnx_path))
            
            # Save pipeline info
            pipeline_info = {
                'checkpoint_path': latest_checkpoint,
                'tokenizer_path': str(self.tokenizer_model),
                'quantized': inference_config.get('quantize_int8', False),
                'exported_at': datetime.now().isoformat()
            }
            
            with open(inference_dir / "pipeline_info.json", 'w') as f:
                json.dump(pipeline_info, f, indent=2)
            
            logger.info(f"Inference pipeline exported to {inference_dir}")
            
            self._log_step_end(step_name, True)
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Step 8 failed - file not found: {e}")
            logger.info("Tip: Ensure training completed successfully and checkpoint exists")
            self._log_step_end(step_name, False)
            return False
        except ImportError as e:
            logger.error(f"Step 8 failed - import error: {e}")
            logger.info("Tip: Ensure all dependencies are installed (sentencepiece, torch, etc.)")
            self._log_step_end(step_name, False)
            return False
        except Exception as e:
            logger.error(f"Step 8 failed: {e}", exc_info=True)
            logger.error(f"   Full traceback:", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def generate_report(self) -> Dict:
        """Generate final pipeline report.
        
        Returns:
            Report dictionary
        """
        total_elapsed = time.time() - self.start_time
        
        # Collect statistics
        report = {
            'pipeline': {
                'config_file': self.config_path,
                'output_dir': str(self.output_dir),
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_elapsed_seconds': total_elapsed,
                'total_elapsed_hours': total_elapsed / 3600
            },
            'steps': self.step_status,
            'files': {
                'metadata_jsonl': str(self.metadata_jsonl) if self.metadata_jsonl.exists() else None,
                'processed_jsonl': str(self.processed_jsonl) if self.processed_jsonl.exists() else None,
                'tokenizer_model': str(self.tokenizer_model) if self.tokenizer_model.exists() else None,
                'checkpoint_dir': str(self.checkpoint_dir) if self.checkpoint_dir.exists() else None,
                'eval_dir': str(self.eval_dir) if self.eval_dir.exists() else None,
                'inference_dir': str(self.inference_dir) if self.inference_dir.exists() else None
            },
            'statistics': {}
        }
        
        # Count papers at each stage
        if self.metadata_jsonl.exists():
            report['statistics']['collected_papers'] = sum(1 for _ in open(self.metadata_jsonl))
        
        if self.processed_jsonl.exists():
            report['statistics']['processed_papers'] = sum(1 for _ in open(self.processed_jsonl))
        
        # Training statistics
        if self.checkpoint_dir.exists():
            checkpoints = list(self.checkpoint_dir.glob("step_*.pt"))
            if checkpoints:
                report['statistics']['checkpoints'] = len(checkpoints)
                # Find latest checkpoint step
                latest_step = max(
                    int(c.stem.split('_')[1]) for c in checkpoints
                    if 'step_' in c.stem
                )
                report['statistics']['latest_training_step'] = latest_step
        
        # Evaluation statistics
        if self.eval_dir.exists():
            eval_json = self.eval_dir / "evaluation_results.json"
            if eval_json.exists():
                with open(eval_json, 'r') as f:
                    eval_results = json.load(f)
                    report['statistics']['evaluation'] = eval_results
        
        return report
    
    def run(self, start_from_step: Optional[int] = None):
        """Run the complete pipeline.
        
        Args:
            start_from_step: Step number to start from (1-8). If None, starts from first incomplete step.
        """
        steps = [
            (1, self.step1_collect_papers),
            (2, self.step2_extract_pdfs),
            (3, self.step3_nemo_curator),
            (4, self.step4_process_curated),
            (5, self.step5_train_tokenizer),
            (6, self.step6_train_model),
            (7, self.step7_evaluate),
            (8, self.step8_export_inference)
        ]
        
        # Determine starting step
        if start_from_step is None:
            # Find first incomplete step
            for step_num, step_func in steps:
                step_name = step_func.__name__.replace('step', '').replace('_', ' ').title()
                # Simple heuristic: if step completed successfully, skip it
                if step_name in self.step_status and self.step_status[step_name]['success']:
                    continue
                start_from_step = step_num
                break
            else:
                logger.info("All steps already complete!")
                return
        
        logger.info(f"Starting pipeline from step {start_from_step}")
        
        # Run steps
        for step_num, step_func in steps:
            if step_num < start_from_step:
                continue
            
            success = step_func()
            if not success:
                logger.error(f"Pipeline stopped at step {step_num}")
                break
        
        # Generate final report
        logger.info("")
        logger.info("=" * 80)
        logger.info("Generating Final Report")
        logger.info("=" * 80)
        
        report = self.generate_report()
        report_path = self.output_dir / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
        logger.info("")
        logger.info("=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"Total time: {report['pipeline']['total_elapsed_hours']:.2f} hours")
        logger.info(f"Steps completed: {sum(1 for s in self.step_status.values() if s['success'])}/8")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete NeMo Curator + training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config.yaml file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--start-from-step',
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help='Start from specific step (1-8). If not specified, resumes from first incomplete step.'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print(f"   Please create config.yaml or specify --config")
        sys.exit(1)
    
    orchestrator = PipelineOrchestrator(args.config)
    orchestrator.run(start_from_step=args.start_from_step)


if __name__ == "__main__":
    main()

