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
    process_curated_dataset,
    train_healthcare_tokenizer
)

try:
    from train_colab import train, find_latest_checkpoint
    TRAIN_COLAB_AVAILABLE = True
except ImportError:
    TRAIN_COLAB_AVAILABLE = False
    print("⚠️  train_colab.py not available")

try:
    from evaluate import evaluate_model
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    print("⚠️  evaluate.py not available")

try:
    from inference import InferencePipeline
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("⚠️  inference.py not available")

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
        logger.info("🚀 Pipeline Orchestrator Initialized")
        logger.info("=" * 80)
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📋 Config file: {config_path}")
        logger.info(f"🔄 Resume mode: {self.config['pipeline']['resume']}")
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
    
    def _check_step_complete(self, step_name: str, output_files: List[Path]) -> bool:
        """Check if a step is complete by verifying output files exist.
        
        Args:
            step_name: Name of the step
            output_files: List of output file paths to check
            
        Returns:
            True if all output files exist, False otherwise
        """
        if not self.config['pipeline']['resume']:
            return False
        
        all_exist = all(f.exists() for f in output_files)
        if all_exist:
            logger.info(f"✅ Step '{step_name}' already complete (output files exist)")
        return all_exist
    
    def _log_step_start(self, step_name: str, step_num: int, total_steps: int):
        """Log step start.
        
        Args:
            step_name: Name of the step
            step_num: Step number (1-indexed)
            total_steps: Total number of steps
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"📦 Step {step_num}/{total_steps}: {step_name}")
        logger.info("=" * 80)
        self.step_times[step_name] = time.time()
    
    def _log_step_end(self, step_name: str, success: bool = True):
        """Log step completion.
        
        Args:
            step_name: Name of the step
            success: Whether step completed successfully
        """
        elapsed = time.time() - self.step_times.get(step_name, time.time())
        status = "✅" if success else "❌"
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
        
        # Check if already complete
        if self._check_step_complete(step_name, [self.metadata_jsonl]):
            self._log_step_end(step_name, True)
            return True
        
        try:
            collection_config = self.config['collection']
            rate_limit_delay = 1.0 / collection_config['rate_limit']
            
            collect_arxiv_papers(
                output_dir=str(self.output_dir),
                max_papers=collection_config['max_papers'],
                cache_file=collection_config.get('cache_file'),
                rate_limit_delay=rate_limit_delay
            )
            
            if not self.metadata_jsonl.exists():
                raise FileNotFoundError(f"Metadata file not created: {self.metadata_jsonl}")
            
            # Count collected papers
            count = sum(1 for _ in open(self.metadata_jsonl))
            logger.info(f"📊 Collected {count} papers")
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 1 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step2_extract_pdfs(self) -> bool:
        """Step 2: Extract PDF texts.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Extract PDFs"
        self._log_step_start(step_name, 2, 8)
        
        # Check if already complete
        if self._check_step_complete(step_name, [self.text_dir]):
            if self.text_dir.exists() and any(self.text_dir.iterdir()):
                self._log_step_end(step_name, True)
                return True
        
        try:
            if not self.metadata_jsonl.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_jsonl}")
            
            extraction_config = self.config['extraction']
            
            extract_pdf_texts(
                input_jsonl=str(self.metadata_jsonl),
                output_dir=str(self.text_dir),
                num_workers=extraction_config['num_workers'],
                rate_limit_delay=extraction_config['rate_limit_delay']
            )
            
            if not self.text_dir.exists():
                raise FileNotFoundError(f"Text directory not created: {self.text_dir}")
            
            # Count extracted texts
            text_files = list(self.text_dir.glob("*.txt"))
            logger.info(f"📊 Extracted {len(text_files)} text files")
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 2 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step3_nemo_curator(self) -> bool:
        """Step 3: NeMo Curator curation.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "NeMo Curator Curation"
        self._log_step_start(step_name, 3, 8)
        
        # Check if already complete
        if self._check_step_complete(step_name, [self.curated_jsonl]):
            self._log_step_end(step_name, True)
            return True
        
        try:
            if not self.text_dir.exists() or not any(self.text_dir.iterdir()):
                raise FileNotFoundError(f"Text directory not found or empty: {self.text_dir}")
            if not self.metadata_jsonl.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_jsonl}")
            
            nemo_config = self.config['nemo_curator']
            
            curate_with_nemo(
                text_dir=str(self.text_dir),
                metadata_jsonl=str(self.metadata_jsonl),
                output_jsonl=str(self.curated_jsonl),
                use_gpu=nemo_config.get('use_gpu', False),
                skip_dedup=nemo_config.get('skip_dedup', False),
                min_relevance_score=nemo_config.get('min_relevance_score', 0.5)
            )
            
            if not self.curated_jsonl.exists():
                raise FileNotFoundError(f"Curated dataset not created: {self.curated_jsonl}")
            
            # Count curated papers
            count = sum(1 for _ in open(self.curated_jsonl))
            logger.info(f"📊 Curated {count} papers")
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 3 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step4_process_curated(self) -> bool:
        """Step 4: Healthcare-specific processing.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Process Curated Dataset"
        self._log_step_start(step_name, 4, 8)
        
        # Check if already complete
        if self._check_step_complete(step_name, [self.processed_jsonl]):
            self._log_step_end(step_name, True)
            return True
        
        try:
            if not self.curated_jsonl.exists():
                raise FileNotFoundError(f"Curated dataset not found: {self.curated_jsonl}")
            
            processing_config = self.config['processing']
            
            process_curated_dataset(
                input_jsonl=str(self.curated_jsonl),
                output_jsonl=str(self.processed_jsonl),
                num_workers=processing_config['num_workers']
            )
            
            if not self.processed_jsonl.exists():
                raise FileNotFoundError(f"Processed dataset not created: {self.processed_jsonl}")
            
            # Count processed papers
            count = sum(1 for _ in open(self.processed_jsonl))
            logger.info(f"📊 Processed {count} papers")
            
            # Cleanup intermediate files if requested (but keep text_dir for training)
            if self.config['pipeline']['cleanup_intermediate']:
                if self.curated_jsonl.exists() and self.processed_jsonl.exists():
                    logger.info(f"🧹 Cleaning up intermediate file: {self.curated_jsonl}")
                    self.curated_jsonl.unlink()
                # Note: text_dir will be cleaned up after training (step 6)
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 4 failed: {e}", exc_info=True)
            self._log_step_end(step_name, False)
            return False
    
    def step5_train_tokenizer(self) -> bool:
        """Step 5: Train SentencePiece tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        step_name = "Train Tokenizer"
        self._log_step_start(step_name, 5, 8)
        
        # Check if already complete
        if self._check_step_complete(step_name, [self.tokenizer_model, self.tokenizer_vocab]):
            self._log_step_end(step_name, True)
            return True
        
        try:
            if not self.processed_jsonl.exists():
                raise FileNotFoundError(f"Processed dataset not found: {self.processed_jsonl}")
            
            tokenizer_config = self.config['tokenizer']
            
            train_healthcare_tokenizer(
                input_jsonl=str(self.processed_jsonl),
                output_dir=str(self.output_dir),
                model_prefix=tokenizer_config['model_prefix'],
                vocab_size=tokenizer_config['vocab_size']
            )
            
            if not self.tokenizer_model.exists() or not self.tokenizer_vocab.exists():
                raise FileNotFoundError(f"Tokenizer files not created: {self.tokenizer_model}")
            
            logger.info(f"📊 Tokenizer trained: {self.tokenizer_model}")
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 5 failed: {e}", exc_info=True)
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
            logger.error("❌ train_colab.py not available")
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
                        logger.info(f"✅ Training already complete (step {step_num} >= {max_steps})")
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
                logger.info(f"✅ Loaded pretrained model from {training_config['model_path']}")
            
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
            logger.error(f"❌ Step 6 failed: {e}", exc_info=True)
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
            logger.error("❌ evaluate.py not available")
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
            logger.error(f"❌ Step 7 failed: {e}", exc_info=True)
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
            logger.error("❌ inference.py not available")
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
                quantize_int8=inference_config.get('quantize_int8', False)
            )
            
            # Precompute embeddings if requested
            if inference_config.get('precompute_embeddings', False) and self.processed_jsonl.exists():
                logger.info("📊 Precomputing corpus embeddings...")
                embeddings_path = inference_dir / "corpus_embeddings.npz"
                pipeline.precompute_corpus_embeddings(
                    dataset_metadata=str(self.processed_jsonl),
                    output_path=str(embeddings_path)
                )
            
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
            
            logger.info(f"✅ Inference pipeline exported to {inference_dir}")
            
            self._log_step_end(step_name, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 8 failed: {e}", exc_info=True)
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
                logger.info("✅ All steps already complete!")
                return
        
        logger.info(f"🚀 Starting pipeline from step {start_from_step}")
        
        # Run steps
        for step_num, step_func in steps:
            if step_num < start_from_step:
                continue
            
            success = step_func()
            if not success:
                logger.error(f"❌ Pipeline stopped at step {step_num}")
                break
        
        # Generate final report
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 Generating Final Report")
        logger.info("=" * 80)
        
        report = self.generate_report()
        report_path = self.output_dir / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Report saved to: {report_path}")
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total time: {report['pipeline']['total_elapsed_hours']:.2f} hours")
        logger.info(f"📊 Steps completed: {sum(1 for s in self.step_status.values() if s['success'])}/8")


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
        print(f"❌ Config file not found: {args.config}")
        print(f"   Please create config.yaml or specify --config")
        sys.exit(1)
    
    orchestrator = PipelineOrchestrator(args.config)
    orchestrator.run(start_from_step=args.start_from_step)


if __name__ == "__main__":
    main()

