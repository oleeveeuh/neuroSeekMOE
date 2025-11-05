#!/usr/bin/env python3
"""
Compare Different Expert Configurations for NeuroSeek-MoE

This script trains models with different expert configurations and selects
the best performing setup based on validation metrics.

Usage:
    python compare_expert_configs.py \
      --multimodal-jsonl processed/multimodal_dataset.jsonl \
      --epochs 10 \
      --expert-configs 1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from train_real import train_real_model

def _ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def parse_expert_config(config_str: str) -> Tuple[int, int, int, str]:
    """Parse expert configuration string.
    
    Formats supported:
    - "N" -> (N, N, 0) - N text + N image experts
    - "Ntext" -> (N, 0, 0) - N text-only experts
    - "Nimage" -> (0, N, 0) - N image-only experts
    - "Nmultimodal" -> (0, 0, N) - N multimodal experts
    - "Ntext+Mimage" -> (N, M, 0) - N text + M image
    - "Ntext+Mimage+Kmultimodal" -> (N, M, K) - Combined
    
    Returns:
        Tuple of (num_text_experts, num_image_experts, num_multimodal_experts, config_name)
    """
    config_str = config_str.strip().strip(",").lower()  # Strip whitespace and commas
    
    # Simple number format: "N" means N text + N image
    if config_str.isdigit():
        n = int(config_str)
        return (n, n, 0, f"{n}expert")
    
    num_text = 0
    num_image = 0
    num_multimodal = 0
    
    # Parse parts like "2text", "3image", "1multimodal"
    parts = config_str.replace("expert", "").replace("_", "").split("+")
    
    for part in parts:
        part = part.strip().strip(",")  # Strip whitespace and commas from each part
        if not part:  # Skip empty parts
            continue
        if part.endswith("text"):
            num_text = int(part.replace("text", "")) if part.replace("text", "").isdigit() else 0
        elif part.endswith("image"):
            num_image = int(part.replace("image", "")) if part.replace("image", "").isdigit() else 0
        elif part.endswith("multimodal") or part.endswith("multi"):
            num_multimodal = int(part.replace("multimodal", "").replace("multi", "")) if part.replace("multimodal", "").replace("multi", "").isdigit() else 0
    
    # Generate config name
    name_parts = []
    if num_text > 0:
        name_parts.append(f"{num_text}text")
    if num_image > 0:
        name_parts.append(f"{num_image}image")
    if num_multimodal > 0:
        name_parts.append(f"{num_multimodal}multi")
    
    config_name = "+".join(name_parts) if name_parts else "unknown"
    
    return (num_text, num_image, num_multimodal, config_name)


def train_configuration(
    config_name: str,
    expert_config_str: str,
    multimodal_jsonl: str,
    base_outputs: str,
    base_checkpoints: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    disable_diagrams: bool = True,
    resume_from_epoch: int = None,
    vocab_path: str = None,
    early_stopping_patience: int = 5,
) -> Dict:
    """Train a single expert configuration and return metrics.
    
    Supports resuming from checkpoints if training was interrupted.
    
    Args:
        expert_config_str: Configuration string like "2", "1text+1image", "1multimodal", etc.
    """
    
    # Parse expert configuration
    num_text_experts, num_image_experts, num_multimodal_experts, parsed_name = parse_expert_config(expert_config_str)
    
    # Use provided config_name or parsed name
    if config_name is None or config_name == "auto":
        config_name = parsed_name
    
    config_outputs = os.path.join(base_outputs, f"config_{config_name}")
    config_checkpoints = os.path.join(base_checkpoints, f"config_{config_name}")
    config_results = os.path.join(base_outputs, f"config_{config_name}_results.json")
    
    total_experts = num_text_experts + num_image_experts + num_multimodal_experts
    
    # Check if configuration already has results (completed training)
    if os.path.exists(config_results) and resume_from_epoch is None:
        try:
            with open(config_results, "r") as f:
                existing_results = json.load(f)
            if existing_results.get("training_complete", False):
                print(f"\n{'='*70}")
                print(f"⏭️  Configuration {config_name} already completed")
                print(f"{'='*70}")
                print(f"   Train Loss: {existing_results.get('final_loss', 'N/A'):.4f}")
                print(f"   Test Loss:  {existing_results.get('test_loss', 'N/A'):.4f}")
                print(f"   Train BERTScore: {existing_results.get('final_bertscore', 'N/A'):.4f}")
                print(f"   Test BERTScore: {existing_results.get('test_bertscore', 'N/A'):.4f}")
                
                # Return existing metrics with config info
                existing_results["config_name"] = config_name
                existing_results["training_time"] = existing_results.get("training_time", 0)
                return existing_results
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}, will retrain")
    
    # Check for existing checkpoints to resume from
    if resume_from_epoch is None:
        # Auto-detect latest checkpoint
        if os.path.isdir(config_checkpoints):
            checkpoint_files = [
                f for f in os.listdir(config_checkpoints)
                if f.startswith("model_epoch_") and f.endswith(".pt")
            ]
            if checkpoint_files:
                # Extract epoch numbers and find max
                try:
                    epoch_nums = [
                        int(f.replace("model_epoch_", "").replace(".pt", ""))
                        for f in checkpoint_files
                    ]
                    latest_epoch = max(epoch_nums) if epoch_nums else 0
                    if latest_epoch < epochs:
                        print(f"📂 Found checkpoint at epoch {latest_epoch}, resuming from there...")
                        resume_from_epoch = latest_epoch
                except Exception:
                    pass
    
    print(f"\n{'='*70}")
    print(f"🔬 Training Configuration: {config_name}")
    print(f"   Expert breakdown:")
    if num_text_experts > 0:
        print(f"     - Text-only: {num_text_experts}")
    if num_image_experts > 0:
        print(f"     - Image-only: {num_image_experts}")
    if num_multimodal_experts > 0:
        print(f"     - Multimodal: {num_multimodal_experts}")
    print(f"   Total experts: {total_experts}")
    if resume_from_epoch:
        print(f"   Resuming from epoch {resume_from_epoch}/{epochs}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Use shared vocabulary path for all configs (optional, can use per-config vocab)
    shared_vocab_path = vocab_path
    if shared_vocab_path is None:
        shared_vocab_path = os.path.join(base_outputs, "vocabulary.json")
    
    metrics = train_real_model(
        multimodal_jsonl=multimodal_jsonl,
        results_path=config_results,
        outputs_dir=config_outputs,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=config_checkpoints,
        disable_diagrams=disable_diagrams,
        num_text_experts=num_text_experts,
        num_image_experts=num_image_experts,
        num_multimodal_experts=num_multimodal_experts,
        resume_from_epoch=resume_from_epoch,
        vocab_path=shared_vocab_path,
        comparison_mode=True,
        early_stopping_patience=early_stopping_patience if early_stopping_patience > 0 else None,
    )
    
    training_time = time.time() - start_time
    metrics["config_name"] = config_name
    metrics["num_text_experts"] = num_text_experts
    metrics["num_image_experts"] = num_image_experts
    metrics["num_multimodal_experts"] = num_multimodal_experts
    metrics["total_experts"] = total_experts
    metrics["training_time"] = training_time
    metrics["training_complete"] = True
    
    # Save individual config results
    with open(config_results, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Configuration {config_name} complete!")
    print(f"   Train Loss: {metrics.get('final_loss', 'N/A'):.4f}")
    print(f"   Test Loss:  {metrics.get('test_loss', 'N/A'):.4f}")
    print(f"   Train BERTScore: {metrics.get('final_bertscore', 'N/A'):.4f}")
    print(f"   Test BERTScore:  {metrics.get('test_bertscore', 'N/A'):.4f}")
    print(f"   Training time: {training_time:.2f}s ({training_time/60:.1f} min)")
    
    return metrics


def select_best_configuration(
    all_metrics: List[Dict],
    selection_metric: str = "bertscore"
) -> Tuple[Dict, str]:
    """Select the best configuration based on a metric.
    
    Args:
        all_metrics: List of metrics dictionaries from all configurations
        selection_metric: Metric to use for selection ('loss', 'bertscore', or 'combined')
    
    Returns:
        Tuple of (best_metrics_dict, best_config_name)
    """
    
    if not all_metrics:
        raise ValueError("No metrics provided")
    
    if selection_metric == "loss":
        # Lower is better for loss - use TEST loss
        best = min(all_metrics, key=lambda m: m.get("test_loss", m.get("final_loss", float('inf'))))
        metric_value = best.get("test_loss", best.get("final_loss", 0))
        print(f"\n🏆 Best configuration by TEST LOSS: {best['config_name']}")
        print(f"   Test Loss: {metric_value:.4f}")
        print(f"   Train Loss: {best.get('final_loss', 'N/A'):.4f}")
        
    elif selection_metric == "bertscore":
        # Higher is better for BERTScore - use TEST BERTScore
        best = max(all_metrics, key=lambda m: m.get("test_bertscore", m.get("final_bertscore", 0)))
        metric_value = best.get("test_bertscore", best.get("final_bertscore", 0))
        print(f"\n🏆 Best configuration by TEST BERTScore: {best['config_name']}")
        print(f"   Test BERTScore: {metric_value:.4f}")
        print(f"   Train BERTScore: {best.get('final_bertscore', 'N/A'):.4f}")
        
    elif selection_metric == "combined":
        # Combined score: weighted combination - use TEST metrics
        # Normalize each metric to 0-1 range, then combine
        def compute_score(m):
            loss = m.get("test_loss", m.get("final_loss", 10.0))
            bert = m.get("test_bertscore", m.get("final_bertscore", 0.0))
            
            # Normalize loss (assume max loss of 20, lower is better)
            norm_loss = max(0, 1 - (loss / 20.0))
            
            # Combined: 50% loss, 50% BERTScore
            score = 0.5 * norm_loss + 0.5 * bert
            return score
        
        best = max(all_metrics, key=compute_score)
        score = compute_score(best)
        print(f"\n🏆 Best configuration by COMBINED TEST SCORE: {best['config_name']}")
        print(f"   Combined Score: {score:.4f}")
        print(f"   Test Loss: {best.get('test_loss', best.get('final_loss', 0)):.4f}")
        print(f"   Test BERTScore: {best.get('test_bertscore', best.get('final_bertscore', 0)):.4f}")
        
    else:
        raise ValueError(f"Unknown selection metric: {selection_metric}")
    
    return best, best['config_name']


def show_configuration_status(
    outputs_dir: str,
    checkpoints_dir: str,
    comparison_output: str,
    best_config_output: str,
) -> None:
    """Display current configuration status without running training."""
    
    print("="*70)
    print("📊 Current Expert Configuration Status")
    print("="*70)
    
    # Load comparison results if available
    comparison_data = None
    if os.path.exists(comparison_output):
        try:
            with open(comparison_output, "r") as f:
                comparison_data = json.load(f)
            print(f"\n✅ Comparison results found: {comparison_output}")
        except Exception as e:
            print(f"\n⚠️  Could not load comparison results: {e}")
    else:
        print(f"\n⚠️  No comparison results found at: {comparison_output}")
    
    # Load best config if available
    best_config_data = None
    if os.path.exists(best_config_output):
        try:
            with open(best_config_output, "r") as f:
                best_config_data = json.load(f)
            print(f"✅ Best configuration info found: {best_config_output}")
        except Exception as e:
            print(f"⚠️  Could not load best config: {e}")
    else:
        print(f"⚠️  No best config found at: {best_config_output}")
    
    # Discover configurations from directories
    discovered_configs = []
    if os.path.exists(checkpoints_dir):
        for item in os.listdir(checkpoints_dir):
            if item.startswith("config_") and os.path.isdir(os.path.join(checkpoints_dir, item)):
                config_name = item.replace("config_", "")
                config_dir = os.path.join(checkpoints_dir, item)
                results_file = os.path.join(outputs_dir, f"config_{config_name}_results.json")
                
                # Check checkpoint status
                checkpoint_files = [
                    f for f in os.listdir(config_dir)
                    if f.startswith("model_epoch_") and f.endswith(".pt")
                ]
                latest_epoch = 0
                if checkpoint_files:
                    try:
                        epoch_nums = [
                            int(f.replace("model_epoch_", "").replace(".pt", ""))
                            for f in checkpoint_files
                        ]
                        latest_epoch = max(epoch_nums)
                    except Exception:
                        pass
                
                # Check completion status
                is_complete = False
                if os.path.exists(results_file):
                    try:
                        with open(results_file, "r") as f:
                            results = json.load(f)
                        is_complete = results.get("training_complete", False)
                    except Exception:
                        pass
                
                discovered_configs.append({
                    "name": config_name,
                    "latest_epoch": latest_epoch,
                    "checkpoints": len(checkpoint_files),
                    "complete": is_complete,
                    "results_file": results_file,
                })
    
    if discovered_configs:
        print(f"\n📁 Discovered Configurations ({len(discovered_configs)}):")
        print(f"{'Config Name':<20} {'Latest Epoch':<15} {'Checkpoints':<12} {'Status':<15}")
        print("-"*70)
        for cfg in sorted(discovered_configs, key=lambda x: x["name"]):
            status = "✅ Complete" if cfg["complete"] else f"🔄 Epoch {cfg['latest_epoch']}"
            print(f"{cfg['name']:<20} {cfg['latest_epoch']:<15} {cfg['checkpoints']:<12} {status:<15}")
    else:
        print(f"\n⚠️  No configurations found in {checkpoints_dir}")
    
    # Display best configuration
    if best_config_data:
        print(f"\n🏆 Best Configuration:")
        print(f"   Name: {best_config_data.get('config_name', 'N/A')}")
        
        # Show expert breakdown
        num_text = best_config_data.get('num_text_experts', best_config_data.get('num_experts', 0))
        num_image = best_config_data.get('num_image_experts', best_config_data.get('num_experts', 0))
        num_multi = best_config_data.get('num_multimodal_experts', 0)
        if num_multi > 0 or (num_text != num_image):
            # Show detailed breakdown
            desc_parts = []
            if num_text > 0:
                desc_parts.append(f"{num_text} text")
            if num_image > 0:
                desc_parts.append(f"{num_image} image")
            if num_multi > 0:
                desc_parts.append(f"{num_multi} multimodal")
            print(f"   Expert breakdown: {' + '.join(desc_parts)}")
        else:
            # Legacy format
            print(f"   Experts: {best_config_data.get('num_experts', 'N/A')} per modality")
        
        metrics = best_config_data.get('metrics', {})
        print(f"   Test Loss: {metrics.get('test_loss', 'N/A')}")
        print(f"   Test BERTScore: {metrics.get('test_bertscore', 'N/A')}")
    
    # Load individual result files for configurations not in comparison JSON
    additional_configs = []
    for cfg in discovered_configs:
        if cfg.get("results_file") and os.path.exists(cfg["results_file"]):
            try:
                with open(cfg["results_file"], "r") as f:
                    result_data = json.load(f)
                if result_data.get("training_complete"):
                    # Check if this config is already in comparison_data
                    config_name = cfg["name"]
                    if comparison_data:
                        existing = [c for c in comparison_data.get('all_configs', []) if c.get('config_name') == config_name]
                        if existing:
                            continue  # Already in comparison data
                    additional_configs.append(result_data)
            except Exception:
                pass
    
    # Display comparison summary if available
    if comparison_data:
        print(f"\n📊 Comparison Summary (from comparison file):")
        all_configs = comparison_data.get('all_configs', [])
        selection_metric = comparison_data.get('selection_metric', 'N/A')
        best_name = comparison_data.get('best_config_name', 'N/A')
        
        print(f"   Configurations in comparison file: {len(all_configs)}")
        print(f"   Selection metric: {selection_metric}")
        print(f"   Best config (from comparison): {best_name}")
        
        # Combine with additional configs found
        combined_configs = all_configs + additional_configs
        total_configs = len(combined_configs)
        
        if total_configs > 0:
            print(f"\n   Total configurations found: {total_configs} ({len(all_configs)} in comparison + {len(additional_configs)} additional)")
            print(f"\n   Detailed Metrics:")
            print(f"   {'Config':<20} {'Experts (T/I/M)':<18} {'Test Loss':<12} {'Test BERT':<12}")
            print("   " + "-"*70)
            for cfg in sorted(combined_configs, key=lambda x: x.get('config_name', '')):
                name = cfg.get('config_name', 'unknown')
                num_text = cfg.get('num_text_experts', cfg.get('num_experts', 0))
                num_image = cfg.get('num_image_experts', cfg.get('num_experts', 0))
                num_multi = cfg.get('num_multimodal_experts', 0)
                experts_str = f"{num_text}/{num_image}/{num_multi}"
                test_loss = cfg.get('test_loss', 0)
                test_bert = cfg.get('test_bertscore', 0)
                marker = " ⭐" if name == best_name else ""
                source_marker = " [from file]" if cfg in additional_configs else ""
                print(f"   {name:<20} {experts_str:<18} {test_loss:<12.4f} {test_bert:<12.4f}{marker}{source_marker}")
    elif additional_configs:
        # No comparison file, but found individual results
        print(f"\n📊 Individual Configuration Results:")
        print(f"   Found {len(additional_configs)} completed configurations")
        print(f"\n   Detailed Metrics:")
        print(f"   {'Config':<20} {'Experts (T/I/M)':<18} {'Test Loss':<12} {'Test BERT':<12}")
        print("   " + "-"*70)
        for cfg in sorted(additional_configs, key=lambda x: x.get('config_name', '')):
            name = cfg.get('config_name', 'unknown')
            num_text = cfg.get('num_text_experts', cfg.get('num_experts', 0))
            num_image = cfg.get('num_image_experts', cfg.get('num_experts', 0))
            num_multi = cfg.get('num_multimodal_experts', 0)
            experts_str = f"{num_text}/{num_image}/{num_multi}"
            test_loss = cfg.get('test_loss', 0)
            test_bert = cfg.get('test_bertscore', 0)
            print(f"   {name:<20} {experts_str:<18} {test_loss:<12.4f} {test_bert:<12.4f}")
    
    print("\n" + "="*70)
    print("💡 Tips:")
    print("   - To resume training: --resume-from EPOCH")
    print("   - To resume specific config: --resume-config CONFIGNAME")
    print("   - To compare new configs: --expert-configs 1,2,3,4")
    print("="*70)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare different expert configurations for NeuroSeek-MoE"
    )
    
    # Mode selection
    ap.add_argument("--show-status", action="store_true",
                    help="Show current configuration status without training")
    
    # Data input
    ap.add_argument("--multimodal-jsonl", required=False,
                    help="Path to multimodal JSONL dataset (required unless --show-status)")
    
    # Training options
    ap.add_argument("--epochs", type=int, default=10,
                    help="Number of epochs per configuration")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size")
    ap.add_argument("--learning-rate", type=float, default=0.0001,
                    help="Learning rate (default: 0.0001, reduced to prevent overfitting)")
    ap.add_argument("--early-stopping-patience", type=int, default=5,
                    help="Number of epochs without improvement before early stopping (default: 5, set to 0 to disable)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device for training")
    ap.add_argument("--disable-diagrams", action="store_true",
                    help="Disable diagram generation for faster training")
    
    # Expert configuration options
    ap.add_argument("--expert-configs", type=str, default="1,2,3,4",
                    help="Comma-separated list of expert configurations. "
                         "Formats: 'N' (N text + N image), 'Ntext+Mimage' (N text + M image), "
                         "'Nmultimodal' (N multimodal experts), or 'Ntext+Mimage+Kmultimodal' "
                         "(e.g., '1,2,1text+1image,1multimodal,2text+1image+1multimodal')")
    ap.add_argument("--selection-metric", default="combined",
                    choices=["loss", "bertscore", "combined"],
                    help="Metric to use for selecting best configuration")
    ap.add_argument("--resume-from", type=int, default=None, metavar="EPOCH",
                    help="Resume all configurations from this epoch (or auto-detect if not specified)")
    ap.add_argument("--resume-config", type=str, default=None,
                    help="Resume specific configuration only (e.g., '2expert')")
    
    # Output options
    ap.add_argument("--comparison-output", default="evaluation/expert_comparison.json",
                    help="Path to save comparison results")
    ap.add_argument("--best-config-output", default="evaluation/best_config.json",
                    help="Path to save best configuration info")
    ap.add_argument("--outputs-dir", default="outputs",
                    help="Base outputs directory for all configurations")
    ap.add_argument("--checkpoints-dir", default="checkpoints",
                    help="Base checkpoints directory for all configurations")
    
    args = ap.parse_args()
    
    # Handle status mode
    if args.show_status:
        show_configuration_status(
            outputs_dir=args.outputs_dir,
            checkpoints_dir=args.checkpoints_dir,
            comparison_output=args.comparison_output,
            best_config_output=args.best_config_output,
        )
        return
    
    # Validate required arguments for training mode
    if not args.multimodal_jsonl:
        ap.error("--multimodal-jsonl is required (unless using --show-status)")
    
    # Parse expert configurations
    # Support both old format (numbers) and new format (descriptive strings)
    # Filter out empty strings that may result from trailing commas
    expert_config_strings = [x.strip().strip(",") for x in args.expert_configs.split(",") if x.strip().strip(",")]
    
    # Validate and parse each configuration
    parsed_configs = []
    for config_str in expert_config_strings:
        try:
            num_text, num_image, num_multi, config_name = parse_expert_config(config_str)
            parsed_configs.append({
                "string": config_str,
                "name": config_name,
                "num_text": num_text,
                "num_image": num_image,
                "num_multimodal": num_multi,
            })
        except Exception as e:
            print(f"⚠️  Invalid expert configuration '{config_str}': {e}")
            print(f"   Supported formats: 'N', 'Ntext', 'Nimage', 'Nmultimodal', 'Ntext+Mimage+Kmultimodal'")
            continue
    
    if not parsed_configs:
        print("❌ No valid expert configurations provided!")
        return
    
    print("="*70)
    print("🔬 Expert Configuration Comparison")
    print("="*70)
    print(f"\nConfigurations to test ({len(parsed_configs)}):")
    for cfg in parsed_configs:
        desc_parts = []
        if cfg["num_text"] > 0:
            desc_parts.append(f"{cfg['num_text']} text")
        if cfg["num_image"] > 0:
            desc_parts.append(f"{cfg['num_image']} image")
        if cfg["num_multimodal"] > 0:
            desc_parts.append(f"{cfg['num_multimodal']} multimodal")
        desc = " + ".join(desc_parts) if desc_parts else "unknown"
        print(f"   - {cfg['name']}: {desc}")
    print(f"\nSelection metric: {args.selection_metric}")
    print(f"Epochs per config: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    if args.resume_from:
        print(f"Resume from epoch: {args.resume_from}")
    if args.resume_config:
        print(f"Resume only config: {args.resume_config}")
    
    _ensure_dir(args.outputs_dir)
    _ensure_dir(args.checkpoints_dir)
    _ensure_dir(os.path.dirname(args.comparison_output) or ".")
    
    # Train each configuration
    all_metrics = []
    config_names = []
    
    for cfg_info in parsed_configs:
        config_name = cfg_info["name"]
        config_str = cfg_info["string"]
        config_names.append(config_name)
        
        # Skip if resuming specific config and this isn't it
        if args.resume_config and config_name != args.resume_config:
            # Try to load existing results for skipped configs
            config_results = os.path.join(args.outputs_dir, f"config_{config_name}_results.json")
            if os.path.exists(config_results):
                try:
                    with open(config_results, "r") as f:
                        existing = json.load(f)
                    if existing.get("training_complete"):
                        print(f"\n⏭️  Skipping {config_name} (already completed), loading results...")
                        existing["config_name"] = config_name
                        # Expert counts should already be in existing results
                        all_metrics.append(existing)
                        continue
                except Exception:
                    pass
            continue
        
        try:
            metrics = train_configuration(
                config_name=config_name,
                expert_config_str=config_str,
                multimodal_jsonl=args.multimodal_jsonl,
                base_outputs=args.outputs_dir,
                base_checkpoints=args.checkpoints_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                disable_diagrams=args.disable_diagrams,
                resume_from_epoch=args.resume_from,
                early_stopping_patience=args.early_stopping_patience,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n❌ Configuration {config_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_metrics:
        print("\n❌ No configurations completed successfully!")
        return
    
    # Select best configuration
    best_metrics, best_config_name = select_best_configuration(
        all_metrics,
        selection_metric=args.selection_metric
    )
    
    # Save comparison results
    comparison_results = {
        "all_configs": all_metrics,
        "best_config": best_metrics,
        "best_config_name": best_config_name,
        "selection_metric": args.selection_metric,
        "expert_configs_tested": [cfg["string"] for cfg in parsed_configs],
    }
    
    with open(args.comparison_output, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    # Save best configuration info (for evaluation notebook)
    best_config_info = {
        "config_name": best_config_name,
        "num_text_experts": best_metrics.get("num_text_experts", best_metrics.get("num_experts", 0)),
        "num_image_experts": best_metrics.get("num_image_experts", best_metrics.get("num_experts", 0)),
        "num_multimodal_experts": best_metrics.get("num_multimodal_experts", 0),
        "total_experts": best_metrics.get("total_experts", best_metrics.get("num_experts", 0) * 2 if best_metrics.get("num_experts") else 0),
        "metrics": {
            "final_loss": best_metrics.get("final_loss"),
            "final_bertscore": best_metrics.get("final_bertscore"),
            "test_loss": best_metrics.get("test_loss"),
            "test_bertscore": best_metrics.get("test_bertscore"),
        },
        "model_path": os.path.join(
            args.outputs_dir,
            f"config_{best_config_name}",
            "final_model.pt"
        ),
        "checkpoint_dir": os.path.join(
            args.checkpoints_dir,
            f"config_{best_config_name}"
        ),
    }
    
    with open(args.best_config_output, "w") as f:
        json.dump(best_config_info, f, indent=2)
    
    # Print summary table
    print("\n" + "="*70)
    print("📊 CONFIGURATION COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'Experts (T/I/M)':<18} {'Train Loss':<12} {'Test Loss':<12} {'Test BERT':<12} {'Time (min)':<12}")
    print("-"*92)
    
    for metrics in all_metrics:
        config = metrics.get("config_name", "unknown")
        num_text = metrics.get("num_text_experts", metrics.get("num_experts", 0))
        num_image = metrics.get("num_image_experts", metrics.get("num_experts", 0))
        num_multi = metrics.get("num_multimodal_experts", 0)
        experts_str = f"{num_text}/{num_image}/{num_multi}"
        train_loss = metrics.get("final_loss", 0)
        test_loss = metrics.get("test_loss", 0)
        test_bert = metrics.get("test_bertscore", 0)
        time_min = metrics.get("training_time", 0) / 60
        
        marker = " ⭐" if config == best_config_name else ""
        print(f"{config:<20} {experts_str:<18} {train_loss:<12.4f} {test_loss:<12.4f} {test_bert:<12.4f} {time_min:<12.2f}{marker}")
    
    print("\n" + "="*70)
    print(f"✅ Best configuration: {best_config_name}")
    print(f"   Saved to: {args.best_config_output}")
    print(f"   Full comparison: {args.comparison_output}")
    print("="*70)
    
    # Check if any configurations are incomplete
    incomplete_configs = []
    for metrics in all_metrics:
        if not metrics.get("training_complete", False):
            incomplete_configs.append(metrics.get("config_name", "unknown"))
    
    if incomplete_configs:
        print(f"\n⚠️  Incomplete configurations: {', '.join(incomplete_configs)}")
        print(f"   Resume with: --resume-from EPOCH or --resume-config CONFIG")
    else:
        print("\n💡 All configurations completed!")
        print(f"   Use the best configuration in evaluation:")
        print(f"   python -c \"import json; print(json.load(open('{args.best_config_output}'))['model_path'])\"")


if __name__ == "__main__":
    main()

