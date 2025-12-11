#!/usr/bin/env python3
"""
Quick script to check available checkpoints in Google Drive.
Run this in Colab to see what model checkpoints are available.
"""

import os
import glob
import torch
from pathlib import Path

def list_checkpoints(checkpoint_dir):
    """List all available checkpoints in the directory."""

    print(f"🔍 Checking checkpoints in: {checkpoint_dir}")
    print("=" * 60)

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory does not exist: {checkpoint_dir}")
        return []

    # Find all .pt files
    pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

    if not pt_files:
        print(f"❌ No .pt checkpoint files found in {checkpoint_dir}")
        return []

    checkpoints = []
    print(f"📁 Found {len(pt_files)} checkpoint files:")

    for i, file_path in enumerate(pt_files, 1):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        print(f"   {i}. {filename}")
        print(f"      Path: {file_path}")
        print(f"      Size: {file_size_mb:.1f} MB")

        # Try to check if it's a valid checkpoint
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            print(f"      ✅ Valid checkpoint")
            print(f"      📊 Keys: {list(checkpoint.keys())}")

            # Check for model components
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"      🔧 Model has {len(state_dict)} parameters")

                # Check for MoE components
                moe_keys = [k for k in state_dict.keys() if 'expert' in k or 'gate' in k or 'router' in k]
                if moe_keys:
                    print(f"      🎯 Found {len(moe_keys)} MoE parameters")
                else:
                    print(f"      ⚠️  No MoE parameters found (baseline model)")

            # Check for model config
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f"      ⚙️  Model config: vocab_size={config.get('vocab_size', 'N/A')}, "
                      f"embedding_dim={config.get('embedding_dim', 'N/A')}")

            checkpoints.append({
                'path': file_path,
                'filename': filename,
                'size_mb': file_size_mb,
                'valid': True,
                'checkpoint': checkpoint
            })

        except Exception as e:
            print(f"      ❌ Invalid checkpoint: {e}")
            checkpoints.append({
                'path': file_path,
                'filename': filename,
                'size_mb': file_size_mb,
                'valid': False,
                'error': str(e)
            })

        print()

    return checkpoints

def main():
    """Main function to check checkpoints."""

    # Common checkpoint directories in your Google Drive setup
    checkpoint_dirs = [
        "/content/drive/MyDrive/neuroMOE_results/checkpoints",
        "/content/drive/MyDrive/neuroMOE_results/checkpoints/tokenizer",
        "/content/drive/MyDrive/neuroMOE_results/data/arxiv/checkpoints",
        "./checkpoints"
    ]

    print("🚀 NeuroMOE Checkpoint Inspector")
    print("=" * 50)

    all_checkpoints = []

    for checkpoint_dir in checkpoint_dirs:
        checkpoints = list_checkpoints(checkpoint_dir)
        all_checkpoints.extend(checkpoints)
        print()

    # Summary
    print("📋 SUMMARY")
    print("=" * 30)

    valid_checkpoints = [c for c in all_checkpoints if c.get('valid', False)]
    print(f"✅ Valid checkpoints: {len(valid_checkpoints)}")
    print(f"❌ Invalid checkpoints: {len(all_checkpoints) - len(valid_checkpoints)}")

    if valid_checkpoints:
        print(f"\n🎯 Recommended checkpoints for evaluation:")
        for checkpoint in valid_checkpoints:
            if 'model_step_' in checkpoint['filename'] or 'step_' in checkpoint['filename']:
                print(f"   📄 {checkpoint['path']}")
                print(f"      Use this with: --model-checkpoint {checkpoint['path']}")

    # Check specifically for the issue
    print(f"\n🔍 DIAGNOSIS:")
    print(f"You're currently trying to load:")
    print(f"   /content/drive/MyDrive/neuroMOE_results/checkpoints/tokenizer/step_50000.pt")

    tokenizer_checkpoint = next((c for c in all_checkpoints
                               if c['path'] == '/content/drive/MyDrive/neuroMOE_results/checkpoints/tokenizer/step_50000.pt'), None)

    if tokenizer_checkpoint:
        if tokenizer_checkpoint.get('valid'):
            print(f"✅ This file exists and is valid, but it's in the 'tokenizer' subdirectory")
            print(f"⚠️  You probably want a checkpoint from the main 'checkpoints' directory")
        else:
            print(f"❌ This file is corrupted or invalid")

    # Find the correct model checkpoints
    model_checkpoints = [c for c in valid_checkpoints
                        if 'model_step_' in c['filename'] and '/checkpoints/tokenizer/' not in c['path']]

    if model_checkpoints:
        print(f"\n✅ Found these model checkpoints:")
        for checkpoint in model_checkpoints:
            print(f"   📄 {checkpoint['path']} ({checkpoint['size_mb']:.1f} MB)")
            print(f"      Command: --model-checkpoint {checkpoint['path']}")
    else:
        print(f"\n❌ No valid model checkpoints found in expected locations")
        print(f"   Look for files named 'model_step_XXXX.pt' in the main checkpoints directory")

if __name__ == "__main__":
    main()