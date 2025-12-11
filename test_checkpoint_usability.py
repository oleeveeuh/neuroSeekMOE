#!/usr/bin/env python3
"""
Test if the available checkpoint can be used for evaluation.
Run this in Colab to check if step_50000.pt is a valid model checkpoint.
"""

import torch
import sys
import traceback
from pathlib import Path

def test_checkpoint_loading(checkpoint_path):
    """Test if checkpoint can be loaded and used as a model."""

    print(f"🧪 Testing checkpoint: {checkpoint_path}")
    print("=" * 60)

    try:
        # Load checkpoint
        print("1. Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"   ✅ Checkpoint loaded successfully")
        print(f"   📊 Keys: {list(checkpoint.keys())}")

        # Check model state
        print("\n2. Analyzing model state...")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   ✅ Model state found with {len(state_dict)} parameters")

            # Check parameter sizes
            total_params = 0
            print("   📏 Parameter analysis:")
            for key, param in state_dict.items():
                if hasattr(param, 'numel'):
                    param_count = param.numel()
                    total_params += param_count
                    print(f"      {key}: {list(param.shape)} ({param_count:,} params)")

            print(f"   🔢 Total parameters: {total_params:,}")

            # Check if this looks like a real MoE model
            if total_params < 1000000:  # Less than 1M params
                print(f"   ⚠️  WARNING: Very small model ({total_params:,} params)")
                print(f"        This might be a tokenizer or incomplete model")
            elif total_params > 100000000:  # More than 100M params
                print(f"   ✅ Good size for MoE model ({total_params:,} params)")
            else:
                print(f"   🤔 Medium size model ({total_params:,} params)")

        else:
            print(f"   ❌ No 'model_state_dict' found")
            return False

        # Check training step
        print("\n3. Training info...")
        if 'step' in checkpoint:
            step = checkpoint['step']
            print(f"   📈 Training step: {step}")

        # Test model loading
        print("\n4. Testing model creation...")
        try:
            # Try to import and create the model
            sys.path.append('.')
            from train_real import SimpleMoEModel

            # Try to infer model config from parameter shapes
            embedding_dim = None
            vocab_size = None
            num_routed_experts = None
            num_shared_experts = None

            for key, param in state_dict.items():
                if 'embedding.weight' in key:
                    vocab_size, embedding_dim = param.shape
                elif 'routed_experts' in key and 'weight' in key:
                    if num_routed_experts is None:
                        # Count expert layers
                        expert_keys = [k for k in state_dict.keys() if 'routed_experts' in k]
                        expert_nums = []
                        for k in expert_keys:
                            parts = k.split('.')
                            for i, part in enumerate(parts):
                                if part == 'routed_experts' and i+1 < len(parts):
                                    try:
                                        expert_nums.append(int(parts[i+1]))
                                    except:
                                        pass
                        if expert_nums:
                            num_routed_experts = max(expert_nums) + 1

            # Create model with inferred config
            print(f"   🔧 Inferred config:")
            print(f"      vocab_size: {vocab_size or 30000}")
            print(f"      embedding_dim: {embedding_dim or 768}")
            print(f"      num_routed_experts: {num_routed_experts or 8}")

            model = SimpleMoEModel(
                vocab_size=vocab_size or 30000,
                embedding_dim=embedding_dim or 768,
                num_routed_experts=num_routed_experts or 8,
                num_shared_experts=2,
                top_k=2
            )

            # Load state dict
            print("   🔄 Loading state dict into model...")
            model.load_state_dict(state_dict, strict=False)
            print(f"   ✅ Model loaded successfully!")

            # Test forward pass
            print("   🚀 Testing forward pass...")
            model.eval()
            with torch.no_grad():
                input_ids = torch.randint(0, 1000, (2, 10))
                output, routing_info = model(input_ids, image_features=None, return_load_balance_loss=True)
                print(f"   ✅ Forward pass successful!")
                print(f"      Output shape: {output.shape}")
                print(f"      Routing info: {routing_info.keys() if isinstance(routing_info, dict) else type(routing_info)}")

            return True

        except Exception as e:
            print(f"   ❌ Model creation/loading failed: {e}")
            print(f"      Traceback: {traceback.format_exc()}")
            return False

    except Exception as e:
        print(f"   ❌ Checkpoint loading failed: {e}")
        return False

def main():
    """Main function."""

    checkpoint_path = "/content/drive/MyDrive/neuroMOE_results/checkpoints/tokenizer/step_50000.pt"

    print("🔬 NeuroMOE Checkpoint Usability Test")
    print("=" * 50)

    is_usable = test_checkpoint_loading(checkpoint_path)

    print(f"\n🎯 RESULT:")
    print("=" * 30)

    if is_usable:
        print("✅ This checkpoint appears to be a usable MoE model!")
        print("\n📋 Next steps:")
        print("1. Run evaluation with this checkpoint:")
        print(f"   !python evaluate.py \\")
        print(f"       --model-checkpoint {checkpoint_path} \\")
        print(f"       --dataset-metadata /content/drive/MyDrive/neuroMOE_results/data/arxiv/processed_dataset.jsonl \\")
        print(f"       --tokenizer-path /content/drive/MyDrive/neuroMOE_results/data/arxiv/tokenizer/healthcare_tokenizer.model \\")
        print(f"       --output-dir /content/drive/MyDrive/neuroMOE_results/evaluations/test_checkpoint")
        print("\n2. If evaluation still fails, the issue might be:")
        print("   - Tokenizer vocabulary mismatch")
        print("   - Missing model config in checkpoint")
        print("   - Dataset loading issues")
    else:
        print("❌ This checkpoint is NOT usable as a model")
        print("\n🚨 This means:")
        print("1. Your model checkpoints were not saved properly")
        print("2. You may need to retrain the model")
        print("3. Or this is a tokenizer-only checkpoint")
        print("\n💡 Recovery options:")
        print("- Check if you have backups in Google Drive")
        print("- Re-run training from scratch or from an earlier checkpoint")
        print("- Check training logs for where checkpoints were actually saved")

if __name__ == "__main__":
    main()