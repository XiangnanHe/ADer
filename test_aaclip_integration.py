#!/usr/bin/env python
"""
Quick test script to verify AA-CLIP integration

Run this from the ADer directory:
    python test_aaclip_integration.py
"""

import sys
import os

def test_imports():
    """Test that all necessary modules can be imported"""
    print("Testing AA-CLIP integration...")
    print("=" * 60)

    try:
        # Add AA-CLIP to path
        sys.path.insert(0, '/mnt/task_runtime/AA-CLIP')
        print("✓ AA-CLIP path added")

        # Test model import
        from model.aaclip import aaclip
        print("✓ AA-CLIP model import successful")

        # Test trainer import
        from trainer.aaclip_trainer import AAClipTrainer
        print("✓ AA-CLIP trainer import successful")

        # Test config import
        from configs.benchmark.aa_clip.aaclip_mvtec_518 import cfg
        print("✓ AA-CLIP config import successful")

        # Instantiate config
        config = cfg()
        print("✓ Config instantiation successful")
        print()
        print("Configuration summary:")
        print(f"  - Trainer: {config.trainer.name}")
        print(f"  - Model: {config.model.name}")
        print(f"  - Text epochs: {config.text_epochs}")
        print(f"  - Image epochs: {config.image_epochs}")
        print(f"  - Total epochs: {config.epoch_full}")
        print(f"  - Image size: {config.size}")
        print(f"  - Batch size: {config.batch_train}")
        print(f"  - Text LR: {config.text_lr}")
        print(f"  - Image LR: {config.image_lr}")
        print()

        # Check checkpoint
        ckpt_path = config.model.kwargs['checkpoint_path']
        if os.path.exists(ckpt_path):
            size_gb = os.path.getsize(ckpt_path) / (1024**3)
            print(f"✓ CLIP checkpoint found: {ckpt_path}")
            print(f"  Size: {size_gb:.2f} GB")
        else:
            print(f"✗ WARNING: CLIP checkpoint not found at {ckpt_path}")
            print(f"  Please download the checkpoint first")

        print()
        print("=" * 60)
        print("All tests passed! AA-CLIP is ready to use.")
        print()
        print("To train AA-CLIP, run:")
        print("  python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train")
        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == '__main__':
    test_imports()
