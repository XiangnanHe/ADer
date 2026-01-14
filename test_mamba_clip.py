"""
Quick test script for Mamba-CLIP model

This script verifies:
1. Model can be instantiated
2. Forward pass works
3. Predict method works
4. Config loading works
"""

import torch
import sys
sys.path.insert(0, '/mnt/task_runtime/ADer')

def test_model_instantiation():
    """Test if model can be created"""
    print("=" * 60)
    print("Test 1: Model Instantiation")
    print("=" * 60)

    from model.mamba_clip import MambaCLIP

    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model = MambaCLIP(
        image_size=256,  # Smaller for quick test
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_adapters=3,
        use_ss2d=True,
    )

    if device == 'cuda':
        model = model.cuda()

    print(f"✓ Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, device


def test_forward_pass(model, device='cpu'):
    """Test forward pass"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)

    if device == 'cuda':
        x = x.cuda()

    # Forward pass
    img_scores, pixel_maps = model(x)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Image scores shape: {img_scores.shape}")
    print(f"  Pixel maps shape: {pixel_maps.shape}")

    assert img_scores.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {img_scores.shape}"
    assert pixel_maps.shape == (batch_size, 1, 256, 256), f"Expected shape ({batch_size}, 1, 256, 256), got {pixel_maps.shape}"

    return img_scores, pixel_maps


def test_predict_method(model, device='cpu'):
    """Test predict method"""
    print("\n" + "=" * 60)
    print("Test 3: Predict Method")
    print("=" * 60)

    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)

    if device == 'cuda':
        x = x.cuda()

    # Predict
    scores, maps = model.predict(x)

    print(f"✓ Predict method successful")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Scores type: {type(scores)}")
    print(f"  Maps shape: {maps.shape}")
    print(f"  Maps type: {type(maps)}")

    assert scores.shape == (batch_size,), f"Expected shape ({batch_size},), got {scores.shape}"
    assert maps.shape == (batch_size, 256, 256), f"Expected shape ({batch_size}, 256, 256), got {maps.shape}"


def test_config_loading():
    """Test if config can be loaded"""
    print("\n" + "=" * 60)
    print("Test 4: Config Loading")
    print("=" * 60)

    try:
        from configs.benchmark.mamba_clip.mamba_clip_mvtec_518 import cfg
        config = cfg()

        print(f"✓ Config loaded successfully")
        print(f"  Model name: {config.model.name}")
        print(f"  Image size: {config.image_size}")
        print(f"  Batch size: {config.batch_train}")
        print(f"  Learning rate: {config.lr}")
        print(f"  Epochs: {config.epoch_full}")

    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

    return True


def test_model_registry():
    """Test if model is registered"""
    print("\n" + "=" * 60)
    print("Test 5: Model Registry")
    print("=" * 60)

    from model import MODEL

    # Check if MambaCLIP is registered
    if 'MambaCLIP' in MODEL.name_to_fn:
        print(f"✓ MambaCLIP is registered in MODEL registry")
    else:
        print(f"✗ MambaCLIP NOT found in registry")
        print(f"  Available models: {list(MODEL.name_to_fn.keys())[:10]}...")
        return False

    # Try to get the model
    try:
        model_class = MODEL.get_module('MambaCLIP')
        print(f"✓ Successfully retrieved MambaCLIP class from registry")
    except Exception as e:
        print(f"✗ Failed to get MambaCLIP: {e}")
        return False

    return True


def test_trainer_registry():
    """Test if trainer is registered"""
    print("\n" + "=" * 60)
    print("Test 6: Trainer Registry")
    print("=" * 60)

    from trainer import TRAINER

    # Check if MambaCLIPTrainer is registered
    if 'MambaCLIPTrainer' in TRAINER.name_to_fn:
        print(f"✓ MambaCLIPTrainer is registered in TRAINER registry")
    else:
        print(f"✗ MambaCLIPTrainer NOT found in registry")
        print(f"  Available trainers: {list(TRAINER.name_to_fn.keys())}")
        return False

    return True


def main():
    print("\n" + "=" * 60)
    print("MAMBA-CLIP MODEL TEST SUITE")
    print("=" * 60)

    try:
        # Test 1: Instantiation
        model, device = test_model_instantiation()

        # Test 2: Forward pass
        test_forward_pass(model, device)

        # Test 3: Predict method
        test_predict_method(model, device)

        # Test 4: Config loading
        test_config_loading()

        # Test 5: Model registry
        test_model_registry()

        # Test 6: Trainer registry
        test_trainer_registry()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nModel is ready for training. Run:")
        print("  python run.py -c configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py -m train")
        print()

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
