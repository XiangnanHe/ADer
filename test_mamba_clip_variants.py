"""
Test script for all three Mamba-CLIP variants

Tests:
1. Supervised Mamba-CLIP
2. Zero-Shot Mamba-CLIP
3. Self-Supervised Mamba-CLIP
"""

import torch
import sys
import os

# Add ADer to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_supervised():
    """Test supervised Mamba-CLIP variant"""
    print("\n" + "="*80)
    print("Testing Supervised Mamba-CLIP")
    print("="*80)

    from model import MODEL, get_model
    from argparse import Namespace

    # Check registration
    assert 'MambaCLIP' in MODEL, "MambaCLIP not registered!"
    print("✓ MambaCLIP registered successfully")

    # Create model
    cfg_model = Namespace()
    cfg_model.name = 'MambaCLIP'
    cfg_model.kwargs = dict(
        image_size=256,
        patch_size=14,
        embed_dim=384,
        depth=6,
        num_adapters=3,
        use_ss2d=True,
        pretrained=False,
        checkpoint_path='',
        strict=True,
    )

    model = get_model(cfg_model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.cuda() if device == 'cuda' else model
    print(f"✓ Model created on {device}")

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    x = x.cuda() if device == 'cuda' else x

    model.eval()
    with torch.no_grad():
        img_scores, pixel_maps = model(x)

    print(f"✓ Forward pass successful")
    print(f"  - Image scores shape: {img_scores.shape}")
    print(f"  - Pixel maps shape: {pixel_maps.shape}")

    assert img_scores.shape == (2, 1), f"Unexpected image scores shape: {img_scores.shape}"
    assert pixel_maps.shape == (2, 1, 256, 256), f"Unexpected pixel maps shape: {pixel_maps.shape}"

    # Test predict method
    anomaly_scores, anomaly_maps = model.predict(x)

    print(f"✓ Predict method successful")
    print(f"  - Anomaly scores shape: {anomaly_scores.shape}")
    print(f"  - Anomaly maps shape: {anomaly_maps.shape}")

    assert anomaly_scores.shape == (2,), f"Unexpected scores shape: {anomaly_scores.shape}"
    assert anomaly_maps.shape == (2, 256, 256), f"Unexpected maps shape: {anomaly_maps.shape}"

    print("✓ All supervised tests passed!\n")
    return True


def test_zeroshot():
    """Test zero-shot Mamba-CLIP variant"""
    print("\n" + "="*80)
    print("Testing Zero-Shot Mamba-CLIP")
    print("="*80)

    from model import MODEL, get_model
    from argparse import Namespace

    # Check registration
    assert 'MambaCLIPZeroShot' in MODEL, "MambaCLIPZeroShot not registered!"
    print("✓ MambaCLIPZeroShot registered successfully")

    # Create model
    try:
        cfg_model = Namespace()
        cfg_model.name = 'MambaCLIPZeroShot'
        cfg_model.kwargs = dict(
            image_size=256,
            patch_size=14,
            embed_dim=384,
            depth=6,
            num_adapters=3,
            use_ss2d=True,
            clip_model_name='ViT-B/16',
            pretrained=False,
            checkpoint_path='',
            strict=True,
        )

        model = get_model(cfg_model)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.cuda() if device == 'cuda' else model
        print(f"✓ Model created on {device}")

    except ImportError as e:
        print(f"⚠ CLIP not available: {e}")
        print("⚠ Skipping zero-shot tests (install with: pip install git+https://github.com/openai/CLIP.git)")
        return False

    # Test forward pass with text prompts
    x = torch.randn(2, 3, 256, 256)
    x = x.cuda() if device == 'cuda' else x

    prompts = ["a photo of a flawless object", "a photo of a defective object"]

    model.eval()
    with torch.no_grad():
        outputs = model(x, prompts)

    print(f"✓ Forward pass with prompts successful")
    print(f"  - Image similarity shape: {outputs['image_similarity'].shape}")
    print(f"  - Pixel maps shape: {outputs['pixel_maps'].shape}")

    assert outputs['image_similarity'].shape == (2, 2), f"Unexpected similarity shape"
    assert outputs['pixel_maps'].shape == (2, 2, 256, 256), f"Unexpected pixel maps shape"

    # Test predict method
    anomaly_scores, anomaly_maps = model.predict(
        x,
        normal_prompt="a photo of a flawless object",
        anomaly_prompt="a photo of a defective object"
    )

    print(f"✓ Predict method successful")
    print(f"  - Anomaly scores shape: {anomaly_scores.shape}")
    print(f"  - Anomaly maps shape: {anomaly_maps.shape}")

    assert anomaly_scores.shape == (2,), f"Unexpected scores shape"
    assert anomaly_maps.shape == (2, 256, 256), f"Unexpected maps shape"

    print("✓ All zero-shot tests passed!\n")
    return True


def test_selfsupervised():
    """Test self-supervised Mamba-CLIP variant"""
    print("\n" + "="*80)
    print("Testing Self-Supervised Mamba-CLIP")
    print("="*80)

    from model import MODEL, get_model
    from argparse import Namespace

    # Check registration
    assert 'MambaCLIPSelfSupervised' in MODEL, "MambaCLIPSelfSupervised not registered!"
    print("✓ MambaCLIPSelfSupervised registered successfully")

    # Create model
    cfg_model = Namespace()
    cfg_model.name = 'MambaCLIPSelfSupervised'
    cfg_model.kwargs = dict(
        image_size=256,
        patch_size=14,
        embed_dim=384,
        encoder_depth=6,
        decoder_depth=3,
        num_adapters=3,
        use_ss2d=True,
        pretrained=False,
        checkpoint_path='',
        strict=True,
    )

    model = get_model(cfg_model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.cuda() if device == 'cuda' else model
    print(f"✓ Model created on {device}")

    # Test forward pass (reconstruction)
    x = torch.randn(2, 3, 256, 256)
    x = x.cuda() if device == 'cuda' else x

    model.eval()
    with torch.no_grad():
        reconstructed = model(x)

    print(f"✓ Forward pass (reconstruction) successful")
    print(f"  - Reconstructed shape: {reconstructed.shape}")

    assert reconstructed.shape == (2, 3, 256, 256), f"Unexpected reconstruction shape: {reconstructed.shape}"

    # Test predict method
    anomaly_scores, anomaly_maps = model.predict(x)

    print(f"✓ Predict method successful")
    print(f"  - Anomaly scores shape: {anomaly_scores.shape}")
    print(f"  - Anomaly maps shape: {anomaly_maps.shape}")

    assert anomaly_scores.shape == (2,), f"Unexpected scores shape: {anomaly_scores.shape}"
    assert anomaly_maps.shape == (2, 256, 256), f"Unexpected maps shape: {anomaly_maps.shape}"

    print("✓ All self-supervised tests passed!\n")
    return True


def test_trainers():
    """Test trainer registration"""
    print("\n" + "="*80)
    print("Testing Trainer Registration")
    print("="*80)

    from trainer import TRAINER

    # Check all three trainers
    trainers = [
        'MambaCLIPTrainer',
        'MambaCLIPZeroShotTrainer',
        'MambaCLIPSelfSupervisedTrainer'
    ]

    for trainer_name in trainers:
        assert trainer_name in TRAINER, f"{trainer_name} not registered!"
        print(f"✓ {trainer_name} registered successfully")

    print("✓ All trainers registered!\n")
    return True


def test_configs():
    """Test config loading"""
    print("\n" + "="*80)
    print("Testing Config Loading")
    print("="*80)

    configs = [
        'configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py',
        'configs/benchmark/mamba_clip/mamba_clip_zeroshot_mvtec.py',
        'configs/benchmark/mamba_clip/mamba_clip_selfsupervised_mvtec.py',
    ]

    for config_path in configs:
        if os.path.exists(config_path):
            print(f"✓ Config exists: {config_path}")
        else:
            print(f"✗ Config missing: {config_path}")
            return False

    print("✓ All configs found!\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MAMBA-CLIP VARIANTS TEST SUITE")
    print("="*80)

    results = {
        'Supervised': False,
        'Zero-Shot': False,
        'Self-Supervised': False,
        'Trainers': False,
        'Configs': False,
    }

    try:
        results['Supervised'] = test_supervised()
    except Exception as e:
        print(f"✗ Supervised test failed: {e}")

    try:
        results['Zero-Shot'] = test_zeroshot()
    except Exception as e:
        print(f"✗ Zero-Shot test failed: {e}")

    try:
        results['Self-Supervised'] = test_selfsupervised()
    except Exception as e:
        print(f"✗ Self-Supervised test failed: {e}")

    try:
        results['Trainers'] = test_trainers()
    except Exception as e:
        print(f"✗ Trainers test failed: {e}")

    try:
        results['Configs'] = test_configs()
    except Exception as e:
        print(f"✗ Configs test failed: {e}")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED! 🎉")
    else:
        print("SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
