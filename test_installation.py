"""
Installation Test Script
验证所有依赖和模块是否正确安装
"""

import sys
import importlib
from typing import List, Tuple


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    检查模块是否可以导入

    Args:
        module_name: 要导入的模块名
        package_name: 显示的包名 (如果与模块名不同)
    Returns:
        (success, message): 成功标志和消息
    """
    try:
        importlib.import_module(module_name)
        version = ""
        if hasattr(importlib.import_module(module_name), '__version__'):
            version = f" v{importlib.import_module(module_name).__version__}"
        return True, f"✓ {package_name or module_name}{version}"
    except ImportError as e:
        return False, f"✗ {package_name or module_name}: {str(e)}"


def test_basic_imports():
    """测试基础依赖"""
    print("\n" + "=" * 60)
    print("TESTING BASIC DEPENDENCIES")
    print("=" * 60)

    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('tqdm', 'tqdm'),
    ]

    results = []
    for module, name in packages:
        success, msg = check_import(module, name)
        results.append(success)
        print(msg)

    return all(results)


def test_diffusion_imports():
    """测试扩散模型相关依赖"""
    print("\n" + "=" * 60)
    print("TESTING DIFFUSION MODEL DEPENDENCIES")
    print("=" * 60)

    packages = [
        ('diffusers', 'Diffusers'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
    ]

    results = []
    for module, name in packages:
        success, msg = check_import(module, name)
        results.append(success)
        print(msg)

    return all(results)


def test_project_imports():
    """测试项目模块"""
    print("\n" + "=" * 60)
    print("TESTING PROJECT MODULES")
    print("=" * 60)

    modules = [
        'models.ns_diff',
        'models.baselines',
        'data.datasets',
        'evaluation.metrics',
        'evaluation.visualization',
    ]

    results = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
            results.append(True)
        except ImportError as e:
            print(f"✗ {module}: {str(e)}")
            results.append(False)

    return all(results)


def test_model_instantiation():
    """测试模型实例化"""
    print("\n" + "=" * 60)
    print("TESTING MODEL INSTANTIATION")
    print("=" * 60)

    try:
        import torch
        from models.ns_diff_error import NSDiff
        from models.baselines import build_model

        # 测试NS-Diff
        print("Creating NS-Diff model...")
        model = NSDiff(
            num_concepts=8,
            num_classes=2,
            latent_dim=512
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ NS-Diff created successfully ({num_params / 1e6:.2f}M parameters)")

        # 测试前向传播
        print("Testing forward pass...")
        x = torch.randn(2, 3, 256, 256)
        outputs = model(x)
        print(f"✓ Forward pass successful")
        print(f"  - Predictions shape: {outputs['predictions'].shape}")
        print(f"  - Concepts shape: {outputs['concepts'].shape}")

        # 测试反事实生成
        print("Testing counterfactual generation...")
        x_cf, info = model.generate_counterfactual(x[:1], target_concept_idx=0, target_value=1.0)
        print(f"✓ Counterfactual generation successful")
        print(f"  - Generated image shape: {x_cf.shape}")

        # 测试基线模型
        print("\nTesting baseline models...")
        baselines = ['resnet50', 'standard_cbm', 'posthoc_cbm', 'disdiff_fnnc']
        for baseline in baselines:
            try:
                model = build_model(baseline, num_concepts=8, num_classes=2)
                print(f"✓ {baseline} created successfully")
            except Exception as e:
                print(f"✗ {baseline}: {str(e)}")
                return False

        return True

    except Exception as e:
        print(f"✗ Model instantiation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载 (需要实际数据文件)"""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)

    try:
        from data.datasets import Shapes3DDataset, CelebAHQDataset
        import torch
        import numpy as np

        # 创建虚拟数据测试
        print("Testing dataset classes...")

        # 注意: 这里只测试类定义,不测试实际数据加载
        print("✓ Dataset classes imported successfully")
        print("  Note: Actual data loading requires dataset files")

        return True

    except Exception as e:
        print(f"✗ Data loading test failed: {str(e)}")
        return False


def test_metrics():
    """测试评估指标"""
    print("\n" + "=" * 60)
    print("TESTING EVALUATION METRICS")
    print("=" * 60)

    try:
        from evaluation.metrics import compute_mig, compute_mutual_information
        import numpy as np

        # 创建测试数据
        print("Testing MIG computation...")
        n_samples = 1000
        n_concepts = 3
        n_factors = 3

        # 模拟完美解耦
        factors = np.random.randint(0, 10, size=(n_samples, n_factors))
        concepts = np.zeros((n_samples, n_concepts))
        for i in range(n_concepts):
            concepts[:, i] = factors[:, i] / 10.0

        mig = compute_mig(concepts, factors)
        print(f"✓ MIG computation successful: {mig:.4f}")

        if mig > 0.5:
            print("  MIG value looks reasonable for disentangled data")
        else:
            print("  Warning: MIG value lower than expected")

        return True

    except Exception as e:
        print(f"✗ Metrics test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """测试可视化"""
    print("\n" + "=" * 60)
    print("TESTING VISUALIZATION")
    print("=" * 60)

    try:
        from evaluation.visualization import plot_performance_comparison
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端

        print("Testing performance comparison plot...")
        results = {
            'Model A': {'accuracy': 90.0, 'mig': 0.5, 'isr': 80.0},
            'Model B': {'accuracy': 88.0, 'mig': 0.6, 'isr': 75.0}
        }

        plot_performance_comparison(results, save_path='/tmp/test_plot.png')
        print("✓ Visualization test successful")

        return True

    except Exception as e:
        print(f"✗ Visualization test failed: {str(e)}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("NS-DIFF INSTALLATION TEST SUITE")
    print("=" * 80)

    tests = [
        ("Basic Dependencies", test_basic_imports),
        ("Diffusion Dependencies", test_diffusion_imports),
        ("Project Modules", test_project_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Data Loading", test_data_loading),
        ("Evaluation Metrics", test_metrics),
        ("Visualization", test_visualization),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {str(e)}")
            results[test_name] = False

    # 总结
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<40}{status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print("\n" + "=" * 80)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("🎉 All tests passed! Installation is complete.")
        print("\nYou can now:")
        print("  1. Run training: python train.py --help")
        print("  2. Run experiments: python experiments/run_all_experiments.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check PYTHONPATH includes project root")
        print("  3. Ensure CUDA is properly installed (if using GPU)")

    print("=" * 80 + "\n")

    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)