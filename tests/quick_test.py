#!/usr/bin/env python3
"""
Quick Test Script for SDB-Diffusion-Framework

This script performs basic verification tests to ensure the framework
is correctly installed and configured. It tests:
1. Module imports
2. GPU availability (optional)
3. Basic Random Forest functionality
4. Configuration file loading
5. S3GM wrapper initialization

Usage:
    python tests/quick_test.py

Expected runtime: < 30 seconds
"""

import sys
import os
from pathlib import Path

# Safe stdout encoding handling for Windows cross-platform compatibility
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(text):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_test(name, passed, message=""):
    """Print test result safely across all operating systems and encodings"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status}: {name}")
    if message:
        print(f"       {message}")

def test_imports():
    """Test 1: Verify all required modules can be imported"""
    print_header("Test 1: Module Imports")
    
    tests = []
    
    # Test core Python libraries
    try:
        import numpy as np
        tests.append(("NumPy", True, f"version {np.__version__}"))
    except ImportError as e:
        tests.append(("NumPy", False, str(e)))
    
    try:
        import yaml
        tests.append(("PyYAML", True, ""))
    except ImportError as e:
        tests.append(("PyYAML", False, str(e)))
    
    # Test PyTorch
    try:
        import torch
        tests.append(("PyTorch", True, f"version {torch.__version__}"))
    except ImportError as e:
        tests.append(("PyTorch", False, str(e)))
    
    # Test scikit-learn
    try:
        import sklearn
        tests.append(("scikit-learn", True, f"version {sklearn.__version__}"))
    except ImportError as e:
        tests.append(("scikit-learn", False, str(e)))
    
    # Test project modules (with fallback for GEE dependency)
    try:
        from bathymetry import classic_models
        tests.append(("bathymetry.classic_models", True, ""))
    except ImportError as e:
        tests.append(("bathymetry.classic_models", False, str(e)))
    
    try:
        from bathymetry import preprocessor
        tests.append(("bathymetry.preprocessor", True, ""))
    except ImportError as e:
        if "ee" in str(e):
            tests.append(("bathymetry.preprocessor (GEE warning)", True, "Note: 'ee' module optional for offline testing"))
        else:
            tests.append(("bathymetry.preprocessor", False, str(e)))
    
    try:
        from bathymetry import s3gm_wrapper
        tests.append(("bathymetry.s3gm_wrapper", True, ""))
    except ImportError as e:
        if "ee" in str(e):
            tests.append(("bathymetry.s3gm_wrapper (GEE warning)", True, "Note: 'ee' module optional for offline testing"))
        else:
            tests.append(("bathymetry.s3gm_wrapper", False, str(e)))
    
    try:
        from bathymetry import s3gm_config
        tests.append(("bathymetry.s3gm_config", True, ""))
    except ImportError as e:
        tests.append(("bathymetry.s3gm_config", False, str(e)))
    
    try:
        from bathymetry import utils
        tests.append(("bathymetry.utils", True, ""))
    except ImportError as e:
        if "ee" in str(e):
            tests.append(("bathymetry.utils (GEE warning)", True, "Note: 'ee' module optional for offline testing"))
        else:
            tests.append(("bathymetry.utils", False, str(e)))
    
    # Print results
    for name, passed, message in tests:
        print_test(name, passed, message)
    
    return all(passed for _, passed, _ in tests)

def test_gpu():
    """Test 2: Check GPU availability (optional)"""
    print_header("Test 2: GPU Availability (Optional)")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_test("CUDA GPU", True, 
                      f"{device_name}, {memory_gb:.1f} GB memory")
            return True
        else:
            print_test("CUDA GPU", False, 
                      "No GPU detected. Framework will run on CPU (slower)")
            print("       Note: GPU with >=6GB VRAM recommended for S3GM")
            return True  # Not a critical failure
    except Exception as e:
        print_test("GPU Check", False, str(e))
        return True  # Not a critical failure

def test_random_forest():
    """Test 3: Test basic Random Forest functionality"""
    print_header("Test 3: Random Forest Functionality")
    
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.rand(100, 7)  # 7 features as in the framework
        y_train = np.random.rand(100) * 50  # Depths 0-50m
        X_test = np.random.rand(20, 7)
        
        # Train RF model
        rf = RandomForestRegressor(
            n_estimators=10,  # Small for quick test
            max_depth=5,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # Make predictions
        predictions = rf.predict(X_test)
        
        # Verify predictions are reasonable
        if len(predictions) == 20 and np.all(np.isfinite(predictions)):
            print_test("RF Training & Prediction", True, 
                      f"Predicted depths: {predictions[0]:.2f}m (example)")
            return True
        else:
            print_test("RF Training & Prediction", False, 
                      "Invalid predictions")
            return False
            
    except Exception as e:
        print_test("RF Training & Prediction", False, str(e))
        return False

def test_config_loading():
    """Test 4: Test configuration file loading"""
    print_header("Test 4: Configuration File Loading")
    
    try:
        import yaml
        
        # Test classic models config
        config_path = Path(__file__).parent.parent / "configs" / "classic_models.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'rf_params' in config:
                n_est = config['rf_params'].get('model_params', {}).get('n_estimators', 'N/A')
                print_test("classic_models.yaml", True, 
                          f"n_estimators={n_est}")
            else:
                print_test("classic_models.yaml", False, 
                          "Missing 'rf_params' section")
                return False
        else:
            print_test("classic_models.yaml", False, "File not found")
            return False
        
        # Test S3GM config
        config_path = Path(__file__).parent.parent / "configs" / "s3gm_default.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'beta_max' in config and 'sampling' in config:
                alpha = config['sampling'].get('alpha', 'N/A')
                beta_max = config.get('beta_max', 'N/A')
                print_test("s3gm_default.yaml", True, 
                          f"alpha={alpha}, beta_max={beta_max}")
            else:
                print_test("s3gm_default.yaml", False, 
                          "Missing required parameters (beta_max or sampling)")
                return False
        else:
            print_test("s3gm_default.yaml", False, "File not found")
            return False
        
        return True
        
    except Exception as e:
        print_test("Configuration Loading", False, str(e))
        return False

def test_s3gm_wrapper():
    """Test 5: Test S3GM wrapper initialization"""
    print_header("Test 5: S3GM Wrapper Initialization")
    
    try:
        from bathymetry.s3gm_config import S3GMConfig
        
        # Create minimal config
        config = S3GMConfig(
            num_components=5,
            image_size=64,
            num_frames=6,
            beta_max=1000
        )
        
        print_test("S3GMConfig Creation", True, 
                  f"components={config.num_components}, size={config.image_size}x{config.image_size}")
        
        # Note: We don't initialize the full S3GM model here as it requires
        # significant memory and time. The config test is sufficient for quick verification.
        print("       Note: Full S3GM model initialization skipped (requires ~8 hours pre-training)")
        
        return True
        
    except Exception as e:
        print_test("S3GM Wrapper", False, str(e))
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  SDB-Diffusion-Framework Quick Test")
    print("="*60)
    print("\nThis test verifies basic framework functionality.")
    print("Expected runtime: < 30 seconds\n")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("GPU Availability", test_gpu()))
    results.append(("Random Forest", test_random_forest()))
    results.append(("Configuration Loading", test_config_loading()))
    results.append(("S3GM Wrapper", test_s3gm_wrapper()))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[PASS] All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("  1. Update data_acquisition_preprocessing.py with your GEE project ID")
        print("  2. Prepare your input data (Sentinel-2, GEBCO, nautical charts)")
        print("  3. Run: python run_bathymetry.py --stage 1")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("  - Ensure conda environment is activated")
        print("  - Reinstall dependencies: conda env update -f environment.yml")
        print("  - Check GitHub issues: https://github.com/sxlong2022/SDB-Diffusion-Framework/issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())