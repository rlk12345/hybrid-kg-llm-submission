#!/usr/bin/env python3
"""
Quick verification script to test if the project setup is correct.
Run this after installation to verify everything works.
"""
import sys
import os

def check_imports():
    """Test if all required modules can be imported."""
    print("Checking imports...")
    try:
        from src.config import HybridConfig
        print("  ✓ src.config imported successfully")
        
        from src.kg_data import read_triples_jsonl
        print("  ✓ src.kg_data imported successfully")
        
        from src.hybrid_dpo import train_hybrid_dpo
        print("  ✓ src.hybrid_dpo imported successfully")
        
        from src.sns_ranker import SNSSimilarityRanker
        print("  ✓ src.sns_ranker imported successfully")
        
        from src.kg_visualize import render_kg
        print("  ✓ src.kg_visualize imported successfully")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("  Make sure PYTHONPATH is set to project root, or run scripts from project root")
        return False

def check_dependencies():
    """Test if required packages are installed."""
    print("\nChecking dependencies...")
    required = [
        'torch', 'transformers', 'datasets', 'trl', 
        'networkx', 'matplotlib', 'sentence_transformers'
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} installed")
        except ImportError:
            print(f"  ✗ {pkg} missing")
            missing.append(pkg)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    return True

def check_data_files():
    """Check if sample data files exist."""
    print("\nChecking data files...")
    data_files = [
        'data/sample_triples.jsonl',
        'data/entity_texts.jsonl'
    ]
    all_exist = True
    for fpath in data_files:
        if os.path.exists(fpath):
            print(f"  ✓ {fpath} exists")
        else:
            print(f"  ✗ {fpath} missing")
            all_exist = False
    return all_exist

def check_graphviz():
    """Check if Graphviz is installed."""
    print("\nChecking Graphviz...")
    try:
        import subprocess
        result = subprocess.run(['dot', '-V'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("  ✓ Graphviz installed")
            return True
        else:
            print("  ✗ Graphviz not found in PATH")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ✗ Graphviz not found")
        print("  Install: brew install graphviz (macOS) or apt-get install graphviz (Linux)")
        return False

def main():
    print("=" * 60)
    print("Hybrid-KG-LLM-Project Setup Verification")
    print("=" * 60)
    
    # Add current directory to path if not already there
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    results = []
    results.append(("Imports", check_imports()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Data Files", check_data_files()))
    results.append(("Graphviz", check_graphviz()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed! Setup is correct.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

