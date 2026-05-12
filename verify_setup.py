#!/usr/bin/env python3
"""
Verify Streamlit app setup and dependencies
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_imports():
    """Check all required imports."""
    print("🔍 Checking dependency imports...")
    
    dependencies = {
        "streamlit": "Streamlit framework",
        "pandas": "Data processing",
        "numpy": "Numerical computation",
        "matplotlib": "Plotting",
        "seaborn": "Statistical visualization",
    }
    
    missing = []
    for pkg, desc in dependencies.items():
        try:
            __import__(pkg)
            print(f"  ✅ {pkg:20} ({desc})")
        except ImportError:
            print(f"  ❌ {pkg:20} ({desc}) - Missing!")
            missing.append(pkg)
    
    return len(missing) == 0

def check_project_files():
    """Check required project files."""
    print("\n🗂️  Checking project files...")
    
    required_files = [
        "config.py",
        "data_loader.py",
        "baseline_run.py",
        "strategies.py",
        "metrics.py",
        "src/app/main.py",
        "data/CLC",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            missing.append(file_path)
    
    return len(missing) == 0

def check_data():
    """Check data files."""
    print("\n📊 Checking data files...")
    
    data_dir = PROJECT_ROOT / "data" / "CLC"
    if not data_dir.exists():
        print(f"  ❌ {data_dir} does not exist!")
        return False
    
    # Check for both .csv and .CSV files
    csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.CSV"))
    if not csv_files:
        print(f"  ❌ No CSV files found in {data_dir}!")
        return False
    
    print(f"  ✅ Found {len(csv_files)} CSV files")
    print(f"     Example: {csv_files[0].name}")
    
    return True

def main():
    """Run all checks."""
    print("=" * 60)
    print("🔧 Streamlit App Setup Check")
    print("=" * 60)
    
    checks = [
        ("Dependencies", check_imports),
        ("Project Files", check_project_files),
        ("Data Files", check_data),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Check Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ Pass" if result else "❌ Fail"
        print(f"{name:20} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! You can start the app:")
        print("   $ streamlit run src/app/main.py")
        print("   Or use: $ ./run_app.sh")
    else:
        print("❌ Some checks failed. Please fix and retry")
        print("\nSolutions:")
        print("1️⃣  Missing dependencies? Run: pip install -r streamlit_requirements.txt")
        print("2️⃣  Missing data? Check if data/CLC/ directory exists")
        print("3️⃣  Missing files? Verify project root file integrity")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
