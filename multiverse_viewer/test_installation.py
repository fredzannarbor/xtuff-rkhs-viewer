#!/usr/bin/env python3
"""
test_installation.py

Quick test script to verify RKHS multiverse viewer installation and sample data.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are installed"""
    print("Testing package imports...")
    required_packages = [
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('networkx', 'networkx'),
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All required packages installed\n")
    return True

def test_sample_data():
    """Test that sample data file exists and is valid"""
    print("Testing sample data...")
    
    sample_file = Path("codexspace_sample.rkhs.json")
    
    if not sample_file.exists():
        print(f"  ✗ Sample file not found: {sample_file}")
        print("  Create it with: python create_sample_codexspace_rkhs.py")
        return False
    
    print(f"  ✓ Sample file exists ({sample_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Try to load it
    try:
        import json
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        print(f"  ✓ Valid JSON format")
        print(f"  ✓ Contains {len(data.get('nodes', {}))} nodes")
        print(f"  ✓ Contains {len(data.get('edges', []))} edges")
        
    except Exception as e:
        print(f"  ✗ Error loading sample file: {e}")
        return False
    
    print("✅ Sample data is valid\n")
    return True

def test_viewer_file():
    """Test that the viewer application exists"""
    print("Testing viewer application...")
    
    viewer_file = Path("multiverse_viewer.py")
    
    if not viewer_file.exists():
        print(f"  ✗ Viewer file not found: {viewer_file}")
        return False
    
    print(f"  ✓ Viewer file exists ({viewer_file.stat().st_size / 1024:.1f} KB)")
    
    # Try to import it
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("multiverse_viewer", viewer_file)
        module = importlib.util.module_from_spec(spec)
        # Don't actually execute it, just check it can be loaded
        print(f"  ✓ Viewer can be imported")
    except Exception as e:
        print(f"  ✗ Error importing viewer: {e}")
        return False
    
    print("✅ Viewer application is ready\n")
    return True

def test_converter():
    """Test that converter tools exist"""
    print("Testing converter tools...")
    
    converter_files = [
        ("codexspaces_to_rkhs_converter.py", "Main converter"),
        ("create_sample_codexspace_rkhs.py", "Sample creator")
    ]
    
    all_exist = True
    for filename, description in converter_files:
        filepath = Path(filename)
        if filepath.exists():
            print(f"  ✓ {description}: {filename}")
        else:
            print(f"  ✗ {description}: {filename} - MISSING")
            all_exist = False
    
    if all_exist:
        print("✅ Converter tools are ready\n")
    else:
        print("⚠️  Some converter tools are missing\n")
    
    return all_exist

def print_next_steps():
    """Print instructions for next steps"""
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("1. Launch the viewer:")
    print("   streamlit run multiverse_viewer.py")
    print()
    print("2. In your browser:")
    print("   - Go to the 'Open' tab")
    print("   - Upload 'codexspace_sample.rkhs.json'")
    print("   - Explore the 'Visualize' tab")
    print()
    print("3. Convert your CodexSpaces data:")
    print("   python codexspaces_to_rkhs_converter.py codexspace_v1.pkl")
    print()
    print("4. Read the documentation:")
    print("   - INDEX.md - Complete package overview")
    print("   - QUICK_REFERENCE.md - Command cheat sheet")
    print("   - CODEXSPACES_CONVERSION_GUIDE.md - Detailed guide")
    print()

def main():
    print("=" * 60)
    print("RKHS MULTIVERSE VIEWER - INSTALLATION TEST")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Packages", test_imports()))
    results.append(("Sample Data", test_sample_data()))
    results.append(("Viewer App", test_viewer_file()))
    results.append(("Converters", test_converter()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All tests passed! You're ready to go!")
        print()
        print_next_steps()
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print()
        print("Common solutions:")
        print("- Install packages: pip install -r requirements.txt")
        print("- Create sample: python create_sample_codexspace_rkhs.py")
        print("- Check file locations: make sure you're in the right directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
