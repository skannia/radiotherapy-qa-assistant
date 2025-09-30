#!/usr/bin/env python3
"""
Quick validation script for medical physicists to test the QA system.
This validates that all components can import and run basic functions.
"""

import sys
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import openpyxl
        print("✓ Core dependencies imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_qa_functions():
    """Test that QA functions work with sample data."""
    print("\nTesting QA functions...")
    try:
        import pandas as pd
        from qa_checks import tg142_check, tg224_check, electron_beam_check, tps_check
        
        # Create minimal test data
        test_data = pd.DataFrame({
            'output': [100.0, 99.0, 101.0],
            'flatness': [100.0, 102.0, 98.0],
            'symmetry': [100.0, 101.0, 99.0],
            'dose_uniformity': [100.0, 101.0, 99.0],
            'beam_energy': [6.0, 6.1, 5.9],
            'electron_output': [100.0, 102.0, 98.0],
            'PDD': [100.0, 99.0, 101.0],
            'planned_vs_measured': [100.0, 98.0, 102.0],
            'gamma_index': [98.0, 96.0, 97.0]
        })
        
        # Test each QA function
        results = []
        results.extend(tg142_check(test_data))
        results.extend(tg224_check(test_data))
        results.extend(electron_beam_check(test_data))
        results.extend(tps_check(test_data))
        
        # Verify results structure
        if results and all('status' in r and 'parameter' in r for r in results):
            print(f"✓ QA functions working - {len(results)} results generated")
            
            # Count pass/fail
            passes = sum(1 for r in results if r['status'] == 'PASS')
            fails = sum(1 for r in results if r['status'] == 'FAIL')
            print(f"  Results: {passes} PASS, {fails} FAIL")
            return True
        else:
            print("✗ QA functions returned unexpected format")
            return False
            
    except Exception as e:
        print(f"✗ QA function error: {e}")
        traceback.print_exc()
        return False

def test_app_utils():
    """Test app utility functions."""
    print("\nTesting app utilities...")
    try:
        from app_utils import normalize_uploaded_df, run_qa_and_pivot
        print("✓ App utilities imported successfully")
        return True
    except Exception as e:
        print(f"✗ App utils error: {e}")
        return False

def test_dose_calculator():
    """Test MU calculator function."""
    print("\nTesting dose calculator...")
    try:
        # Import from dose_calculator_streamlit.py
        import sys
        sys.path.append('.')
        
        # Read the file and extract the function
        with open('dose_calculator_streamlit.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Execute the file to get the functions
        exec_globals = {}
        exec(content, exec_globals)
        
        calculate_mu = exec_globals['calculate_mu']
        compare_with_tps = exec_globals['compare_with_tps']
        
        # Test calculation
        mu = calculate_mu(200.0, 1.0, 10.0, "10x10")
        deviation, within_tol = compare_with_tps(mu, 300.0)
        
        print(f"✓ MU Calculator working - Calculated: {mu:.2f}, Test deviation: {deviation:.2f}%")
        return True
        
    except Exception as e:
        print(f"✗ Dose calculator error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("AI-Assisted QA System Validation")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_qa_functions, 
        test_app_utils,
        test_dose_calculator
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nThe system is ready for medical physics testing!")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Upload your CSV data")
        print("3. Select appropriate QA modules") 
        print("4. Review results and download reports")
        return True
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total})")
        print("\nPlease address the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)