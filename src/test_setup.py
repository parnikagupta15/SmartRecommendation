"""
Test Setup Script
Verifies all installations and configurations are correct
"""

import sys
import pandas as pd
import numpy as np
# import torch
# from config.config import config

def test_python_version():
    """Test Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3 and version.minor >= 9, "Python 3.9+ required"

def test_imports():
    """Test critical imports"""
    print("✓ pandas imported successfully")
    print(f"  - pandas version: {pd.__version__}")
    print("✓ numpy imported successfully")
    print(f"  - numpy version: {np.__version__}")

def test_data_generation():
    """Test data generation"""
    print("\n✓ Running quick data generation test...")
    from data_collection import DataCollector
    
    collector = DataCollector(data_dir="data/test")
    df_prod, df_users, df_inter = collector.generate_synthetic_data(
        num_users=100, 
        num_products=50, 
        num_interactions=500
    )
    
    print(f"  - Generated {len(df_prod)} products")
    print(f"  - Generated {len(df_users)} users")
    print(f"  - Generated {len(df_inter)} interactions")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 SmartRecommendation Setup Test")
    print("=" * 60 + "\n")
    
    try:
        test_python_version()
        print()
        test_imports()
        print()
        test_data_generation()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Setup is complete.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)