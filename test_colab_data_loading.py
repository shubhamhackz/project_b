#!/usr/bin/env python3
"""
Test script to verify data loading works in Google Colab
"""

import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_colab_data_loading():
    """Test all data loading functions for Colab compatibility"""
    
    print("🔍 Testing Google Colab Data Loading Compatibility")
    print("=" * 60)
    
    # Import our modules
    try:
        from utils import (
            load_conll_with_fallback,
            load_llm_generated_data,
            load_census_with_fallback,
            prepare_combined_dataset
        )
        from data_cleaning import ProductionDataCleaner
        print("✅ All modules imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test file existence
    print("\n📁 Checking data files...")
    files_to_check = [
        'data/conll2003/train.txt',
        'data/census_clean.json', 
        'data/llm_generated.json'
    ]
    
    files_found = 0
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"   Contains {len(data)} examples")
                    files_found += 1
                except Exception as e:
                    print(f"   ⚠️  Error reading file: {e}")
            else:
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    print(f"   Contains {len(lines)} lines")
                    files_found += 1
                except Exception as e:
                    print(f"   ⚠️  Error reading file: {e}")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Test individual loading functions
    print("\n🔄 Testing individual data loading functions...")
    
    # Test LLM data loading
    try:
        llm_data = load_llm_generated_data()
        print(f"✅ LLM data: {len(llm_data)} examples loaded")
    except Exception as e:
        print(f"❌ LLM data loading failed: {e}")
        llm_data = []
    
    # Test CoNLL loading
    try:
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE']
        conll_data = load_conll_with_fallback(label_list)
        print(f"✅ CoNLL-2003 data: {len(conll_data)} examples loaded")
    except Exception as e:
        print(f"❌ CoNLL-2003 loading failed: {e}")
        conll_data = []
    
    # Test Census loading
    try:
        cleaner = ProductionDataCleaner()
        census_data = load_census_with_fallback(label_list, cleaner)
        print(f"✅ Census data: {len(census_data)} examples loaded")
    except Exception as e:
        print(f"❌ Census loading failed: {e}")
        census_data = []
    
    # Test combined dataset preparation
    print("\n🔄 Testing combined dataset preparation...")
    try:
        combined_data = prepare_combined_dataset(label_list, cleaner, synthetic_count=100)
        print(f"✅ Combined dataset: {len(combined_data)} examples")
        
        # Analyze the combined data
        source_counts = {}
        for example in combined_data:
            source = example.get('source', 'synthetic')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("📊 Data source breakdown:")
        for source, count in source_counts.items():
            print(f"   {source}: {count} examples")
            
    except Exception as e:
        print(f"❌ Combined dataset preparation failed: {e}")
        return False
    
    # Final assessment
    print("\n🎯 FINAL ASSESSMENT:")
    
    if len(combined_data) >= 1000:
        print("✅ Dataset size: EXCELLENT (1000+ examples)")
        status = "🚀 READY FOR TRAINING"
    elif len(combined_data) >= 500:
        print("✅ Dataset size: GOOD (500+ examples)")
        status = "✅ READY FOR TRAINING"
    elif len(combined_data) >= 100:
        print("⚠️  Dataset size: MINIMAL (100+ examples)")
        status = "⚠️  CAN TRAIN WITH LIMITATIONS"
    else:
        print("❌ Dataset size: INSUFFICIENT")
        status = "❌ NEEDS MORE DATA"
    
    print(f"\n{status}")
    
    if "READY" in status:
        print("\n🎉 Google Colab data loading is working perfectly!")
        print("You can now run: python train.py")
        return True
    else:
        print("\n📋 Recommendations:")
        if files_found == 0:
            print("1. Upload data files to the data/ directory")
            print("2. Or run: python generate_llm_corpus.py")
        print("3. Check the COLAB_SETUP.md for detailed instructions")
        return False

if __name__ == "__main__":
    success = test_colab_data_loading()
    exit(0 if success else 1) 