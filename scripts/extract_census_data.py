#!/usr/bin/env python3
"""
Extract and save clean Census data locally for efficient reuse
Only keeps the ~0.3% of clean examples instead of downloading 1M+ each time
"""

import json
import logging
from datasets import load_dataset
from data_cleaning import ProductionDataCleaner
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_clean_census_data():
    """Extract clean census data and save locally"""
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    output_file = "data/census_clean.json"
    
    # Check if already exists
    if os.path.exists(output_file):
        logger.info(f"Clean census data already exists at {output_file}")
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
        logger.info(f"Found {len(existing_data)} existing clean examples")
        return existing_data
    
    logger.info("🔄 Downloading and cleaning Census data...")
    logger.info("📊 This will take a few minutes but only needs to be done once")
    
    try:
        # Download full census dataset
        logger.info("Downloading full Census dataset (1M+ examples)...")
        census = load_dataset(
            "csv",
            data_files="https://huggingface.co/datasets/Josephgflowers/CENSUS-NER-Name-Email-Address-Phone/resolve/main/FMCSA_CENSUS1_2016Sep_formatted_output.csv"
        )
        
        # Apply cleaning
        logger.info("Applying data cleaning (this may take a few minutes)...")
        cleaner = ProductionDataCleaner()
        cleaned_data = cleaner.clean_census_data(census['train'])
        
        # Save to local file
        logger.info(f"💾 Saving {len(cleaned_data)} clean examples to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        logger.info("✅ Census data extraction complete!")
        logger.info(f"📁 Clean data saved to: {output_file}")
        logger.info(f"📊 Data reduction: 1,048,575 → {len(cleaned_data)} examples ({len(cleaned_data)/1048575*100:.1f}%)")
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"❌ Failed to extract census data: {e}")
        return []

def load_local_census_data():
    """Load clean census data from local file if available"""
    local_file = "data/census_clean.json"
    
    if not os.path.exists(local_file):
        logger.info("Local census data not found, will download and extract...")
        return extract_clean_census_data()
    
    try:
        logger.info(f"📂 Loading clean census data from {local_file}")
        with open(local_file, 'r') as f:
            data = json.load(f)
        logger.info(f"✅ Loaded {len(data)} clean census examples from local file")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load local census data: {e}")
        logger.info("Falling back to extraction...")
        return extract_clean_census_data()

if __name__ == "__main__":
    print("🗂️  Census Data Extraction Script")
    print("=" * 50)
    
    data = extract_clean_census_data()
    
    if data:
        print(f"\n✅ SUCCESS!")
        print(f"📊 Extracted {len(data)} clean examples")
        print(f"📁 Saved to: data/census_clean.json")
        print(f"💡 Future runs will use this local file (much faster!)")
        
        # Show sample
        if len(data) > 0:
            print(f"\n📝 Sample: {data[0]}")
    else:
        print(f"\n❌ FAILED to extract census data") 