#!/usr/bin/env python3
"""
Simple test script to verify data loading works in all environments
"""

import logging
from utils import load_conll_with_fallback, load_census_with_fallback
from data_cleaning import ProductionDataCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    label_list = [
        'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 
        'B-MISC', 'I-MISC', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE',
        'B-ADDR', 'I-ADDR'
    ]
    
    logger.info("🧪 Testing CoNLL-2003 loading...")
    conll_data = load_conll_with_fallback(label_list)
    logger.info(f"✅ CoNLL-2003 loaded: {len(conll_data)} examples")
    
    logger.info("🧪 Testing Census loading...")
    cleaner = ProductionDataCleaner()
    census_data = load_census_with_fallback(label_list, cleaner)
    logger.info(f"✅ Census loaded: {len(census_data)} examples")
    
    total_real_data = len(conll_data) + len(census_data)
    logger.info(f"🎯 Total real data: {total_real_data} examples")
    
    if total_real_data > 0:
        logger.info("🟢 SUCCESS: Real data loading works!")
        return True
    else:
        logger.warning("🟡 WARNING: No real data loaded, will use synthetic data only")
        return False

if __name__ == "__main__":
    test_data_loading() 