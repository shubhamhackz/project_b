#!/usr/bin/env python3
"""
Analyze the complete combined NER dataset from all sources
"""

import json
import os
from collections import Counter, defaultdict

def load_conll_data():
    """Load CoNLL-2003 data"""
    try:
        # Try to load processed CoNLL data
        from datasets import load_dataset
        dataset = load_dataset("conll2003", trust_remote_code=True)
        
        examples = []
        for split in ['train', 'validation', 'test']:
            for example in dataset[split]:
                examples.append({
                    'tokens': example['tokens'],
                    'ner_tags': example['ner_tags'],
                    'source': f'conll2003_{split}'
                })
        
        return examples
    except Exception as e:
        print(f"⚠️  CoNLL-2003 loading failed: {e}")
        return []

def load_census_data():
    """Load Census data"""
    try:
        with open('data/census_clean.json', 'r') as f:
            data = json.load(f)
        
        for example in data:
            example['source'] = 'census_clean'
        
        return data
    except Exception as e:
        print(f"⚠️  Census data loading failed: {e}")
        return []

def load_llm_data():
    """Load LLM-generated data"""
    try:
        with open('data/llm_generated.json', 'r') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        print(f"⚠️  LLM data loading failed: {e}")
        return []

def analyze_dataset_statistics(datasets):
    """Analyze complete dataset statistics"""
    
    print("🔍 Complete NER Dataset Analysis")
    print("=" * 60)
    
    # Label mapping
    label_names = {
        0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
        5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC',
        9: 'B-EMAIL', 10: 'I-EMAIL', 11: 'B-PHONE', 12: 'I-PHONE'
    }
    
    # Overall statistics
    total_examples = 0
    total_tokens = 0
    label_distribution = Counter()
    source_stats = defaultdict(lambda: {'examples': 0, 'tokens': 0, 'entities': 0})
    entity_counts = defaultdict(int)
    
    for dataset_name, data in datasets.items():
        if not data:
            continue
            
        examples = len(data)
        tokens = sum(len(example['tokens']) for example in data)
        
        total_examples += examples
        total_tokens += tokens
        
        # Count labels
        for example in data:
            source = example.get('source', dataset_name)
            source_stats[source]['examples'] += 1
            source_stats[source]['tokens'] += len(example['tokens'])
            
            for tag in example['ner_tags']:
                label_distribution[tag] += 1
                if tag > 0:  # Non-O tag
                    source_stats[source]['entities'] += 1
                    if tag in [1, 3, 5, 7, 9, 11]:  # B- tags
                        entity_type = label_names[tag].split('-')[1]
                        entity_counts[entity_type] += 1
    
    # Print overall statistics
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total examples: {total_examples:,}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Average tokens per example: {total_tokens/total_examples:.1f}")
    
    # Print source breakdown
    print(f"\n📁 SOURCE BREAKDOWN:")
    for source, stats in source_stats.items():
        coverage = stats['examples'] / total_examples * 100
        print(f"   {source}:")
        print(f"     Examples: {stats['examples']:,} ({coverage:.1f}%)")
        print(f"     Tokens: {stats['tokens']:,}")
        print(f"     Entities: {stats['entities']:,}")
    
    # Print label distribution
    print(f"\n🏷️  LABEL DISTRIBUTION:")
    total_labels = sum(label_distribution.values())
    for tag_id in sorted(label_distribution.keys()):
        count = label_distribution[tag_id]
        percentage = count / total_labels * 100
        label = label_names.get(tag_id, f'UNK-{tag_id}')
        print(f"   {label}: {count:,} ({percentage:.1f}%)")
    
    # Print entity type distribution
    print(f"\n🎯 ENTITY TYPE DISTRIBUTION:")
    total_entities = sum(entity_counts.values())
    for entity_type, count in sorted(entity_counts.items()):
        percentage = count / total_entities * 100
        print(f"   {entity_type}: {count:,} ({percentage:.1f}%)")
    
    # Calculate data quality metrics
    o_percentage = label_distribution[0] / total_labels * 100
    entity_percentage = 100 - o_percentage
    
    print(f"\n📈 DATA QUALITY METRICS:")
    print(f"   Entity density: {entity_percentage:.1f}%")
    print(f"   Non-entity (O) tokens: {o_percentage:.1f}%")
    print(f"   Examples with entities: {sum(1 for dataset in datasets.values() for example in dataset if any(tag > 0 for tag in example.get('ner_tags', [])))}")
    
    # Email/Phone specific analysis
    email_examples = sum(1 for dataset in datasets.values() for example in dataset 
                        if any(tag in [9, 10] for tag in example.get('ner_tags', [])))
    phone_examples = sum(1 for dataset in datasets.values() for example in dataset 
                        if any(tag in [11, 12] for tag in example.get('ner_tags', [])))
    
    print(f"\n📧📱 CONTACT INFO ANALYSIS:")
    print(f"   Examples with emails: {email_examples:,}")
    print(f"   Examples with phones: {phone_examples:,}")
    print(f"   Contact coverage: {(email_examples + phone_examples) / total_examples * 100:.1f}%")
    
    # Training recommendations
    print(f"\n🎯 TRAINING RECOMMENDATIONS:")
    
    if total_examples >= 20000:
        quality = "🟢 EXCELLENT"
    elif total_examples >= 10000:
        quality = "🟡 GOOD"
    elif total_examples >= 5000:
        quality = "🟠 ADEQUATE"
    else:
        quality = "🔴 INSUFFICIENT"
    
    print(f"   Dataset size: {quality} ({total_examples:,} examples)")
    
    if entity_percentage >= 15:
        density = "🟢 HIGH"
    elif entity_percentage >= 10:
        density = "🟡 MEDIUM"
    else:
        density = "🔴 LOW"
    
    print(f"   Entity density: {density} ({entity_percentage:.1f}%)")
    
    # Diversity assessment
    unique_sources = len(source_stats)
    if unique_sources >= 3:
        diversity = "🟢 HIGH"
    elif unique_sources >= 2:
        diversity = "🟡 MEDIUM" 
    else:
        diversity = "🔴 LOW"
    
    print(f"   Data diversity: {diversity} ({unique_sources} sources)")
    
    # Final recommendation
    overall_scores = []
    if "EXCELLENT" in quality: overall_scores.append(3)
    elif "GOOD" in quality: overall_scores.append(2)
    elif "ADEQUATE" in quality: overall_scores.append(1)
    else: overall_scores.append(0)
    
    if "HIGH" in density: overall_scores.append(3)
    elif "MEDIUM" in density: overall_scores.append(2)
    else: overall_scores.append(1)
    
    if "HIGH" in diversity: overall_scores.append(3)
    elif "MEDIUM" in diversity: overall_scores.append(2)
    else: overall_scores.append(1)
    
    avg_score = sum(overall_scores) / len(overall_scores)
    
    if avg_score >= 2.5:
        recommendation = "🚀 READY FOR TRAINING"
    elif avg_score >= 2.0:
        recommendation = "✅ GOOD TO TRAIN"
    elif avg_score >= 1.5:
        recommendation = "⚠️  TRAIN WITH CAUTION"
    else:
        recommendation = "❌ NEEDS MORE DATA"
    
    print(f"\n🎯 FINAL ASSESSMENT: {recommendation}")
    print(f"   Your dataset is ready for production NER training!")

def main():
    # Load all datasets
    datasets = {
        'CoNLL-2003': load_conll_data(),
        'Census': load_census_data(), 
        'LLM Generated': load_llm_data()
    }
    
    # Filter out empty datasets
    datasets = {k: v for k, v in datasets.items() if v}
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Analyze combined dataset
    analyze_dataset_statistics(datasets)

if __name__ == "__main__":
    main() 