#!/usr/bin/env python3
"""
Analyze the quality of email and phone data in census_clean.json
"""

import json
import logging
from collections import Counter, defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_census_data():
    """Load the clean census data"""
    try:
        with open("data/census_clean.json", 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load census data: {e}")
        return []

def analyze_email_phone_quality(data, sample_size=50):
    """Analyze email and phone data quality"""
    
    # Label mappings
    label_names = {
        0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
        5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC',
        9: 'B-EMAIL', 10: 'I-EMAIL', 11: 'B-PHONE', 12: 'I-PHONE'
    }
    
    # Statistics
    stats = {
        'total_examples': len(data),
        'email_examples': 0,
        'phone_examples': 0,
        'both_examples': 0,
        'email_entities': [],
        'phone_entities': [],
        'email_patterns': Counter(),
        'phone_patterns': Counter(),
        'bio_issues': [],
        'quality_issues': []
    }
    
    print("🔍 Analyzing Census Email/Phone Data Quality")
    print("=" * 60)
    
    # Sample analysis
    sample_data = data[:sample_size] if len(data) > sample_size else data
    
    for i, example in enumerate(sample_data):
        tokens = example['tokens']
        tags = example['ner_tags']
        
        # Find email and phone entities
        has_email = any(tag in [9, 10] for tag in tags)
        has_phone = any(tag in [11, 12] for tag in tags)
        
        if has_email:
            stats['email_examples'] += 1
        if has_phone:
            stats['phone_examples'] += 1
        if has_email and has_phone:
            stats['both_examples'] += 1
            
        # Extract entities
        current_entity = []
        current_type = None
        
        for j, (token, tag) in enumerate(zip(tokens, tags)):
            tag_name = label_names.get(tag, f'UNK-{tag}')
            
            if tag in [9, 11]:  # B-EMAIL or B-PHONE
                # Save previous entity if exists
                if current_entity and current_type:
                    entity_text = ' '.join(current_entity)
                    if current_type == 'EMAIL':
                        stats['email_entities'].append(entity_text)
                        stats['email_patterns'][get_email_pattern(entity_text)] += 1
                    elif current_type == 'PHONE':
                        stats['phone_entities'].append(entity_text)
                        stats['phone_patterns'][get_phone_pattern(entity_text)] += 1
                
                # Start new entity
                current_entity = [token]
                current_type = 'EMAIL' if tag == 9 else 'PHONE'
                
            elif tag in [10, 12]:  # I-EMAIL or I-PHONE
                expected_type = 'EMAIL' if tag == 10 else 'PHONE'
                if current_type == expected_type:
                    current_entity.append(token)
                else:
                    # BIO tagging issue
                    stats['bio_issues'].append({
                        'example_idx': i,
                        'position': j,
                        'issue': f'I-{expected_type} without preceding B-{expected_type}',
                        'context': ' '.join(tokens[max(0, j-3):j+4])
                    })
            else:
                # Save current entity if exists
                if current_entity and current_type:
                    entity_text = ' '.join(current_entity)
                    if current_type == 'EMAIL':
                        stats['email_entities'].append(entity_text)
                        stats['email_patterns'][get_email_pattern(entity_text)] += 1
                    elif current_type == 'PHONE':
                        stats['phone_entities'].append(entity_text)
                        stats['phone_patterns'][get_phone_pattern(entity_text)] += 1
                current_entity = []
                current_type = None
        
        # Handle final entity
        if current_entity and current_type:
            entity_text = ' '.join(current_entity)
            if current_type == 'EMAIL':
                stats['email_entities'].append(entity_text)
                stats['email_patterns'][get_email_pattern(entity_text)] += 1
            elif current_type == 'PHONE':
                stats['phone_entities'].append(entity_text)
                stats['phone_patterns'][get_phone_pattern(entity_text)] += 1
    
    return stats

def get_email_pattern(email_text):
    """Categorize email patterns"""
    email_clean = email_text.replace(' ', '').lower()
    
    if '@' not in email_clean:
        return 'INVALID-NO-AT'
    
    parts = email_clean.split('@')
    if len(parts) != 2:
        return 'INVALID-MULTIPLE-AT'
    
    local, domain = parts
    
    if not local or not domain:
        return 'INVALID-EMPTY-PARTS'
    
    if '.' not in domain:
        return 'INVALID-NO-DOT-IN-DOMAIN'
    
    # Common patterns
    if 'gmail' in domain:
        return 'GMAIL'
    elif 'hotmail' in domain or 'outlook' in domain:
        return 'HOTMAIL/OUTLOOK'
    elif 'yahoo' in domain:
        return 'YAHOO'
    elif domain.endswith('.com'):
        return 'DOT-COM'
    elif domain.endswith('.org'):
        return 'DOT-ORG'
    elif domain.endswith('.net'):
        return 'DOT-NET'
    else:
        return 'OTHER-DOMAIN'

def get_phone_pattern(phone_text):
    """Categorize phone patterns"""
    phone_clean = re.sub(r'[^\d]', '', phone_text)
    
    if len(phone_clean) < 7:
        return 'TOO-SHORT'
    elif len(phone_clean) == 7:
        return 'LOCAL-7-DIGIT'
    elif len(phone_clean) == 10:
        return 'US-10-DIGIT'
    elif len(phone_clean) == 11:
        return 'US-11-DIGIT'
    elif len(phone_clean) > 11:
        return 'TOO-LONG'
    else:
        return f'{len(phone_clean)}-DIGITS'

def print_analysis(stats):
    """Print detailed analysis"""
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total examples analyzed: {stats['total_examples']}")
    print(f"   Examples with emails: {stats['email_examples']}")
    print(f"   Examples with phones: {stats['phone_examples']}")
    print(f"   Examples with both: {stats['both_examples']}")
    
    print(f"\n📧 EMAIL ANALYSIS:")
    print(f"   Total email entities: {len(stats['email_entities'])}")
    print(f"   Top email patterns:")
    for pattern, count in stats['email_patterns'].most_common(5):
        print(f"     {pattern}: {count}")
    
    print(f"\n📱 PHONE ANALYSIS:")
    print(f"   Total phone entities: {len(stats['phone_entities'])}")
    print(f"   Top phone patterns:")
    for pattern, count in stats['phone_patterns'].most_common(5):
        print(f"     {pattern}: {count}")
    
    print(f"\n🔴 BIO TAGGING ISSUES:")
    print(f"   Total issues found: {len(stats['bio_issues'])}")
    for issue in stats['bio_issues'][:5]:  # Show first 5 issues
        print(f"     {issue['issue']} at position {issue['position']}")
        print(f"     Context: {issue['context']}")
    
    print(f"\n📝 EMAIL SAMPLES:")
    for i, email in enumerate(stats['email_entities'][:10]):
        print(f"   {i+1:2d}. {email}")
    
    print(f"\n📞 PHONE SAMPLES:")
    for i, phone in enumerate(stats['phone_entities'][:10]):
        print(f"   {i+1:2d}. {phone}")
    
    # Quality assessment
    email_valid_rate = 0
    if stats['email_entities']:
        valid_emails = sum(1 for email in stats['email_entities'] 
                          if '@' in email.replace(' ', '') and '.' in email.replace(' ', ''))
        email_valid_rate = valid_emails / len(stats['email_entities']) * 100
    
    phone_valid_rate = 0
    if stats['phone_entities']:
        valid_phones = sum(1 for phone in stats['phone_entities'] 
                          if 7 <= len(re.sub(r'[^\d]', '', phone)) <= 11)
        phone_valid_rate = valid_phones / len(stats['phone_entities']) * 100
    
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print(f"   Email validity rate: {email_valid_rate:.1f}%")
    print(f"   Phone validity rate: {phone_valid_rate:.1f}%")
    print(f"   BIO tagging error rate: {len(stats['bio_issues']) / stats['total_examples'] * 100:.1f}%")
    
    overall_quality = (email_valid_rate + phone_valid_rate) / 2
    if overall_quality >= 90:
        print(f"   Overall quality: 🟢 EXCELLENT ({overall_quality:.1f}%)")
    elif overall_quality >= 75:
        print(f"   Overall quality: 🟡 GOOD ({overall_quality:.1f}%)")
    elif overall_quality >= 50:
        print(f"   Overall quality: 🟠 FAIR ({overall_quality:.1f}%)")
    else:
        print(f"   Overall quality: 🔴 POOR ({overall_quality:.1f}%)")

if __name__ == "__main__":
    data = load_census_data()
    if data:
        stats = analyze_email_phone_quality(data, sample_size=100)
        print_analysis(stats)
    else:
        print("❌ No census data found. Run extract_census_data.py first.") 